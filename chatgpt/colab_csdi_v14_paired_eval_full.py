"""Google Colab -- CSDI conditionnel pour l'imputation Enedis.

Le script entraîne un modèle de diffusion PyTorch par repeat sur le protocole
apparié 10 x 10 de ``colab_helix_v14_paired_eval_full.py``. Il est autonome :
seuls les deux CSV du challenge sont requis. Un checkpoint JSON TabPFN/HELIX
peut être fourni pour ajouter une comparaison strictement appariée.

Entrées attendues dans DATA_DIR : X_train_78VdSWL.csv et y_train_u0UkKEh.csv.
Activez un GPU Colab et, pour survivre aux déconnexions, placez OUT_DIR sur
Google Drive. Les valeurs observées ne sont jamais remplacées.
"""

import copy
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# Configuration Colab
# -----------------------------------------------------------------------------
DATA_DIR = "/content/data"
OUT_DIR = "/content/csdi_enedis"  # pointez vers Drive si nécessaire
X_TRAIN_FILE = "X_train_78VdSWL.csv"
Y_TRAIN_FILE = "y_train_u0UkKEh.csv"
TABPFN_PROGRESS_PATH = None  # checkpoint du script HELIX, facultatif

SEED = 42
N_REPEATS = 10
N_TARGETS = 10
N_CALIBRATION = 10
TOTAL_FEATURES = 64             # 10 cibles + 10 calibration + 44 donneurs
WINDOW_LEN = 96
STRIDE = 48
TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 8
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0
VALIDATE_EVERY = 5
PATIENCE_VALIDATIONS = 8
N_DIFFUSION_SAMPLES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class DiffusionConfig:
    residual_layers: int = 4
    channels: int = 64
    n_heads: int = 8
    time_embedding_dim: int = 128
    feature_embedding_dim: int = 16
    diffusion_embedding_dim: int = 128
    diffusion_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.5


CFG = DiffusionConfig()
RUN_TAG = "csdi_v14_paired_10x10_w96_f64_s5"
RUN_DIR = Path(OUT_DIR) / RUN_TAG
PLOTS_DIR = RUN_DIR / "plots"
PROGRESS_PATH = RUN_DIR / "progress.json"
RESULTS_PATH = RUN_DIR / "results.json"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(payload, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def robust_scale(values):
    center = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - center), axis=0)
    scale = np.maximum(1.4826 * mad, 0.02 * (np.abs(center) + 1.0))
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return (values - center) / scale, center, scale


def run_lengths(mask):
    out = np.zeros(len(mask), dtype=np.int64)
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        out[i:j] = j - i
        i = j
    return out


def empirical_gap_lengths(frame):
    lengths = []
    for col in frame.columns:
        missing = frame[col].isna().to_numpy()
        runs = run_lengths(missing)
        lengths.extend(runs[missing].tolist())
    lengths = np.asarray(lengths, dtype=np.int64)
    return lengths[lengths > 0] if len(lengths) else np.array([1, 2, 4, 8, 16])


def select_donors(X, target_cols, donor_cols, count):
    """Round-robin des donneurs les plus corrélés à chaque cible observée."""
    donor_values = X[donor_cols].to_numpy(np.float32)
    med = np.nanmedian(donor_values, axis=0)
    donor_values = np.where(np.isfinite(donor_values), donor_values, med)
    rankings = []
    for col in target_cols:
        target = X[col].to_numpy(np.float32)
        obs = np.isfinite(target)
        if obs.sum() < 8:
            rankings.append([])
            continue
        t = target[obs]
        pool = donor_values[obs]
        tc = t - t.mean()
        pc = pool - pool.mean(axis=0, keepdims=True)
        corr = (pc.T @ tc) / (np.sqrt((pc * pc).sum(0) * (tc * tc).sum()) + 1e-8)
        rankings.append(np.argsort(-np.nan_to_num(corr, nan=-1.0)).tolist())
    selected, seen, rank = [], set(), 0
    while len(selected) < count and any(rank < len(x) for x in rankings):
        for order in rankings:
            if rank < len(order) and order[rank] not in seen:
                seen.add(order[rank])
                selected.append(donor_cols[order[rank]])
                if len(selected) == count:
                    break
        rank += 1
    return selected


def curve_profiles(X, columns):
    rows = []
    for col in columns:
        x = X[col].to_numpy(float)
        finite = x[np.isfinite(x)]
        missing = ~np.isfinite(x)
        runs = run_lengths(missing)
        rows.append([
            np.median(finite) if len(finite) else 0.0,
            np.std(finite) if len(finite) else 0.0,
            missing.mean(),
            runs[missing].mean() if missing.any() else 0.0,
        ])
    z = np.asarray(rows)
    med = np.median(z, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(z - med), axis=0), 1e-6)
    return pd.DataFrame((z - med) / scale, index=columns)


def choose_calibration(X, targets, dev_cols, count):
    profiles = curve_profiles(X, list(targets) + list(dev_cols))
    centroid = profiles.loc[targets].to_numpy().mean(0)
    return sorted(dev_cols, key=lambda c: float(((profiles.loc[c] - centroid) ** 2).sum()))[:count]


def window_starts(length, window, stride):
    starts = list(range(0, max(length - window + 1, 1), stride))
    final = max(length - window, 0)
    return sorted(set(starts + [final]))


def take_window(values, start, length):
    block = values[start:start + length]
    if len(block) == length:
        return block
    out = np.full((length, values.shape[1]), np.nan, dtype=values.dtype)
    out[:len(block)] = block
    return out


class EnedisWindowDataset(Dataset):
    def __init__(self, scaled, starts, gap_lengths, training, truth=None):
        self.scaled = scaled.astype(np.float32)
        self.starts = starts
        self.gaps = gap_lengths
        self.training = training
        self.truth = None if truth is None else truth.astype(np.float32)

    def __len__(self):
        return len(self.starts)

    def _training_condition(self, observed):
        cond = observed.copy()
        rng = np.random
        for feature in range(cond.shape[1]):
            candidates = np.flatnonzero(cond[:, feature])
            if not len(candidates):
                continue
            # 80 % blocs réalistes, 20 % points dispersés.
            if rng.rand() < 0.8:
                length = int(rng.choice(self.gaps))
                length = max(1, min(length, cond.shape[0] // 2))
                start = rng.randint(0, max(cond.shape[0] - length + 1, 1))
                cond[start:start + length, feature] = False
            else:
                n = max(1, int(0.1 * len(candidates)))
                cond[rng.choice(candidates, n, replace=False), feature] = False
        return cond

    def __getitem__(self, idx):
        start = self.starts[idx]
        x = take_window(self.scaled, start, WINDOW_LEN)
        observed = np.isfinite(x)
        clean = np.nan_to_num(x, nan=0.0)
        if self.training:
            cond = self._training_condition(observed)
            target = observed & ~cond
        else:
            truth = take_window(self.truth, start, WINDOW_LEN)
            truth_mask = np.isfinite(truth) & ~observed
            clean = np.where(truth_mask, truth, clean)
            cond, target = observed, truth_mask
        return {
            "values": torch.from_numpy(clean.T.copy()),
            "observed_mask": torch.from_numpy(observed.T.copy()),
            "cond_mask": torch.from_numpy(cond.T.copy()),
            "target_mask": torch.from_numpy(target.T.copy()),
            "time": torch.arange(WINDOW_LEN, dtype=torch.float32),
            "start": start,
        }


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        half = self.size // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / max(half - 1, 1))
        phase = x.float().unsqueeze(-1) * freq
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, cfg, side_dim):
        super().__init__()
        c = cfg.channels
        self.diffusion_proj = nn.Linear(cfg.diffusion_embedding_dim, c)
        self.side_proj = nn.Conv1d(side_dim, 2 * c, 1)
        self.mid_proj = nn.Conv1d(c, 2 * c, 1)
        self.out_proj = nn.Conv1d(c, 2 * c, 1)
        temporal = nn.TransformerEncoderLayer(c, cfg.n_heads, 4 * c, batch_first=True, activation="gelu")
        feature = nn.TransformerEncoderLayer(c, cfg.n_heads, 4 * c, batch_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(temporal, 1)
        self.feature = nn.TransformerEncoder(feature, 1)

    def forward(self, x, side, diff_emb):
        b, c, k, length = x.shape
        y = x + self.diffusion_proj(diff_emb)[:, :, None, None]
        y = y.permute(0, 2, 3, 1).reshape(b * k, length, c)
        y = self.temporal(y).reshape(b, k, length, c)
        y = y.permute(0, 2, 1, 3).reshape(b * length, k, c)
        y = self.feature(y).reshape(b, length, k, c).permute(0, 3, 2, 1)
        flat = y.reshape(b, c, k * length)
        h = self.mid_proj(flat) + self.side_proj(side.reshape(b, side.shape[1], k * length))
        gate, filt = h.chunk(2, dim=1)
        h = torch.sigmoid(gate) * torch.tanh(filt)
        residual, skip = self.out_proj(h).chunk(2, dim=1)
        residual = residual.reshape(b, c, k, length)
        skip = skip.reshape(b, c, k, length)
        return (x + residual) / math.sqrt(2.0), skip


class CSDI(nn.Module):
    def __init__(self, n_features, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_features = n_features
        self.time_embed = SinusoidalEmbedding(cfg.time_embedding_dim)
        self.diff_embed = SinusoidalEmbedding(cfg.diffusion_embedding_dim)
        self.feature_embed = nn.Embedding(n_features, cfg.feature_embedding_dim)
        side_dim = cfg.time_embedding_dim + cfg.feature_embedding_dim + 1
        self.input_proj = nn.Conv1d(2, cfg.channels, 1)
        self.blocks = nn.ModuleList([ResidualBlock(cfg, side_dim) for _ in range(cfg.residual_layers)])
        self.output1 = nn.Conv1d(cfg.channels, cfg.channels, 1)
        self.output2 = nn.Conv1d(cfg.channels, 1, 1)
        nn.init.zeros_(self.output2.weight)
        betas = torch.linspace(math.sqrt(cfg.beta_start), math.sqrt(cfg.beta_end), cfg.diffusion_steps) ** 2
        alphas_hat = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_hat", alphas_hat)
        self.register_buffer("alphas", torch.cumprod(alphas_hat, dim=0))

    def side_info(self, times, cond_mask):
        b, k, length = cond_mask.shape
        te = self.time_embed(times).unsqueeze(2).expand(-1, -1, k, -1)
        ids = torch.arange(k, device=cond_mask.device)
        fe = self.feature_embed(ids)[None, None].expand(b, length, -1, -1)
        return torch.cat([te, fe, cond_mask.permute(0, 2, 1).unsqueeze(-1)], -1).permute(0, 3, 2, 1)

    def denoise(self, noisy, values, cond_mask, step):
        cond = values * cond_mask
        model_input = torch.stack([cond, noisy * (~cond_mask)], dim=1)
        b, _, k, length = model_input.shape
        x = torch.relu(self.input_proj(model_input.reshape(b, 2, k * length))).reshape(b, -1, k, length)
        side = self.side_info(torch.arange(length, device=x.device)[None].expand(b, -1), cond_mask.float())
        de = self.diff_embed(step)
        skips = []
        for block in self.blocks:
            x, skip = block(x, side, de)
            skips.append(skip)
        x = torch.stack(skips).sum(0) / math.sqrt(len(skips))
        x = torch.relu(self.output1(x.reshape(b, self.cfg.channels, k * length)))
        return self.output2(x).reshape(b, k, length)

    def loss(self, batch):
        values = batch["values"]
        cond = batch["cond_mask"].bool()
        target = batch["target_mask"].float()
        b = len(values)
        step = torch.randint(self.cfg.diffusion_steps, (b,), device=values.device)
        alpha = self.alphas[step][:, None, None]
        noise = torch.randn_like(values)
        noisy = alpha.sqrt() * values + (1 - alpha).sqrt() * noise
        predicted = self.denoise(noisy, values, cond, step)
        return (((predicted - noise) ** 2) * target).sum() / target.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, values, cond_mask, n_samples):
        outputs = []
        for _ in range(n_samples):
            current = torch.randn_like(values)
            for t in range(self.cfg.diffusion_steps - 1, -1, -1):
                step = torch.full((len(values),), t, device=values.device, dtype=torch.long)
                eps = self.denoise(current, values, cond_mask.bool(), step)
                current = (current - self.betas[t] / torch.sqrt(1 - self.alphas[t]) * eps) / torch.sqrt(self.alphas_hat[t])
                if t:
                    sigma = torch.sqrt((1 - self.alphas[t - 1]) / (1 - self.alphas[t]) * self.betas[t])
                    current += sigma * torch.randn_like(current)
                current = torch.where(cond_mask.bool(), values, current)
            outputs.append(current)
        return torch.stack(outputs, dim=1)


def to_device(batch):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def validation_mae(model, loader, scales, samples=1):
    model.eval()
    total, count = 0.0, 0.0
    scale = torch.as_tensor(scales, device=DEVICE)[None, :, None]
    for batch in loader:
        batch = to_device(batch)
        generated = model.sample(batch["values"], batch["cond_mask"], samples).median(1).values
        mask = batch["target_mask"].float()
        total += (torch.abs(generated - batch["values"]) * scale * mask).sum().item()
        count += mask.sum().item()
    return total / max(count, 1.0)


def train_model(model, train_loader, val_loader, scales, repeat):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    best_state, best_mae, stale, history = None, float("inf"), 0, []
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = to_device(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
                loss = model.loss(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        val_mae = None
        if epoch == 1 or epoch % VALIDATE_EVERY == 0:
            val_mae = validation_mae(model, val_loader, scales, samples=1)
            if val_mae < best_mae:
                best_mae, stale = val_mae, 0
                best_state = copy.deepcopy(model.state_dict())
                torch.save({"model": best_state, "epoch": epoch, "mae_w": best_mae}, CHECKPOINT_DIR / f"repeat_{repeat}.pt")
            else:
                stale += 1
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_mae_w": val_mae})
        suffix = f", val MAE={val_mae:.2f} W" if val_mae is not None else ""
        print(f"  epoch {epoch:03d}/{TRAIN_EPOCHS}: diffusion MSE={np.mean(losses):.6f}{suffix}", flush=True)
        if stale >= PATIENCE_VALIDATIONS:
            print(f"  early stopping: {stale} validations sans amélioration", flush=True)
            break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history, best_mae


def taper(length):
    center = (length - 1) / 2
    return 0.15 + 0.85 * (1 - np.abs(np.arange(length) - center) / max(center, 1))


@torch.no_grad()
def impute_full(model, scaled, starts):
    model.eval()
    total = np.zeros_like(scaled, dtype=np.float64)
    weights = np.zeros_like(scaled, dtype=np.float64)
    quantile_low = np.zeros_like(scaled, dtype=np.float64)
    quantile_high = np.zeros_like(scaled, dtype=np.float64)
    tw = taper(WINDOW_LEN)[:, None]
    for start in starts:
        x = take_window(scaled, start, WINDOW_LEN).astype(np.float32)
        observed = np.isfinite(x)
        values = torch.from_numpy(np.nan_to_num(x).T[None]).to(DEVICE)
        mask = torch.from_numpy(observed.T[None]).to(DEVICE)
        samples = model.sample(values, mask, N_DIFFUSION_SAMPLES)[0].cpu().numpy().transpose(0, 2, 1)
        median = np.median(samples, axis=0)
        low, high = np.quantile(samples, [0.1, 0.9], axis=0)
        valid_len = min(WINDOW_LEN, len(scaled) - start)
        total[start:start + valid_len] += median[:valid_len] * tw[:valid_len]
        quantile_low[start:start + valid_len] += low[:valid_len] * tw[:valid_len]
        quantile_high[start:start + valid_len] += high[:valid_len] * tw[:valid_len]
        weights[start:start + valid_len] += tw[:valid_len]
    denom = np.maximum(weights, 1e-9)
    pred = total / denom
    pred[np.isfinite(scaled)] = scaled[np.isfinite(scaled)]
    return pred, quantile_low / denom, quantile_high / denom


def plot_history(history, path, repeat):
    epochs = [x["epoch"] for x in history]
    losses = [x["loss"] for x in history]
    val = [(x["epoch"], x["val_mae_w"]) for x in history if x["val_mae_w"] is not None]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, losses, color="#4C72B0", label="diffusion MSE")
    ax1.set_xlabel("époque"); ax1.set_ylabel("MSE bruit")
    if val:
        ax2 = ax1.twinx()
        ax2.plot(*zip(*val), color="#DD8452", marker="o", label="MAE calibration")
        ax2.set_ylabel("MAE (W)")
    fig.suptitle(f"CSDI Enedis -- repeat {repeat + 1}")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def load_tabpfn_reference(path):
    if not path:
        return {}
    with open(path) as handle:
        return json.load(handle).get("repeats", {})


def main():
    for directory in (RUN_DIR, PLOTS_DIR, CHECKPOINT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    print(f"device={DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""), flush=True)
    X = pd.read_csv(Path(DATA_DIR) / X_TRAIN_FILE, index_col=0)
    y = pd.read_csv(Path(DATA_DIR) / Y_TRAIN_FILE, index_col=0)
    donors = [c for c in X.columns if not str(c).startswith("holed_")]
    holed = list(y.columns)
    gap_lengths = empirical_gap_lengths(X[holed])
    tabpfn_ref = load_tabpfn_reference(TABPFN_PROGRESS_PATH)
    signature = {
        "schema": 1, "seed": SEED, "repeats": N_REPEATS, "targets": N_TARGETS,
        "calibration": N_CALIBRATION, "features": TOTAL_FEATURES,
        "window": WINDOW_LEN, "stride": STRIDE, "samples": N_DIFFUSION_SAMPLES,
        "diffusion": asdict(CFG),
    }
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as handle:
            progress = json.load(handle)
        if progress["signature"] != signature:
            raise RuntimeError("Checkpoint CSDI incompatible avec la configuration courante")
    else:
        progress = {"signature": signature, "repeats": {}}

    for repeat in range(N_REPEATS):
        key = str(repeat)
        if progress["repeats"].get(key, {}).get("complete"):
            print(f"repeat {repeat + 1}: déjà terminé", flush=True)
            continue
        seed = SEED + repeat
        seed_everything(seed)
        dev, val_full = train_test_split(holed, test_size=0.1, random_state=seed)
        targets = list(np.random.RandomState(seed).choice(val_full, N_TARGETS, replace=False))
        calibration = choose_calibration(X, targets, list(dev), N_CALIBRATION)
        selected_donors = select_donors(X, targets, donors, TOTAL_FEATURES - N_TARGETS - N_CALIBRATION)
        columns = targets + calibration + selected_donors
        raw = X[columns].to_numpy(float)
        scaled, center, scale = robust_scale(raw)
        # La sélection du modèle ne voit jamais y_true des dix cibles
        # officielles. Seules les courbes de calibration issues du dev split
        # fournissent une vérité interne.
        truth = np.full_like(scaled, np.nan)
        for j, col in enumerate(columns[N_TARGETS:N_TARGETS + N_CALIBRATION], start=N_TARGETS):
            truth[:, j] = (y[col].to_numpy(float) - center[j]) / scale[j]
        starts = window_starts(len(X), WINDOW_LEN, STRIDE)
        train_ds = EnedisWindowDataset(scaled, starts, gap_lengths, training=True)
        val_ds = EnedisWindowDataset(scaled, starts, gap_lengths, training=False, truth=truth)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, TRAIN_BATCH_SIZE, shuffle=True, generator=generator, num_workers=0)
        val_loader = DataLoader(val_ds, TRAIN_BATCH_SIZE, shuffle=False, num_workers=0)
        print(f"\nrepeat {repeat + 1}/{N_REPEATS}: T={len(X)}, K={len(columns)}, fenêtres={len(starts)}", flush=True)
        started = time.time()
        model = CSDI(len(columns), CFG).to(DEVICE)
        try:
            history, best_cal_mae = train_model(model, train_loader, val_loader, scale, repeat)
            imputed, q10, q90 = impute_full(model, scaled, starts)
        except torch.OutOfMemoryError:
            print("CUDA OOM: repeat non sauvegardé; réduisez batch/channels/window puis relancez", flush=True)
            del model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue
        errors, uncertainty = {}, {}
        for j, col in enumerate(targets):
            missing = ~np.isfinite(raw[:, j])
            target_truth = y[col].to_numpy(float)
            valid = missing & np.isfinite(target_truth)
            prediction = imputed[:, j] * scale[j] + center[j]
            errors[col] = np.abs(prediction[valid] - target_truth[valid]).tolist()
            uncertainty[col] = ((q90[:, j] - q10[:, j]) * scale[j])[valid].tolist()
        mae = float(np.concatenate([np.asarray(v) for v in errors.values()]).mean())
        rec = {
            "complete": True, "targets": targets, "calibration": calibration,
            "donors": selected_donors, "mae_w": mae, "best_calibration_mae_w": best_cal_mae,
            "errors": errors, "uncertainty_width_w": uncertainty,
            "seconds": time.time() - started,
        }
        reference = tabpfn_ref.get(key, {})
        if reference and set(targets) == set(reference.get("val_cols", [])):
            tab = reference.get("tabpfn", {})
            if all(c in tab for c in targets):
                rec["tabpfn_mae_w"] = float(np.concatenate([np.asarray(tab[c]) for c in targets]).mean())
        progress["repeats"][key] = rec
        atomic_json(progress, PROGRESS_PATH)
        plot_history(history, PLOTS_DIR / f"repeat_{repeat + 1}.png", repeat)
        print(f"repeat {repeat + 1}: CSDI MAE={mae:.2f} W, calibration best={best_cal_mae:.2f} W", flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    complete = [r for r in progress["repeats"].values() if r.get("complete")]
    report = {
        "signature": signature,
        "completed_repeats": len(complete),
        "mean_repeat_mae_w": float(np.mean([r["mae_w"] for r in complete])) if complete else None,
        "std_repeat_mae_w": float(np.std([r["mae_w"] for r in complete])) if complete else None,
        "repeats": complete,
    }
    atomic_json(report, RESULTS_PATH)
    print(json.dumps({k: v for k, v in report.items() if k != "repeats"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
