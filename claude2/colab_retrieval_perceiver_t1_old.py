#!/usr/bin/env python3
# %% [markdown]
# # Enedis — Retrieval-conditioned Perceiver + T1/TimeMixer imputer
#
# Script Colab autonome pour GPU 80 Go. Dans Colab :
# 1. Runtime > Change runtime type > GPU (A100/H100 recommandé).
# 2. Méthode recommandée avec `!python` : téléverser d'abord les CSV dans
#    une cellule Colab :
#      from google.colab import files
#      files.upload()
#    puis lancer :
#      !python /content/colab_retrieval_perceiver_t1.py --profile full --data-dir /content
# 3. Avec `%run /content/colab_retrieval_perceiver_t1.py --profile full`, le
#    script peut ouvrir lui-même la boîte de téléversement interactive.
#
# Le modèle n'utilise ni SAITS, ni PyPOTS, ni TabPFN. Il comporte :
# - un pré-entraînement auto-supervisé sur les 57 140 courbes complètes ;
# - un retrieval top-256 avec calibration affine sur les points observés ;
# - un Perceiver global auquel toutes les références participent ;
# - un décodeur temporel multi-échelle inspiré T1/TimeMixer ;
# - quatre experts et une prédiction résiduelle par rapport à PCHIP.

# %%
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _bootstrap_colab() -> None:
    """Installe uniquement les petites dépendances absentes de Colab."""
    try:
        import torch  # noqa: F401
        import pandas  # noqa: F401
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "numpy", "pandas", "scipy", "scikit-learn", "tqdm", "matplotlib",
        ])


_bootstrap_colab()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm


# %% [markdown]
# ## Configuration

# %%
@dataclass
class Config:
    seed: int = 42
    profile: str = "full"
    n_steps: int = 1057
    k_local: int = 256
    k_coarse: int = 2048
    embed_dim: int = 128
    temporal_channels: int = 32
    latent_dim: int = 128
    n_latents: int = 64
    n_heads: int = 8
    local_feature_channels: int = 8
    decoder_layers: int = 4
    dropout: float = 0.10
    pretrain_epochs: int = 60
    finetune_epochs: int = 60
    final_epochs: int = 60
    global_pretrain_steps: int = 200
    pretrain_batch_size: int = 128
    train_batch_size: int = 2
    grad_accum: int = 8
    pretrain_samples_per_epoch: int = 30000
    synthetic_samples_per_epoch: int = 6000
    supervised_repeat: int = 5
    lr_pretrain: float = 2e-3
    lr_finetune: float = 3e-4
    weight_decay: float = 1e-3
    patience: int = 7
    val_fraction: float = 0.20
    clip_grad: float = 1.0
    num_workers: int = 2
    use_test_time_adaptation: bool = True
    tta_steps: int = 4
    tta_lr: float = 5e-5
    exact_global_perceiver: bool = True

    @classmethod
    def for_profile(cls, name: str) -> "Config":
        cfg = cls(profile=name)
        if name == "smoke":
            cfg.k_local = 16
            cfg.k_coarse = 64
            cfg.embed_dim = 64
            cfg.temporal_channels = 16
            cfg.latent_dim = 64
            cfg.n_latents = 16
            cfg.n_heads = 4
            cfg.local_feature_channels = 4
            cfg.decoder_layers = 2
            cfg.pretrain_epochs = 1
            cfg.finetune_epochs = 2
            cfg.final_epochs = 1
            cfg.global_pretrain_steps = 3
            cfg.pretrain_batch_size = 32
            cfg.pretrain_samples_per_epoch = 256
            cfg.synthetic_samples_per_epoch = 128
            cfg.supervised_repeat = 1
            cfg.patience = 2
            cfg.use_test_time_adaptation = False
        return cfg


FILES = {
    "x_train": "X_train_78VdSWL.csv",
    "y_train": "y_train_u0UkKEh.csv",
    "x_test": "X_test_XKVc4no.csv",
    "sample": "new_output_sample.csv",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _has_all_csv(root: Path) -> bool:
    return all((root / name).is_file() for name in FILES.values())


def _interactive_colab_kernel_available() -> bool:
    """`files.upload()` exige le kernel IPython du notebook parent.

    `!python script.py` démarre un nouveau processus : google.colab y est
    importable, mais `get_ipython()` vaut None. Tester seulement l'import de
    google.colab provoquait donc le traceback observé dans Colab.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and getattr(shell, "kernel", None) is not None
    except Exception:
        return False


def _subprocess_upload_help(missing: Sequence[str], searched: Sequence[Path]) -> str:
    locations = ", ".join(str(p) for p in searched)
    return (
        "CSV du challenge introuvables.\n"
        f"Fichiers manquants : {', '.join(missing)}\n"
        f"Dossiers examinés : {locations}\n\n"
        "Ce script est lancé avec `!python`, donc `google.colab.files.upload()` "
        "ne peut pas ouvrir le sélecteur depuis ce sous-processus.\n"
        "Exécute d'abord cette cellule Colab :\n\n"
        "    from google.colab import files\n"
        "    files.upload()\n\n"
        "Puis relance :\n\n"
        "    !python /content/colab_retrieval_perceiver_t1.py "
        "--profile full --data-dir /content\n\n"
        "Alternative interactive :\n\n"
        "    %run /content/colab_retrieval_perceiver_t1.py --profile full\n"
    )


def upload_data_if_needed(data_dir: Optional[str]) -> Path:
    explicit = Path(data_dir).expanduser().resolve() if data_dir else None
    if explicit is not None:
        candidates = [explicit]
    elif Path("/content").exists():
        candidates = [Path("/content/enedis_data"), Path("/content")]
    else:
        candidates = [Path("data").resolve()]

    # Les CSV déjà téléversés dans /content sont découverts sans interaction.
    for candidate in candidates:
        if _has_all_csv(candidate):
            print(f"CSV détectés dans {candidate}")
            return candidate

    root = candidates[0]
    root.mkdir(parents=True, exist_ok=True)
    missing = [name for name in FILES.values() if not (root / name).is_file()]

    if not _interactive_colab_kernel_available():
        raise SystemExit(_subprocess_upload_help(missing, candidates))

    try:
        from google.colab import files  # type: ignore
    except ImportError as exc:
        raise FileNotFoundError(f"CSV absents dans {root}: {missing}") from exc

    print("Téléverse les CSV manquants du challenge :", ", ".join(missing))
    uploaded = files.upload()
    for filename, content in uploaded.items():
        (root / Path(filename).name).write_bytes(content)
    missing = [name for name in FILES.values() if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Fichiers toujours absents après upload: {missing}")
    return root


def load_frames(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Chargement des CSV…")
    frames = [pd.read_csv(root / FILES[k], index_col=0) for k in ("x_train", "y_train", "x_test", "sample")]
    x_train, y_train, x_test, sample = frames
    if len(x_train) != len(x_test) or not x_train.index.equals(x_test.index):
        raise ValueError("X_train et X_test doivent partager les mêmes 1 057 timestamps")
    if not y_train.index.equals(x_train.index) or not sample.index.equals(x_test.index):
        raise ValueError("Index temporels incompatibles")
    return x_train, y_train, x_test, sample


def complete_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not str(c).startswith("holed_")]


def robust_curve_normalize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise indépendamment chaque ligne (courbe), uniquement sur ses valeurs finies."""
    center = np.nanmedian(x, axis=1).astype(np.float32)
    mad = np.nanmedian(np.abs(x - center[:, None]), axis=1).astype(np.float32)
    scale = np.maximum(1.4826 * mad, 0.02 * (np.abs(center) + 1.0)).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-5)] = 1.0
    center[~np.isfinite(center)] = 0.0
    z = (x - center[:, None]) / scale[:, None]
    return z.astype(np.float32), center, scale


def baseline_pchip(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=True)
    obs = np.flatnonzero(np.isfinite(values))
    if len(obs) == 0:
        return np.zeros_like(values, dtype=np.float32)
    if len(obs) == 1:
        return np.full_like(values, values[obs[0]], dtype=np.float32)
    grid = np.arange(len(values))
    try:
        pred = PchipInterpolator(obs, values[obs], extrapolate=False)(grid)
    except ValueError:
        pred = np.interp(grid, obs, values[obs])
    linear = np.interp(grid, obs, values[obs])
    pred = np.where(np.isfinite(pred), pred, linear)
    pred[obs] = values[obs]
    return pred.astype(np.float32)


def gap_features(missing: np.ndarray) -> np.ndarray:
    """Longueur, position relative et distances aux bords pour chaque timestamp."""
    t = len(missing)
    out = np.zeros((t, 4), np.float32)
    i = 0
    while i < t:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j + 1 < t and missing[j + 1]:
            j += 1
        length = j - i + 1
        p = np.arange(length, dtype=np.float32)
        out[i:j + 1, 0] = np.log1p(length) / np.log1p(t)
        out[i:j + 1, 1] = p / max(length - 1, 1)
        out[i:j + 1, 2] = np.minimum(p + 1, length - p) / max(length, 1)
        out[i:j + 1, 3] = np.select([length <= 3, length <= 47, length <= 95], [0, 1, 2], default=3) / 3.0
        i = j + 1
    return out


def time_features(index: pd.Index) -> np.ndarray:
    dt = pd.to_datetime(index)
    hour = dt.hour.to_numpy() * 2 + dt.minute.to_numpy() / 30.0
    dow = dt.dayofweek.to_numpy()
    doy = dt.dayofyear.to_numpy()
    return np.stack([
        np.sin(2 * np.pi * hour / 48), np.cos(2 * np.pi * hour / 48),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        (dow >= 5).astype(np.float32),
    ], axis=1).astype(np.float32)


def observed_stats(x: np.ndarray) -> Tuple[float, float]:
    obs = x[np.isfinite(x)]
    if len(obs) == 0:
        return 0.0, 1.0
    center = float(np.median(obs))
    scale = max(float(1.4826 * np.median(np.abs(obs - center))), 0.02 * (abs(center) + 1.0), 1e-5)
    return center, scale


# %% [markdown]
# ## Masques challenge-like et pré-entraînement

# %%
def extract_gap_lengths(holed: pd.DataFrame) -> np.ndarray:
    lengths: List[int] = []
    for col in holed.columns:
        m = holed[col].isna().to_numpy()
        padded = np.r_[False, m, False]
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        lengths.extend((ends - starts).tolist())
    return np.asarray(lengths or [1, 2, 48], dtype=np.int32)


def sample_challenge_mask(rng: np.random.Generator, n_steps: int, gap_lengths: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_steps, dtype=bool)
    # Nombre de blocs proche de celui d'une target réelle, borné pour éviter les masques dégénérés.
    n_blocks = int(np.clip(rng.poisson(5) + 1, 1, 20))
    for _ in range(n_blocks):
        length = int(rng.choice(gap_lengths))
        length = min(max(length, 1), n_steps - 2)
        start = int(rng.integers(0, n_steps - length + 1))
        mask[start:start + length] = True
    if mask.mean() > 0.70:
        idx = rng.choice(np.flatnonzero(mask), size=int(0.70 * n_steps), replace=False)
        mask[:] = False
        mask[idx] = True
    return mask


class PretrainDataset(Dataset):
    def __init__(self, curves: np.ndarray, gap_lengths: np.ndarray, n_samples: int, seed: int):
        self.curves, self.gap_lengths, self.n_samples, self.seed = curves, gap_lengths, n_samples, seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx * 104729)
        row = self.curves[int(rng.integers(len(self.curves)))].copy()
        original_observed = np.isfinite(row)
        hide = sample_challenge_mask(rng, len(row), self.gap_lengths) & original_observed
        if not hide.any():
            hide[int(rng.choice(np.flatnonzero(original_observed)))] = True
        center, scale = observed_stats(np.where(hide, np.nan, row))
        truth = (np.nan_to_num(row, nan=center) - center) / scale
        visible = original_observed & ~hide
        value = np.where(visible, truth, 0.0).astype(np.float32)
        return (
            torch.from_numpy(value), torch.from_numpy(visible.astype(np.float32)),
            torch.from_numpy(truth.astype(np.float32)), torch.from_numpy(hide.astype(np.float32)),
        )


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.dw = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation, groups=channels)
        self.pw1 = nn.Conv1d(channels, channels * 4, 1)
        self.pw2 = nn.Conv1d(channels * 4, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw2(self.dropout(F.gelu(self.pw1(y))))
        return self.norm(x + self.dropout(y))


class SharedCurveEncoder(nn.Module):
    """Encodeur mask-aware partagé par toutes les courbes."""
    def __init__(self, cfg: Config):
        super().__init__()
        c = cfg.temporal_channels
        self.stem = nn.Conv1d(2, c, kernel_size=9, padding=4)
        self.blocks = nn.Sequential(*[
            ResidualTCNBlock(c, k, d, cfg.dropout)
            for k, d in ((7, 1), (11, 2), (15, 4), (23, 8))
        ])
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(c * 16, cfg.embed_dim), nn.LayerNorm(cfg.embed_dim))
        self.reconstruct = nn.Sequential(
            nn.Conv1d(c, c, 5, padding=2), nn.GELU(), nn.Conv1d(c, 1, 1),
        )

    def features(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(torch.stack([value, mask], dim=1)))

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(value, mask)
        return self.embedding(self.pool(feat)), self.reconstruct(feat).squeeze(1)


def masked_l1(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(truth)
    error = torch.where(valid, (pred - truth).abs(), torch.zeros_like(pred))
    return error.sum() / valid.sum().clamp_min(1)


def autocast_context(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def pretrain_encoder(cfg: Config, encoder: SharedCurveEncoder, curves: np.ndarray,
                     gap_lengths: np.ndarray, device: torch.device, work: Path) -> None:
    ckpt = work / "encoder_pretrained.pt"
    if ckpt.exists():
        encoder.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        print("Checkpoint encodeur chargé:", ckpt)
        return
    ds = PretrainDataset(curves, gap_lengths, cfg.pretrain_samples_per_epoch, cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.pretrain_batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0)
    opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr_pretrain, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=False)
    encoder.to(device)
    for epoch in range(cfg.pretrain_epochs):
        encoder.train()
        total, count = 0.0, 0
        bar = tqdm(loader, desc=f"prétrain {epoch + 1}/{cfg.pretrain_epochs}")
        for value, mask, truth, hide in bar:
            value, mask, truth, hide = [v.to(device, non_blocking=True) for v in (value, mask, truth, hide)]
            opt.zero_grad(set_to_none=True)
            with autocast_context(device):
                _, pred = encoder(value, mask)
                loss = masked_l1(pred, truth, hide)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(encoder.parameters(), cfg.clip_grad)
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
            count += 1
            bar.set_postfix(l1=total / count)
        torch.save(encoder.state_dict(), ckpt)


@torch.inference_mode()
def encode_reference_bank(cfg: Config, encoder: SharedCurveEncoder, curves_z: np.ndarray,
                          device: torch.device, work: Path) -> np.ndarray:
    cache = work / "reference_embeddings.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.shape == (len(curves_z), cfg.embed_dim):
            print("Embeddings chargés:", cache, emb.shape)
            return emb
    encoder.eval().to(device)
    chunks = []
    bs = 512 if device.type == "cuda" else 32
    for start in tqdm(range(0, len(curves_z), bs), desc="encodage des 57k références"):
        x = torch.from_numpy(curves_z[start:start + bs]).to(device)
        m = torch.isfinite(x).float()
        x = torch.nan_to_num(x)
        with autocast_context(device):
            e, _ = encoder(x, m)
        chunks.append(F.normalize(e.float(), dim=-1).cpu().numpy())
    emb = np.concatenate(chunks).astype(np.float32)
    np.save(cache, emb)
    return emb


# %% [markdown]
# ## Retrieval coarse-to-fine et cache anti-fuite

# %%
@torch.inference_mode()
def encode_targets(cfg: Config, encoder: SharedCurveEncoder, targets: np.ndarray,
                   device: torch.device) -> np.ndarray:
    rows = []
    for raw in targets:
        center, scale = observed_stats(raw)
        z = ((raw - center) / scale).astype(np.float32)
        x = torch.from_numpy(np.nan_to_num(z)[None]).to(device)
        m = torch.from_numpy(np.isfinite(z).astype(np.float32)[None]).to(device)
        with autocast_context(device):
            e, _ = encoder(x, m)
        rows.append(F.normalize(e.float(), dim=-1).cpu().numpy()[0])
    return np.asarray(rows, np.float32)


def coarse_neighbors(target_emb: np.ndarray, ref_emb: np.ndarray, k: int,
                     device: torch.device) -> np.ndarray:
    refs = torch.from_numpy(ref_emb).to(device)
    out = []
    batch = 64
    with torch.inference_mode():
        for start in tqdm(range(0, len(target_emb), batch), desc="retrieval coarse"):
            q = torch.from_numpy(target_emb[start:start + batch]).to(device)
            score = q @ refs.T
            out.append(torch.topk(score, k=min(k, len(ref_emb)), dim=1).indices.cpu().numpy())
    del refs
    return np.concatenate(out)


def affine_rerank_one(target: np.ndarray, ref_curves: np.ndarray, candidate_idx: np.ndarray,
                      k_local: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reranking exact sur points observés. Retourne indices et [D,rho,a,b]."""
    obs = np.isfinite(target)
    y = target[obs].astype(np.float64)
    x = ref_curves[candidate_idx][:, obs].astype(np.float64)
    finite = np.isfinite(x)
    x_mean = np.nanmean(x, axis=1)
    y_mean = float(y.mean()) if len(y) else 0.0
    xc = np.where(finite, x - x_mean[:, None], 0.0)
    yc = y - y_mean
    denom_x = np.sum(xc * xc, axis=1)
    cov = np.sum(xc * yc[None], axis=1)
    a = cov / np.maximum(denom_x, 1e-8)
    a = np.clip(a, -5.0, 5.0)
    b = y_mean - a * x_mean
    calibrated = a[:, None] * x + b[:, None]
    residual = np.abs(calibrated - y[None])
    mae = np.nanmean(residual, axis=1)
    rho = cov / np.sqrt(np.maximum(denom_x * np.sum(yc * yc), 1e-8))
    y_scale = max(1.4826 * np.median(np.abs(y - np.median(y))), 1.0)
    score = mae / y_scale + 0.25 * (1.0 - np.clip(rho, -1, 1))
    score[~np.isfinite(score)] = np.inf
    order = np.argsort(score)[:k_local]
    idx = candidate_idx[order]
    meta = np.stack([score[order], rho[order], a[order], b[order]], axis=1).astype(np.float32)
    return idx.astype(np.int32), meta


def build_retrieval_cache(name: str, cfg: Config, encoder: SharedCurveEncoder,
                          targets: np.ndarray, ref_curves: np.ndarray, ref_emb: np.ndarray,
                          device: torch.device, work: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = work / f"retrieval_{name}_k{cfg.k_local}_c{cfg.k_coarse}.npz"
    if path.exists():
        z = np.load(path)
        if len(z["indices"]) == len(targets):
            print("Cache retrieval chargé:", path)
            return z["indices"], z["meta"]
    target_emb = encode_targets(cfg, encoder, targets, device)
    candidates = coarse_neighbors(target_emb, ref_emb, cfg.k_coarse, device)
    indices, metadata = [], []
    for i in tqdm(range(len(targets)), desc=f"reranking affine {name}"):
        idx, meta = affine_rerank_one(targets[i], ref_curves, candidates[i], cfg.k_local)
        indices.append(idx)
        metadata.append(meta)
    indices_a, metadata_a = np.asarray(indices, np.int32), np.asarray(metadata, np.float32)
    np.savez_compressed(path, indices=indices_a, meta=metadata_a)
    return indices_a, metadata_a


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=0)
    sorted_v = np.take_along_axis(values, order, axis=0)
    sorted_w = np.take_along_axis(np.broadcast_to(weights[:, None], values.shape), order, axis=0)
    cdf = np.cumsum(sorted_w, axis=0)
    threshold = 0.5 * sorted_w.sum(axis=0, keepdims=True)
    pos = (cdf >= threshold).argmax(axis=0)
    return sorted_v[pos, np.arange(values.shape[1])]


def make_model_sample(raw_target: np.ndarray, truth: np.ndarray, ref_curves: np.ndarray,
                      ref_idx: np.ndarray, meta: np.ndarray, tf: np.ndarray) -> Dict[str, np.ndarray]:
    observed = np.isfinite(raw_target)
    missing = ~observed
    center, scale = observed_stats(raw_target)
    base = baseline_pchip(raw_target)
    target_z = np.where(observed, (raw_target - center) / scale, 0.0).astype(np.float32)
    base_z = ((base - center) / scale).astype(np.float32)
    refs = ref_curves[ref_idx].astype(np.float32)
    # Calibration estimée avant toute utilisation des timestamps cachés.
    calibrated = meta[:, 2, None] * refs + meta[:, 3, None]
    refs_z = ((calibrated - center) / scale).astype(np.float32)
    refs_z = np.nan_to_num(refs_z)
    weights = np.exp(-np.clip(meta[:, 0], 0, 20))
    weights /= max(weights.sum(), 1e-8)
    neighbor = weighted_median(refs_z, weights).astype(np.float32)
    truth_z = ((truth - center) / scale).astype(np.float32)
    model_meta = meta.copy()
    # a est sans dimension ; b doit être exprimé dans l'échelle robuste de la cible.
    model_meta[:, 3] = (model_meta[:, 3] - center) / scale
    return {
        "target": target_z, "observed": observed.astype(np.float32), "baseline": base_z,
        "neighbors": refs_z, "neighbor_median": neighbor, "metadata": model_meta.astype(np.float32),
        "time": tf, "gap": gap_features(missing), "truth": truth_z,
        "loss_mask": (missing & np.isfinite(truth)).astype(np.float32),
        "center": np.float32(center), "scale": np.float32(scale),
    }


class RetrievalDataset(Dataset):
    def __init__(self, targets: np.ndarray, truths: np.ndarray, refs: np.ndarray,
                 indices: np.ndarray, meta: np.ndarray, tf: np.ndarray,
                 selected: Sequence[int], repeat: int = 1):
        self.targets, self.truths, self.refs = targets, truths, refs
        self.indices, self.meta, self.tf = indices, meta, tf
        self.selected = np.asarray(selected, dtype=np.int64)
        self.repeat = repeat

    def __len__(self) -> int:
        return len(self.selected) * self.repeat

    def __getitem__(self, item: int):
        i = int(self.selected[item % len(self.selected)])
        sample = make_model_sample(self.targets[i], self.truths[i], self.refs,
                                   self.indices[i], self.meta[i], self.tf)
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v)
                for k, v in sample.items()}


class SyntheticRetrievalDataset(Dataset):
    """Transforme des références complètes en problèmes d'imputation réalistes.

    Le reranking mis en cache n'utilise que les points laissés visibles et la
    cible elle-même est absente des candidats.
    """
    def __init__(self, refs: np.ndarray, target_ids: np.ndarray, masked_targets: np.ndarray,
                 indices: np.ndarray, metadata: np.ndarray, tf: np.ndarray):
        self.refs, self.target_ids, self.masked_targets = refs, target_ids, masked_targets
        self.indices, self.metadata, self.tf = indices, metadata, tf

    def __len__(self) -> int:
        return len(self.target_ids)

    def __getitem__(self, item: int):
        target_id = int(self.target_ids[item])
        truth = self.refs[target_id].copy()
        sample = make_model_sample(self.masked_targets[item], truth, self.refs,
                                   self.indices[item], self.metadata[item], self.tf)
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v)
                for k, v in sample.items()}


def build_synthetic_cache(cfg: Config, refs: np.ndarray, target_ids: np.ndarray,
                          candidates: np.ndarray, gap_lengths: np.ndarray,
                          work: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = work / f"synthetic_training_{len(target_ids)}_k{cfg.k_local}.npz"
    if path.exists():
        z = np.load(path)
        return z["masked"], z["indices"], z["metadata"]
    masked, indices, metadata = [], [], []
    for item, target_id in enumerate(tqdm(target_ids, desc="cache synthétique challenge-like")):
        truth = refs[int(target_id)]
        rng = np.random.default_rng(cfg.seed + 9000 + item * 13007)
        hide = sample_challenge_mask(rng, len(truth), gap_lengths) & np.isfinite(truth)
        raw = truth.copy()
        raw[hide] = np.nan
        candidate = candidates[item]
        candidate = candidate[candidate != int(target_id)]
        idx, meta = affine_rerank_one(raw, refs, candidate, cfg.k_local)
        masked.append(raw)
        indices.append(idx)
        metadata.append(meta)
    arrays = (np.asarray(masked, np.float32), np.asarray(indices, np.int32),
              np.asarray(metadata, np.float32))
    np.savez_compressed(path, masked=arrays[0], indices=arrays[1], metadata=arrays[2])
    return arrays


# %% [markdown]
# ## Perceiver global, T1 local et décodeur TimeMixer

# %%
class PerceiverGlobal(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(cfg.n_latents, cfg.latent_dim) * 0.02)
        self.ref_proj = nn.Linear(cfg.embed_dim, cfg.latent_dim)
        self.cross = nn.MultiheadAttention(cfg.latent_dim, cfg.n_heads, cfg.dropout, batch_first=True)
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(cfg.latent_dim, cfg.n_heads, cfg.dropout, batch_first=True)
            for _ in range(2)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.latent_dim) for _ in range(3)])
        self.ff = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim * 4), nn.GELU(),
                                nn.Dropout(cfg.dropout), nn.Linear(cfg.latent_dim * 4, cfg.latent_dim))

    def forward(self, reference_embeddings: torch.Tensor) -> torch.Tensor:
        refs = self.ref_proj(reference_embeddings).unsqueeze(0)
        q = self.latents.unsqueeze(0)
        x, _ = self.cross(q, refs, refs, need_weights=False)
        x = self.norms[0](q + x)
        for i, attn in enumerate(self.self_attn):
            y, _ = attn(x, x, x, need_weights=False)
            x = self.norms[i + 1](x + y + self.ff(x))
        return x


class T1TargetAttention(nn.Module):
    """Channel-head binding avec requête cible uniquement : O(K*C*L), pas O(K²)."""
    def __init__(self, cfg: Config):
        super().__init__()
        c = cfg.local_feature_channels
        self.c = c
        self.stem = nn.Conv1d(2, c, kernel_size=9, stride=4, padding=4)
        self.q_small = nn.Conv1d(c, c, 5, padding=2, groups=c)
        self.q_large = nn.Conv1d(c, c, 17, padding=8, groups=c)
        self.k_small = nn.Conv1d(c, c, 5, padding=2, groups=c)
        self.k_large = nn.Conv1d(c, c, 17, padding=8, groups=c)
        self.v_small = nn.Conv1d(c, c, 5, padding=2, groups=c)
        self.v_large = nn.Conv1d(c, c, 17, padding=8, groups=c)
        self.meta = nn.Sequential(nn.Linear(4, c), nn.Tanh())
        self.out = nn.Conv1d(c, cfg.latent_dim, 1)

    def forward(self, target: torch.Tensor, observed: torch.Tensor,
                neighbors: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        b, k, t = neighbors.shape
        tgt = self.stem(torch.stack([target, observed], dim=1))
        n_in = torch.stack([neighbors, torch.ones_like(neighbors)], dim=2).reshape(b * k, 2, t)
        nbr = self.stem(n_in).reshape(b, k, self.c, -1)
        q = self.q_small(tgt) + self.q_large(tgt)                       # B,C,L
        flat = nbr.reshape(b * k, self.c, -1)
        keys = (self.k_small(flat) + self.k_large(flat)).reshape_as(nbr)
        vals = (self.v_small(flat) + self.v_large(flat)).reshape_as(nbr)
        # Chaque canal CNN correspond à une tête : score indépendant par canal.
        logits = torch.einsum("bcl,bkcl->bkc", q, keys) / math.sqrt(q.shape[-1])
        logits = logits + self.meta(metadata)
        attn = logits.softmax(dim=1)
        fused = torch.einsum("bkc,bkcl->bcl", attn, vals)
        return self.out(fused)


class MultiScaleMixer(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.scales = (1, 2, 4, 8, 24, 48, 96, 336)
        self.mix = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1, groups=channels),
                          nn.Conv1d(channels, channels, 1), nn.GELU())
            for _ in self.scales
        ])
        self.fuse = nn.Sequential(nn.Conv1d(channels * len(self.scales), channels * 2, 1),
                                  nn.GELU(), nn.Dropout(dropout), nn.Conv1d(channels * 2, channels, 1))
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[-1]
        levels = []
        for scale, block in zip(self.scales, self.mix):
            if scale == 1:
                y = x
            else:
                y = F.avg_pool1d(x, kernel_size=min(scale, t), stride=min(scale, t), ceil_mode=True)
            y = block(y)
            levels.append(F.interpolate(y, size=t, mode="linear", align_corners=False))
        return self.norm(x + self.fuse(torch.cat(levels, dim=1)))


class RetrievalPerceiverImputer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.latent_dim
        self.global_pool = PerceiverGlobal(cfg)
        self.local_t1 = T1TargetAttention(cfg)
        self.target_stem = nn.Conv1d(15, d, 9, padding=4)
        self.local_up = nn.Conv1d(d, d, 1)
        self.global_cross = nn.MultiheadAttention(d, cfg.n_heads, cfg.dropout, batch_first=True)
        self.mixers = nn.ModuleList([MultiScaleMixer(d, cfg.dropout) for _ in range(cfg.decoder_layers)])
        self.experts = nn.ModuleList([nn.Conv1d(d, 1, 1) for _ in range(4)])
        self.gate = nn.Sequential(nn.Conv1d(d + 4, d // 2, 1), nn.GELU(), nn.Conv1d(d // 2, 4, 1))

    def global_tokens(self, reference_embeddings: torch.Tensor) -> torch.Tensor:
        return self.global_pool(reference_embeddings)

    def forward(self, batch: Dict[str, torch.Tensor], reference_embeddings: torch.Tensor,
                cached_global: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        target, observed, baseline = batch["target"], batch["observed"], batch["baseline"]
        local = self.local_t1(target, observed, batch["neighbors"], batch["metadata"])
        local = F.interpolate(self.local_up(local), size=target.shape[-1], mode="linear", align_corners=False)
        inp = torch.cat([
            target[:, None], observed[:, None], baseline[:, None], batch["neighbor_median"][:, None],
            batch["time"].transpose(1, 2), batch["gap"].transpose(1, 2),
        ], dim=1)
        x = self.target_stem(inp) + local
        global_tokens = cached_global if cached_global is not None else self.global_tokens(reference_embeddings)
        g = global_tokens.expand(x.shape[0], -1, -1)
        q = x.transpose(1, 2)
        cross, _ = self.global_cross(q, g, g, need_weights=False)
        x = (q + cross).transpose(1, 2)
        for mixer in self.mixers:
            x = mixer(x)
        expert = torch.cat([head(x) for head in self.experts], dim=1)
        gates = self.gate(torch.cat([x, batch["gap"].transpose(1, 2)], dim=1)).softmax(dim=1)
        residual = (expert * gates).sum(dim=1)
        return baseline + residual, gates


def pretrain_global_perceiver(cfg: Config, model: RetrievalPerceiverImputer,
                              ref_emb: torch.Tensor, device: torch.device, work: Path) -> None:
    """Apprend le bottleneck global comme autoencodeur d'ensemble.

    Les mini-ensembles changent à chaque pas. Après cet apprentissage, un passage
    final avec les 57 140 embeddings fabrique le contexte réellement utilisé.
    """
    path = work / "global_perceiver_pretrained.pt"
    if path.exists():
        model.global_pool.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        print("Checkpoint Perceiver global chargé:", path)
        return
    model.global_pool.to(device)
    ref_emb = ref_emb.to(device)
    decoder = nn.MultiheadAttention(cfg.latent_dim, cfg.n_heads, cfg.dropout, batch_first=True).to(device)
    query_proj = nn.Linear(cfg.embed_dim, cfg.latent_dim).to(device)
    opt = torch.optim.AdamW(list(model.global_pool.parameters()) + list(decoder.parameters()) +
                            list(query_proj.parameters()), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    set_size = min(8192, len(ref_emb))
    query_size = min(512, set_size)
    for step in tqdm(range(cfg.global_pretrain_steps), desc="prétrain Perceiver global"):
        idx = torch.randperm(len(ref_emb), generator=gen, device=device)[:set_size]
        subset = ref_emb[idx]
        q_idx = torch.randperm(set_size, generator=gen, device=device)[:query_size]
        opt.zero_grad(set_to_none=True)
        with autocast_context(device):
            latents = model.global_pool(subset)
            q = query_proj(subset[q_idx]).unsqueeze(0)
            reconstructed, _ = decoder(q, latents, latents, need_weights=False)
            # embed_dim == latent_dim dans les profils fournis : cible fixe,
            # ce qui interdit la solution triviale ref_proj -> zéro.
            target = subset[q_idx].unsqueeze(0).detach()
            loss = F.smooth_l1_loss(reconstructed, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.global_pool.parameters(), cfg.clip_grad)
        opt.step()
    torch.save(model.global_pool.state_dict(), path)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def evaluate_loader(model: RetrievalPerceiverImputer, loader: DataLoader, ref_emb: torch.Tensor,
                    device: torch.device) -> Tuple[float, Dict[int, List[float]]]:
    model.eval()
    errors, by_class = [], {0: [], 1: [], 2: [], 3: []}
    with torch.inference_mode():
        with autocast_context(device):
            global_tokens = model.global_tokens(ref_emb)
        for batch in loader:
            batch = to_device(batch, device)
            with autocast_context(device):
                pred, _ = model(batch, ref_emb, global_tokens)
            valid = batch["loss_mask"].bool() & torch.isfinite(batch["truth"])
            err = torch.where(valid, (pred.float() - batch["truth"]).abs(), torch.zeros_like(pred.float()))
            scale = batch["scale"][:, None]
            errors.extend((err * scale)[valid].cpu().tolist())
            cls = torch.round(batch["gap"][:, :, 3] * 3).long()
            for c in range(4):
                m = valid & (cls == c)
                by_class[c].extend((err * scale)[m].cpu().tolist())
    return float(np.mean(errors)) if errors else float("inf"), by_class


def train_imputer(cfg: Config, model: RetrievalPerceiverImputer, train_loader: DataLoader,
                  val_loader: DataLoader, ref_emb: torch.Tensor, device: torch.device,
                  work: Path, epochs: int, checkpoint_name: str) -> float:
    ckpt = work / checkpoint_name
    model.to(device)
    ref_emb = ref_emb.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=False)
    best, bad = float("inf"), 0
    for epoch in range(epochs):
        model.train()
        # Le contexte est commun à tous les exemples. Le calculer une fois par
        # epoch évite des milliers de cross-attentions inutiles sur 57k tokens.
        with torch.no_grad(), autocast_context(device):
            cached_global = model.global_tokens(ref_emb).detach()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        bar = tqdm(train_loader, desc=f"finetune {epoch + 1}/{epochs}")
        for step, batch in enumerate(bar):
            batch = to_device(batch, device)
            with autocast_context(device):
                pred, _ = model(batch, ref_emb, cached_global)
                loss = masked_l1(pred, batch["truth"], batch["loss_mask"]) / cfg.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % cfg.grad_accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += float(loss.detach()) * cfg.grad_accum
            bar.set_postfix(l1=running / (step + 1))
        val_mae, categories = evaluate_loader(model, val_loader, ref_emb, device)
        cat_means = {k: float(np.mean(v)) if v else None for k, v in categories.items()}
        print(f"epoch={epoch + 1} validation MAE={val_mae:.3f} W catégories={cat_means}")
        if val_mae < best:
            best, bad = val_mae, 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "val_mae": best}, ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                print("Early stopping")
                break
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return best


# %% [markdown]
# ## Inférence, adaptation légère et soumission

# %%
def adapt_target(model: RetrievalPerceiverImputer, raw: np.ndarray, refs: np.ndarray,
                 ref_indices: np.ndarray, tf: np.ndarray, ref_emb: torch.Tensor,
                 global_tokens: torch.Tensor, device: torch.device, cfg: Config) -> RetrievalPerceiverImputer:
    """Adapte seulement le gate et les experts sur des points observés artificiellement cachés."""
    if not cfg.use_test_time_adaptation:
        return model
    adapted = copy.deepcopy(model)
    for p in adapted.parameters():
        p.requires_grad_(False)
    params = list(adapted.gate.parameters()) + list(adapted.experts.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg.tta_lr)
    observed_idx_np = np.flatnonzero(np.isfinite(raw))
    if len(observed_idx_np) < 20:
        return model
    adapted.train()
    for step in range(cfg.tta_steps):
        rng = np.random.default_rng(cfg.seed + step)
        take_np = rng.choice(observed_idx_np, size=max(8, len(observed_idx_np) // 20), replace=False)
        artificial = raw.copy()
        artificial[take_np] = np.nan
        # Recalcule a/b et PCHIP sans les points artificiellement cachés : aucune fuite TTA.
        idx, meta = affine_rerank_one(artificial, refs, ref_indices, len(ref_indices))
        sample_np = make_model_sample(artificial, raw, refs, idx, meta, tf)
        b = {k: (torch.from_numpy(v)[None] if isinstance(v, np.ndarray) else torch.tensor([v]))
             for k, v in sample_np.items()}
        b = to_device(b, device)
        opt.zero_grad(set_to_none=True)
        with autocast_context(device):
            pred, _ = adapted(b, ref_emb, global_tokens)
            loss = masked_l1(pred, b["truth"], b["loss_mask"])
        loss.backward()
        opt.step()
    return adapted


def predict_targets(cfg: Config, model: RetrievalPerceiverImputer, targets: np.ndarray,
                    refs: np.ndarray, indices: np.ndarray, meta: np.ndarray, tf: np.ndarray,
                    ref_emb: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    output = targets.copy().astype(np.float32)
    with torch.inference_mode(), autocast_context(device):
        global_tokens = model.global_tokens(ref_emb).detach()
    for i in tqdm(range(len(targets)), desc="imputation test"):
        raw = targets[i]
        # Vérité factice : seuls les points observés sont utilisés par la TTA ; les vrais trous restent NaN.
        sample_np = make_model_sample(raw, raw, refs, indices[i], meta[i], tf)
        sample_np["truth"] = np.where(np.isfinite(sample_np["truth"]),
                                      sample_np["truth"], sample_np["baseline"])
        sample = {k: (torch.from_numpy(v)[None] if isinstance(v, np.ndarray) else torch.tensor([v]))
                  for k, v in sample_np.items()}
        sample = to_device(sample, device)
        active = adapt_target(model, raw, refs, indices[i], tf, ref_emb,
                              global_tokens, device, cfg)
        active.eval()
        with torch.inference_mode(), autocast_context(device):
            pred_z, _ = active(sample, ref_emb, global_tokens)
        pred = pred_z[0].float().cpu().numpy() * float(sample_np["scale"]) + float(sample_np["center"])
        missing = ~np.isfinite(raw)
        output[i, missing] = pred[missing]
        output[i, ~missing] = raw[~missing]
        del active, sample
        if i % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    return output


def validate_submission(pred: pd.DataFrame, sample: pd.DataFrame, x_test_holed: pd.DataFrame) -> None:
    assert pred.shape == sample.shape, (pred.shape, sample.shape)
    assert pred.index.equals(sample.index)
    assert pred.columns.equals(sample.columns)
    assert np.isfinite(pred.to_numpy()).all()
    observed = np.isfinite(x_test_holed.to_numpy())
    if not np.allclose(pred.to_numpy()[observed], x_test_holed.to_numpy()[observed], rtol=0, atol=1e-5):
        raise AssertionError("Les valeurs observées ont été modifiées")


def fingerprint_frames(x_train: pd.DataFrame, x_test: pd.DataFrame) -> str:
    payload = f"{x_train.shape}|{x_test.shape}|{x_train.columns[0]}|{x_test.columns[-1]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# %% [markdown]
# ## Pipeline principal

# %%
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full", "predict"), default="full")
    parser.add_argument(
        "--data-dir", default=None,
        help="Dossier des CSV; avec !python, téléverser d'abord dans une cellule puis utiliser /content",
    )
    parser.add_argument("--work-dir", default="/content/enedis_retrieval_work" if Path("/content").exists() else "claude2/retrieval_work")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    cfg = Config.for_profile("full" if args.profile == "predict" else args.profile)
    cfg.use_test_time_adaptation &= not args.no_tta
    seed_everything(cfg.seed)
    if not torch.cuda.is_available() and cfg.profile == "full":
        raise RuntimeError("Le profil full requiert CUDA. Choisis une A100/H100 80 Go dans Colab.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, VRAM={props.total_memory / 2**30:.1f} GiB")

    root = upload_data_if_needed(args.data_dir)
    # Sépare les caches incompatibles (dimensions smoke/full).
    work = Path(args.work_dir) / cfg.profile
    work.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, sample = load_frames(root)
    cfg.n_steps = len(x_train)
    train_ref_cols, test_ref_cols = complete_columns(x_train), complete_columns(x_test)
    train_targets_cols = list(y_train.columns)
    test_targets_cols = list(sample.columns)
    print(f"train refs={len(train_ref_cols)}, test refs={len(test_ref_cols)}, "
          f"train targets={len(train_targets_cols)}, test targets={len(test_targets_cols)}")

    # Courbes en lignes : (N, T). La banque n'inclut jamais y_train.
    ref_raw = np.concatenate([
        x_train[train_ref_cols].to_numpy(np.float32).T,
        x_test[test_ref_cols].to_numpy(np.float32).T,
    ], axis=0)
    ref_z, _, _ = robust_curve_normalize(ref_raw)
    train_raw = x_train[train_targets_cols].to_numpy(np.float32).T
    train_truth = y_train[train_targets_cols].to_numpy(np.float32).T
    test_raw = x_test[test_targets_cols].to_numpy(np.float32).T
    gaps = extract_gap_lengths(x_train[train_targets_cols])
    tf = time_features(x_train.index)
    print("Banque transductive:", ref_raw.shape, "distribution gaps:",
          {"<=3": int((gaps <= 3).sum()), "4-47": int(((gaps >= 4) & (gaps <= 47)).sum()),
           "48-95": int(((gaps >= 48) & (gaps <= 95)).sum()), ">=96": int((gaps >= 96).sum())})

    manifest = {"fingerprint": fingerprint_frames(x_train, x_test), "config": asdict(cfg),
                "reference_count": len(ref_raw)}
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    encoder = SharedCurveEncoder(cfg)
    pretrain_encoder(cfg, encoder, ref_raw, gaps, device, work)
    ref_embeddings_np = encode_reference_bank(cfg, encoder, ref_z, device, work)
    encoder.cpu()
    del encoder, ref_z
    gc.collect()
    torch.cuda.empty_cache()

    # Recharge l'encodeur pré-entraîné pour encoder les targets sans le garder ensuite en VRAM.
    target_encoder = SharedCurveEncoder(cfg)
    target_encoder.load_state_dict(torch.load(work / "encoder_pretrained.pt", map_location="cpu", weights_only=True))
    target_encoder.to(device).eval()
    train_idx, train_meta = build_retrieval_cache("train", cfg, target_encoder, train_raw,
                                                   ref_raw, ref_embeddings_np, device, work)
    test_idx, test_meta = build_retrieval_cache("test", cfg, target_encoder, test_raw,
                                                 ref_raw, ref_embeddings_np, device, work)
    del target_encoder
    torch.cuda.empty_cache()

    all_ids = np.arange(len(train_raw))
    fit_ids, val_ids = train_test_split(all_ids, test_size=cfg.val_fraction, random_state=cfg.seed)
    supervised_ds = RetrievalDataset(train_raw, train_truth, ref_raw, train_idx, train_meta, tf,
                                     fit_ids, repeat=cfg.supervised_repeat)
    rng = np.random.default_rng(cfg.seed)
    n_synth = min(cfg.synthetic_samples_per_epoch, len(ref_raw))
    synth_ids = rng.choice(len(ref_raw), size=n_synth, replace=False).astype(np.int32)
    synth_cache = work / f"synthetic_candidates_{n_synth}_k{cfg.k_local}.npy"
    if synth_cache.exists():
        synth_candidates = np.load(synth_cache)
    else:
        # k+1 garantit qu'il reste k voisins après retrait de la cible elle-même.
        synth_candidates = coarse_neighbors(ref_embeddings_np[synth_ids], ref_embeddings_np,
                                             cfg.k_local + 1, device)
        np.save(synth_cache, synth_candidates)
    synth_masked, synth_indices, synth_meta = build_synthetic_cache(
        cfg, ref_raw, synth_ids, synth_candidates, gaps, work)
    synthetic_ds = SyntheticRetrievalDataset(ref_raw, synth_ids, synth_masked,
                                              synth_indices, synth_meta, tf)
    train_ds = ConcatDataset([supervised_ds, synthetic_ds])
    val_ds = RetrievalDataset(train_raw, train_truth, ref_raw, train_idx, train_meta, tf, val_ids)
    train_loader = DataLoader(train_ds, batch_size=cfg.train_batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train_batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    ref_emb = torch.from_numpy(ref_embeddings_np)
    model = RetrievalPerceiverImputer(cfg)
    final_ckpt = work / "retrieval_perceiver_final.pt"

    if args.profile == "predict":
        if not final_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint final absent: {final_ckpt}")
        model.load_state_dict(torch.load(final_ckpt, map_location="cpu", weights_only=False)["model"])
    else:
        pretrain_global_perceiver(cfg, model, ref_emb, device, work)
        best = train_imputer(cfg, model, train_loader, val_loader, ref_emb, device, work,
                             cfg.finetune_epochs, "retrieval_perceiver_best.pt")
        print(f"Meilleure MAE holdout: {best:.3f} W")
        # Entraînement final court sur les 1 000 targets, initialisé au meilleur holdout.
        full_ds = RetrievalDataset(train_raw, train_truth, ref_raw, train_idx, train_meta, tf,
                                   all_ids, repeat=max(1, cfg.supervised_repeat // 2))
        full_loader = DataLoader(full_ds, batch_size=cfg.train_batch_size, shuffle=True,
                                 num_workers=cfg.num_workers, pin_memory=True)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_finetune * 0.35, weight_decay=cfg.weight_decay)
        model.to(device)
        ref_emb_dev = ref_emb.to(device)
        for epoch in range(cfg.final_epochs):
            model.train()
            with torch.no_grad(), autocast_context(device):
                cached_global = model.global_tokens(ref_emb_dev).detach()
            bar = tqdm(full_loader, desc=f"final {epoch + 1}/{cfg.final_epochs}")
            for batch in bar:
                batch = to_device(batch, device)
                opt.zero_grad(set_to_none=True)
                with autocast_context(device):
                    pred, _ = model(batch, ref_emb_dev, cached_global)
                    loss = masked_l1(pred, batch["truth"], batch["loss_mask"])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                opt.step()
                bar.set_postfix(l1=float(loss.detach()))
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "holdout_mae": best}, final_ckpt)

    model.to(device).eval()
    ref_emb = ref_emb.to(device)
    predictions = predict_targets(cfg, model, test_raw, ref_raw, test_idx, test_meta, tf,
                                  ref_emb, device)
    pred_df = pd.DataFrame(predictions.T, index=x_test.index, columns=test_targets_cols)
    pred_df = pred_df[sample.columns]
    validate_submission(pred_df, sample, x_test[sample.columns])
    submission = work / "submission_retrieval_perceiver_t1.csv"
    pred_df.to_csv(submission)
    print("Soumission valide écrite:", submission)

    archive = shutil.make_archive(str(work / "enedis_retrieval_artifacts"), "zip", root_dir=work)
    print("Archive:", archive)
    try:
        from google.colab import files  # type: ignore
        files.download(str(submission))
    except ImportError:
        pass


if __name__ == "__main__":
    main()

