"""Étape 3 : LSTM / GRU / Transformer séquentiel comme baseline documentée.

DEUX NIVEAUX DE SPLIT, à ne pas confondre
-----------------------------------------
1. Split TEMPOREL (interne, ce fichier) -- sert UNIQUEMENT à l'arrêt anticipé.
   Le réseau est entraîné en auto-supervisé sur les 20 000 courbes DONNEUSES
   (jamais sur `y_train`), avec des trous artificiels tirés de la distribution
   empirique des vrais trous. Les fenêtres d'entraînement sont entièrement
   contenues dans `[0, T_SPLIT)` (9 -> ~25 janvier) et les fenêtres de
   validation entièrement dans `[T_SPLIT, 1057)` (~25 -> 31 janvier). On ne
   valide donc jamais sur une fenêtre temporelle vue à l'entraînement, et le
   meilleur checkpoint est choisi sur cette courbe de validation (early
   stopping), pas à la dernière epoch.

2. Split par COLONNES (externe, `feb.evaluate_repeated_splits`) -- c'est la
   mesure de performance officielle : 10 repeats x 10 colonnes `holed_*`,
   MAE sur les seuls vrais trous. C'est le protocole `10x10` déjà utilisé
   pour TabPFN I4 et R0/R1, avec les mêmes graines, donc directement
   comparable.

Pourquoi c'est cohérent avec la métrique officielle : les vrais trous sont
répartis sur TOUTE la période, pas seulement en fin de période. Le split
temporel est donc PLUS STRICT que nécessaire -- il ne sert qu'à choisir
honnêtement quand arrêter l'entraînement. Une fois le checkpoint retenu, le
réseau est appliqué à toute la période. Aucune fuite n'est possible : les
courbes d'entraînement (donneuses) sont disjointes des colonnes `holed_*`
notées, et `y_train` n'entre jamais dans l'entraînement.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

from evaluate import load_data

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))

RANDOM_STATE = 42
T_SPLIT = 792                 # ~25 janvier : avant = entraînement, après = validation
MASK_RATE = 0.123             # taux de trous mesuré sur les vraies colonnes holed_*
PROGRESS_PATH = "lstm_configs_progress.json"

# Distribution empirique des longueurs de trous (mesurée sur X_train[holed_*]) :
# 75 % des trous font 1 pas, mais 90 % des points manquants sont dans des
# trous <= 48 pas. On rejoue cette distribution pour masquer artificiellement.
HOLE_LENGTHS = np.array([1, 2, 3, 6, 12, 24, 48, 96])
HOLE_PROBS = np.array([.751, .187, .017, .005, .021, .001, .015, .003])
HOLE_PROBS = HOLE_PROBS / HOLE_PROBS.sum()


# ---------------------------------------------------------------------------
# 15 configurations, listées AVANT de lancer les runs
# ---------------------------------------------------------------------------
# Contrainte : la fenêtre doit tenir dans la région de validation temporelle
# (1057 - 792 = 265 pas), d'où window <= 240.
CONFIGS = {
    "L01_base_bilstm":      dict(cell="lstm", window=192, layers=2, hidden=64,  dropout=.1, lr=1e-3),
    "L02_hidden128":        dict(cell="lstm", window=192, layers=2, hidden=128, dropout=.1, lr=1e-3),
    "L03_shallow":          dict(cell="lstm", window=192, layers=1, hidden=64,  dropout=.0, lr=1e-3),
    "L04_deep3":            dict(cell="lstm", window=192, layers=3, hidden=64,  dropout=.2, lr=1e-3),
    "L05_window96":         dict(cell="lstm", window=96,  layers=2, hidden=64,  dropout=.1, lr=1e-3),
    "L06_window144":        dict(cell="lstm", window=144, layers=2, hidden=64,  dropout=.1, lr=1e-3),
    "L07_window240":        dict(cell="lstm", window=240, layers=2, hidden=64,  dropout=.1, lr=1e-3),
    "L08_lr3e-3":           dict(cell="lstm", window=192, layers=2, hidden=64,  dropout=.1, lr=3e-3),
    "L09_lr3e-4":           dict(cell="lstm", window=192, layers=2, hidden=64,  dropout=.1, lr=3e-4),
    "L10_dropout03":        dict(cell="lstm", window=192, layers=2, hidden=64,  dropout=.3, lr=1e-3),
    "L11_hidden256":        dict(cell="lstm", window=192, layers=2, hidden=256, dropout=.1, lr=1e-3),
    "L12_gru":              dict(cell="gru",  window=192, layers=2, hidden=64,  dropout=.1, lr=1e-3),
    "L13_transformer":      dict(cell="transformer", window=192, layers=2, hidden=64, dropout=.1, lr=1e-3),
    "L14_wide_regularized": dict(cell="lstm", window=240, layers=2, hidden=128, dropout=.2, lr=1e-3),
    "L15_residual_target":  dict(cell="lstm", window=192, layers=2, hidden=64,  dropout=.1, lr=1e-3,
                                 residual=True),
}


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------

Z_CLIP = 20.0


def robust_scale(curves):
    """Echelle par courbe, robuste À LA FOIS aux valeurs extrêmes et aux
    courbes quasi constantes.

    La convention médiane/MAD du reste du dépôt ne tient pas ici : 2,3 % des
    courbes donneuses sont plates la majorité du temps (paliers nocturnes),
    leur MAD tombe au plancher et le z-score explose jusqu'à 5,6e5 -- ce qui
    faisait diverger l'entraînement (loss 0,84 -> 278 en 100 pas). On prend
    donc le MAX de trois mesures d'étalement, dont deux ne peuvent pas
    s'effondrer sur une courbe à pics."""
    center = np.median(curves, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(curves - center), axis=1, keepdims=True)
    p05, p95 = np.percentile(curves, [5, 95], axis=1)
    spread = ((p95 - p05) / 3.29)[:, None]
    scale = np.maximum(np.maximum(mad, spread), 0.05 * np.abs(center))
    return center, np.maximum(scale, 1.0)


def robust_norm(curves):
    """Normalise puis borne : le réseau apprend des formes, pas des niveaux,
    et aucun pic isolé ne peut dominer la loss."""
    center, scale = robust_scale(curves)
    return np.clip((curves - center) / scale, -Z_CLIP, Z_CLIP), center, scale


def time_channels(index):
    dt = pd.to_datetime(index)
    hour = dt.hour.values + dt.minute.values / 60.0
    dow = dt.dayofweek.values
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
    ]).astype(np.float32)


def sample_mask(shape, rng):
    """Trous artificiels tirés de la distribution empirique réelle."""
    B, W = shape
    mask = np.zeros((B, W), dtype=bool)
    target = int(MASK_RATE * W)
    for b in range(B):
        filled = 0
        while filled < target:
            L = int(rng.choice(HOLE_LENGTHS, p=HOLE_PROBS))
            start = rng.randint(0, max(1, W - L))
            mask[b, start:start + L] = True
            filled = mask[b].sum()
    return mask


def linear_interp_rows(x, mask):
    """Interpolation linéaire ligne par ligne (canal d'entrée du réseau, et
    cible de référence pour la variante `residual`)."""
    out = x.copy()
    for b in range(x.shape[0]):
        m = mask[b]
        if not m.any():
            continue
        idx = np.arange(x.shape[1])
        obs = ~m
        if obs.sum() == 0:
            out[b] = 0.0
        else:
            out[b, m] = np.interp(idx[m], idx[obs], x[b, obs])
    return out


def make_batch(curves, tfeat, lo, hi, window, batch_size, rng):
    """Une fenêtre aléatoire par courbe tirée dans [lo, hi)."""
    n = curves.shape[0]
    rows = rng.randint(0, n, size=batch_size)
    starts = rng.randint(lo, hi - window + 1, size=batch_size)
    x = np.stack([curves[r, s:s + window] for r, s in zip(rows, starts)])
    tf = np.stack([tfeat[s:s + window] for s in starts])
    mask = sample_mask(x.shape, rng)
    x_in = x.copy()
    x_in[mask] = 0.0
    interp = linear_interp_rows(x, mask)
    feats = np.concatenate([
        x_in[..., None], mask[..., None].astype(np.float32), interp[..., None], tf,
    ], axis=-1).astype(np.float32)
    return (torch.from_numpy(feats), torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(mask), torch.from_numpy(interp.astype(np.float32)))


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

N_CHANNELS = 7  # valeur masquée, masque, interpolation, sin/cos heure, sin/cos jour


class SeqImputer(nn.Module):
    def __init__(self, cell="lstm", hidden=64, layers=2, dropout=.1, **_):
        super().__init__()
        self.cell = cell
        if cell == "transformer":
            self.proj = nn.Linear(N_CHANNELS, hidden)
            enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=4,
                dim_feedforward=4 * hidden, dropout=dropout, batch_first=True)
            self.body = nn.TransformerEncoder(enc, num_layers=layers)
            self.head = nn.Linear(hidden, 1)
        else:
            rnn = nn.LSTM if cell == "lstm" else nn.GRU
            self.body = rnn(N_CHANNELS, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.)
            self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        if self.cell == "transformer":
            h = self.body(self.proj(x))
        else:
            h, _ = self.body(x)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Entraînement avec early stopping sur la validation TEMPORELLE
# ---------------------------------------------------------------------------

def train_config(name, cfg, curves, tfeat, max_steps, batch_size=48,
                 eval_every=50, patience=4, verbose=True):
    torch.manual_seed(RANDOM_STATE)
    rng = np.random.RandomState(RANDOM_STATE)
    window = cfg["window"]
    residual = cfg.get("residual", False)

    model = SeqImputer(**cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.L1Loss()

    # jeu de validation temporel FIXE (mêmes fenêtres/masques pour toutes les
    # configs -> comparaison équitable entre configurations)
    val_rng = np.random.RandomState(RANDOM_STATE + 1000)
    val_batches = [make_batch(curves, tfeat, T_SPLIT, curves.shape[1], window, 64, val_rng)
                   for _ in range(6)]

    best, best_state, bad, history = np.inf, None, 0, []
    t0 = time.time()
    for step in range(1, max_steps + 1):
        model.train()
        feats, y, mask, interp = make_batch(curves, tfeat, 0, T_SPLIT, window,
                                            batch_size, rng)
        out = model(feats)
        pred = interp + out if residual else out
        loss = loss_fn(pred[mask], y[mask])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                vals = []
                for feats_v, y_v, mask_v, interp_v in val_batches:
                    out_v = model(feats_v)
                    pred_v = interp_v + out_v if residual else out_v
                    vals.append(float((pred_v[mask_v] - y_v[mask_v]).abs().mean()))
            v = float(np.mean(vals))
            history.append({"step": step, "val_mae_norm": v})
            if verbose:
                print(f"    step {step:4d}  train={float(loss):.4f}  val={v:.4f}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"    early stopping au step {step}", flush=True)
                    break

    model.load_state_dict(best_state)
    return model, best, history


# ---------------------------------------------------------------------------
# Inférence sur une courbe complète (fenêtres glissantes moyennées)
# ---------------------------------------------------------------------------

def impute_curve(model, values, tfeat, window, residual, stride=None):
    """`values` : courbe brute avec NaN. Retourne la courbe complétée."""
    stride = stride or window // 2
    n = len(values)
    missing = np.isnan(values)
    obs = ~missing
    # même convention d'échelle qu'à l'entraînement, calculée sur les seuls
    # points OBSERVES de la courbe (aucun accès aux valeurs à prédire)
    center, scale = robust_scale(values[obs][None, :].astype(np.float32))
    center, scale = center.item(), scale.item()
    z = np.clip((values - center) / scale, -Z_CLIP, Z_CLIP)
    idx = np.arange(n)
    z_interp = z.copy()
    z_interp[missing] = np.interp(idx[missing], idx[obs], z[obs])
    z_in = np.where(missing, 0.0, z)

    acc = np.zeros(n)
    cnt = np.zeros(n)
    starts = list(range(0, max(1, n - window + 1), stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    with torch.no_grad():
        for s in starts:
            sl = slice(s, s + window)
            feats = np.concatenate([
                z_in[sl, None], missing[sl, None].astype(np.float32), z_interp[sl, None],
                tfeat[sl],
            ], axis=-1).astype(np.float32)[None]
            out = model(torch.from_numpy(feats))[0].numpy()
            pred = z_interp[sl] + out if residual else out
            acc[sl] += pred
            cnt[sl] += 1
    z_pred = acc / np.maximum(cnt, 1)
    filled = values.copy()
    filled[missing] = z_pred[missing] * scale + center
    return filled


def make_lstm_model(model, tfeat_full, window, residual):
    """Adaptateur au format attendu par `feb.evaluate_repeated_splits`."""
    def fn(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            if not s.isna().any():
                preds[col] = s
                continue
            filled = impute_curve(model, s.values.astype(np.float64), tfeat_full,
                                  window, residual)
            preds[col] = pd.Series(filled, index=s.index)
        return pd.DataFrame(preds, index=val_holed_df.index)
    return fn


if __name__ == "__main__":
    max_steps = int(os.environ.get("MAX_STEPS", "600"))
    only = sys.argv[1] if len(sys.argv) > 1 else None

    print("Chargement des données...", flush=True)
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]
    donors = X_train[donor_cols].ffill().bfill().values.astype(np.float32).T
    curves, _, _ = robust_norm(donors)
    tfeat = time_channels(X_train.index)
    print(f"{curves.shape[0]} courbes donneuses x {curves.shape[1]} instants ; "
          f"split temporel à t={T_SPLIT} ({X_train.index[T_SPLIT]})", flush=True)

    try:
        results = json.load(open(PROGRESS_PATH))
    except FileNotFoundError:
        results = {}

    for name in ([only] if only else CONFIGS):
        if name in results:
            print(f"=== {name} déjà fait, on saute ===", flush=True)
            continue
        print(f"=== {name} : {CONFIGS[name]} ===", flush=True)
        t0 = time.time()
        model, best, history = train_config(name, CONFIGS[name], curves, tfeat, max_steps)
        results[name] = {"config": CONFIGS[name], "best_val_mae_norm": best,
                         "history": history, "train_seconds": time.time() - t0}
        with open(PROGRESS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        torch.save(model.state_dict(), f"lstm_{name}.pt")
        print(f"  -> best val MAE (normalisée) = {best:.4f}  "
              f"({time.time()-t0:.0f}s)\n", flush=True)
