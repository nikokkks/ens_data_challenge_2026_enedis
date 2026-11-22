"""5 améliorations du réseau séquentiel, centrées sur les features TEMPORELLES
et SEQUENTIELLES (round 8).

Base = meilleure configuration du sweep § 7ter : `L13_transformer`
(Transformer encodeur, fenêtre 192, 2 couches, d_model 64, dropout 0,1,
lr 1e-3), val temporelle 0,4779.

La temporalité est respectée à trois endroits, et c'est le point délicat :

1. **Split temporel inchangé** : fenêtres d'entraînement dans `[0, 792)`,
   fenêtres de validation dans `[792, 1057)`. Le meilleur checkpoint est
   choisi sur la validation, jamais sur la dernière epoch.
2. **Les copies périodiques `t±48/±96/±336` sont calculées sur la courbe
   MASQUEE, pas sur la vérité.** Une copie n'est renseignée que si le point
   source est réellement observé ; sinon elle vaut 0 et son drapeau de
   disponibilité vaut 0. C'est exactement la sémantique de
   `feb.periodic_copy_features`, qui garantit des features identiques à
   l'entraînement et à la prédiction.
3. **Le masque est tiré sur la courbe ENTIERE avant découpage de la
   fenêtre**, sinon une copie à `t-336` pointerait hors fenêtre vers une
   valeur non masquée et donnerait au réseau une information qu'il n'aura
   pas à l'inférence. Les canaux sont donc construits sur les 1057 pas puis
   tranchés.

Une copie prise à `t-336` depuis une fenêtre de validation lit une valeur
de la période d'entraînement : c'est légitime et non fuyant, puisqu'il
s'agit du passé OBSERVE de la courbe elle-même, disponible tel quel à
l'inférence réelle. Le split temporel protège les POIDS, pas les entrées.
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
import feature_engineering_bench as feb
import lstm_bench as lb

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "6")))

RANDOM_STATE = 42
T_SPLIT = lb.T_SPLIT
PROGRESS_PATH = "lstm_v2_progress.json"
LAGS = (48, 96, 336)

BASE = dict(cell="transformer", window=192, layers=2, hidden=64, dropout=.1, lr=1e-3)

CONFIGS = {
    # S0 : contrôle -- doit retrouver L13_transformer (0,4779)
    "S0_control":          dict(BASE),
    "S1_periodic_copies":  dict(BASE, periodic=True),
    "S2_edge_distance":    dict(BASE, edge_distance=True),
    "S3_full_calendar":    dict(BASE, full_calendar=True),
    "S4_gap_weighted_loss": dict(BASE, gap_weight=0.5, residual=True),
    "S5_combo":            dict(BASE),   # rempli après ablation (voir __main__)
}


# ---------------------------------------------------------------------------
# Canaux d'entrée, construits sur la COURBE ENTIERE puis tranchés
# ---------------------------------------------------------------------------

def n_channels(cfg):
    n = 3                                   # valeur masquée, masque, interpolation
    n += 8 if cfg.get("full_calendar") else 4
    if cfg.get("periodic"):
        n += 4 * len(LAGS)                  # 2 copies (avant/arrière) + 2 drapeaux
    if cfg.get("edge_distance"):
        n += 2
    return n


def edge_distance_channels(mask):
    """Distance au point OBSERVE le plus proche en arrière et en avant --
    feature purement séquentielle : elle dit au réseau à quel point il
    extrapole. C'est l'information qui pilote l'alpha adaptatif du pipeline
    principal, jusqu'ici invisible pour le réseau."""
    n = len(mask)
    back = np.zeros(n)
    fwd = np.zeros(n)
    d = 0.0
    for i in range(n):
        d = 0.0 if not mask[i] else d + 1.0
        back[i] = d
    d = 0.0
    for i in range(n - 1, -1, -1):
        d = 0.0 if not mask[i] else d + 1.0
        fwd[i] = d
    return np.column_stack([np.log1p(back), np.log1p(fwd)])


def periodic_channels(curve_z, mask):
    """Copies périodiques calculées sur la courbe MASQUEE (NaN aux trous),
    remplacées par 0 avec un drapeau de disponibilité -- un réseau ne sait
    pas consommer des NaN."""
    masked = np.where(mask, np.nan, curve_z)
    raw = feb.periodic_copy_features(masked, lags=LAGS, include_availability=True)
    return np.nan_to_num(raw, nan=0.0)


def build_channels(curve_z, mask, tfeat_full, cfg):
    n = len(curve_z)
    idx = np.arange(n)
    obs = ~mask
    interp = curve_z.copy()
    if obs.any():
        interp[mask] = np.interp(idx[mask], idx[obs], curve_z[obs])
    cols = [np.where(mask, 0.0, curve_z)[:, None], mask.astype(np.float64)[:, None],
            interp[:, None], tfeat_full]
    if cfg.get("periodic"):
        cols.append(periodic_channels(curve_z, mask))
    if cfg.get("edge_distance"):
        cols.append(edge_distance_channels(mask))
    return np.hstack(cols).astype(np.float32), interp.astype(np.float32)


def make_batch_v2(curves, tfeat_full, lo, hi, cfg, batch_size, rng):
    """Masque tiré sur la courbe entière, canaux construits sur la courbe
    entière, PUIS découpage de la fenêtre [start, start+window)."""
    window = cfg["window"]
    n_times = curves.shape[1]
    rows = rng.randint(0, curves.shape[0], size=batch_size)
    starts = rng.randint(lo, hi - window + 1, size=batch_size)
    masks_full = lb.sample_mask((batch_size, n_times), rng)

    feats, ys, ms, interps, weights = [], [], [], [], []
    for b, (r, st) in enumerate(zip(rows, starts)):
        ch, interp = build_channels(curves[r].astype(np.float64), masks_full[b],
                                    tfeat_full, cfg)
        sl = slice(st, st + window)
        feats.append(ch[sl])
        ys.append(curves[r][sl])
        ms.append(masks_full[b][sl])
        interps.append(interp[sl])
        if cfg.get("gap_weight"):
            run = feb.run_lengths_for(masks_full[b])[sl]
            weights.append(1.0 + cfg["gap_weight"] * np.log1p(run))
        else:
            weights.append(np.ones(window))
    return (torch.from_numpy(np.stack(feats)),
            torch.from_numpy(np.stack(ys).astype(np.float32)),
            torch.from_numpy(np.stack(ms)),
            torch.from_numpy(np.stack(interps)),
            torch.from_numpy(np.stack(weights).astype(np.float32)))


class SeqImputerV2(lb.SeqImputer):
    def __init__(self, n_in, **cfg):
        nn.Module.__init__(self)
        self.cell = cfg.get("cell", "transformer")
        hidden, layers = cfg["hidden"], cfg["layers"]
        dropout = cfg["dropout"]
        if self.cell == "transformer":
            self.proj = nn.Linear(n_in, hidden)
            enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=4,
                dim_feedforward=4 * hidden, dropout=dropout, batch_first=True)
            self.body = nn.TransformerEncoder(enc, num_layers=layers)
            self.head = nn.Linear(hidden, 1)
        else:
            rnn = nn.LSTM if self.cell == "lstm" else nn.GRU
            self.body = rnn(n_in, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.)
            self.head = nn.Linear(2 * hidden, 1)


def train_config(cfg, curves, tfeat_full, max_steps, batch_size=48,
                 eval_every=50, patience=4):
    torch.manual_seed(RANDOM_STATE)
    rng = np.random.RandomState(RANDOM_STATE)
    residual = cfg.get("residual", False)
    n_in = n_channels(cfg)

    model = SeqImputerV2(n_in, **cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    val_rng = np.random.RandomState(RANDOM_STATE + 1000)
    val_batches = [make_batch_v2(curves, tfeat_full, T_SPLIT, curves.shape[1], cfg,
                                 64, val_rng) for _ in range(6)]

    best, best_state, bad, history = np.inf, None, 0, []
    t0 = time.time()
    for step in range(1, max_steps + 1):
        model.train()
        feats, y, mask, interp, w = make_batch_v2(curves, tfeat_full, 0, T_SPLIT,
                                                  cfg, batch_size, rng)
        out = model(feats)
        pred = interp + out if residual else out
        err = (pred[mask] - y[mask]).abs()
        loss = (err * w[mask]).sum() / w[mask].sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                vals = []
                for fv, yv, mv, iv, _ in val_batches:
                    ov = model(fv)
                    pv = iv + ov if residual else ov
                    # la validation reste une MAE NON pondérée -> comparable
                    # au sweep § 7ter quelle que soit la pondération de loss
                    vals.append(float((pv[mv] - yv[mv]).abs().mean()))
            v = float(np.mean(vals))
            history.append({"step": step, "val_mae_norm": v})
            print(f"    step {step:4d}  train={float(loss):.4f}  val={v:.4f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    print(f"    early stopping au step {step}", flush=True)
                    break

    model.load_state_dict(best_state)
    return model, best, history


def impute_curve_v2(model, values, tfeat_full, cfg, stride=None):
    """Inférence sur une courbe réelle : mêmes canaux, construits sur la
    courbe entière avec son vrai masque de NaN, puis fenêtres glissantes
    moyennées."""
    window = cfg["window"]
    residual = cfg.get("residual", False)
    stride = stride or window // 2
    n = len(values)
    missing = np.isnan(values)
    obs = ~missing
    center, scale = lb.robust_scale(values[obs][None, :].astype(np.float32))
    center, scale = center.item(), scale.item()
    z = np.clip((values - center) / scale, -lb.Z_CLIP, lb.Z_CLIP)
    z = np.nan_to_num(z, nan=0.0)

    ch, interp = build_channels(z, missing, tfeat_full, cfg)
    acc, cnt = np.zeros(n), np.zeros(n)
    starts = list(range(0, max(1, n - window + 1), stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    with torch.no_grad():
        for st in starts:
            sl = slice(st, st + window)
            out = model(torch.from_numpy(ch[sl][None]))[0].numpy()
            acc[sl] += interp[sl] + out if residual else out
            cnt[sl] += 1
    z_pred = acc / np.maximum(cnt, 1)
    filled = values.copy()
    filled[missing] = z_pred[missing] * scale + center
    return filled


if __name__ == "__main__":
    max_steps = int(os.environ.get("MAX_STEPS", "600"))
    only = sys.argv[1] if len(sys.argv) > 1 else None

    print("Chargement des données...", flush=True)
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]
    donors = X_train[donor_cols].ffill().bfill().values.astype(np.float32).T
    curves, _, _ = lb.robust_norm(donors)

    try:
        results = json.load(open(PROGRESS_PATH))
    except FileNotFoundError:
        results = {}

    names = [only] if only else list(CONFIGS)
    for name in names:
        if name in results:
            print(f"=== {name} déjà fait ===", flush=True)
            continue
        cfg = dict(CONFIGS[name])
        if name == "S5_combo":
            # combine les seules options validées individuellement
            base_val = results.get("S0_control", {}).get("best_val_mae_norm")
            for opt_name, key in [("S1_periodic_copies", "periodic"),
                                  ("S2_edge_distance", "edge_distance"),
                                  ("S3_full_calendar", "full_calendar")]:
                got = results.get(opt_name, {}).get("best_val_mae_norm")
                if base_val is not None and got is not None and got < base_val:
                    cfg[key] = True
            print(f"  combo retenu : {[k for k in ('periodic','edge_distance','full_calendar') if cfg.get(k)]}",
                  flush=True)

        tf = (feb.time_features_full_seasonality(X_train.index, ("hour", "dow", "dom", "year"))
              if cfg.get("full_calendar") else lb.time_channels(X_train.index)).astype(np.float64)

        print(f"=== {name} : {cfg} ({n_channels(cfg)} canaux) ===", flush=True)
        t0 = time.time()
        model, best, history = train_config(cfg, curves, tf, max_steps)
        results[name] = {"config": cfg, "best_val_mae_norm": best, "history": history,
                         "n_channels": n_channels(cfg), "train_seconds": time.time() - t0}
        with open(PROGRESS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        torch.save(model.state_dict(), f"lstmv2_{name}.pt")
        print(f"  -> best val MAE (normalisée) = {best:.4f}  ({time.time()-t0:.0f}s)\n",
              flush=True)
