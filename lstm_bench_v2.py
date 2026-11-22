"""LSTM v2 -- benchmark torch avec le jeu de features COMPLET du dépôt, RMSE
comme critère de sélection, et plusieurs familles d'architecture.

Différences vs `lstm_bench.py` (v1, 15 configs, 7 canaux bruts) :

1. Features -- v1 n'utilisait que (valeur masquée, masque, interpolation,
   sin/cos heure, sin/cos jour). v2 ajoute les briques déjà validées pour
   les régresseurs classiques (cf. `feature_engineering_bench.BEST_FEATURES`
   et benchmark.md §6) :
     - calendrier complet (heure, jour de semaine, jour du mois, week-end) ;
     - copies périodiques t±48/96/144/336 + disponibilité (§ "copies
       périodiques" / "confiance périodique") ;
     - saisonnalité propre à la courbe (médiane par créneau demi-heure) ;
     - consensus de cluster de forme (moyenne/écart-type du cluster KMeans
       auquel appartient la courbe -- équivalent séquentiel du "clustering"
       §6.2), substitut bon marché à un vrai KNN inter-courbes (voir
       limitation ci-dessous).
   29 canaux au total (1+1+1+7+16+1+1+1), cf. N_CHANNELS_V2.

   LIMITATION ASSUMÉE : contrairement aux modèles arbres/TabPFN du dépôt, le
   vrai KNN corrélation/euclidien par courbe (jusqu'à 40 voisins, recalculé
   par cible) n'est PAS inclus ici -- le brancher proprement dans une boucle
   d'entraînement séquentiel demanderait un index de plus proches voisins
   recalculé à la volée pour un batch de courbes différentes à chaque step,
   ce qui n'a pas été fait faute de temps. Le consensus de cluster (calculé
   UNE FOIS avant l'entraînement, cf. `fit_cluster_consensus`) capture une
   partie du même signal ("courbes de forme proche") à un coût quasi nul.

2. Sélection du meilleur checkpoint -- v1 sélectionnait sur la MAE de
   validation temporelle normalisée ; v2 sélectionne sur la RMSE (racine de
   l'EQM), comme demandé, tout en conservant la MAE dans l'historique pour
   rester comparable à v1 et au reste du dépôt (métrique officielle du
   challenge = MAE).

3. Architectures -- v1 couvrait LSTM bidirectionnel / GRU / Transformer.
   v2 ajoute deux familles : `convlstm` (frontend Conv1d puis BiLSTM, capture
   des motifs locaux avant la récurrence) et `attn` (BiLSTM + une couche
   d'auto-attention multi-tête en sortie, résiduelle).

4. Loss -- configurable `l1` (MAE, comme v1) ou `mse` (optimise directement
   la RMSE). Les deux sont criblées sur la même architecture pour mesurer
   l'effet du choix de loss sur la métrique RMSE, indépendamment de
   l'architecture.

Même protocole de split temporel que v1 (voir docstring de lstm_bench.py) :
entraînement sur `[0, T_SPLIT)`, validation temporelle sur `[T_SPLIT, 1057)`,
uniquement sur les courbes DONNEUSES (jamais `y_train`). L'évaluation
officielle 10x10 par colonnes se fait dans `lstm_official_eval_v2.py`.
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

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))

RANDOM_STATE = 42
T_SPLIT = lb.T_SPLIT
PROGRESS_PATH = "lstm_v2_configs_progress.json"
N_CLUSTERS = 25
CAL_GROUPS = ("hour", "dow", "dom")
PERIODIC_LAGS = (48, 96, 144, 336)
N_CHANNELS_V2 = 29  # 1+1+1 + 7 (cal) + 16 (periodic+avail) + 1 (season) + 1 + 1 (cluster)


# ---------------------------------------------------------------------------
# 10 configurations, listées AVANT de lancer les runs
# ---------------------------------------------------------------------------
CONFIGS_V2 = {
    "W01_bilstm_mse":          dict(cell="lstm", window=192, layers=2, hidden=64,  dropout=.1, lr=1e-3, loss="mse"),
    "W02_bilstm_l1":           dict(cell="lstm", window=192, layers=2, hidden=64,  dropout=.1, lr=1e-3, loss="l1"),
    "W03_gru_mse":             dict(cell="gru",  window=192, layers=2, hidden=64,  dropout=.1, lr=1e-3, loss="mse"),
    "W04_transformer_mse":     dict(cell="transformer", window=192, layers=2, hidden=64, dropout=.1, lr=1e-3, loss="mse"),
    "W05_convlstm_mse":        dict(cell="convlstm", window=192, layers=2, hidden=64, dropout=.1, lr=1e-3, loss="mse"),
    "W06_attn_mse":            dict(cell="attn", window=192, layers=2, hidden=64, dropout=.1, lr=1e-3, loss="mse"),
    "W07_bilstm_mse_hidden128": dict(cell="lstm", window=192, layers=2, hidden=128, dropout=.1, lr=1e-3, loss="mse"),
    "W08_bilstm_mse_window240": dict(cell="lstm", window=240, layers=2, hidden=64, dropout=.1, lr=1e-3, loss="mse"),
    "W09_bilstm_mse_dropout3": dict(cell="lstm", window=192, layers=2, hidden=64,  dropout=.3, lr=1e-3, loss="mse"),
    "W10_bilstm_mse_residual": dict(cell="lstm", window=192, layers=2, hidden=64,  dropout=.1, lr=1e-3, loss="mse",
                                    residual=True),
}


# ---------------------------------------------------------------------------
# Features "toutes features" : calendrier, copies périodiques, saisonnalité
# propre, consensus de cluster
# ---------------------------------------------------------------------------

def build_calendar_full(index):
    return feb.time_features_full_seasonality(index, groups=CAL_GROUPS).astype(np.float32)


def fit_cluster_consensus(X_train, donor_cols, curves, n_clusters=N_CLUSTERS,
                          random_state=RANDOM_STATE):
    km, labels = feb.fit_shape_clusters(X_train[donor_cols], n_clusters=n_clusters,
                                        random_state=random_state)
    labels_arr = labels.values
    T = curves.shape[1]
    mean_c = np.zeros((n_clusters, T), dtype=np.float32)
    std_c = np.zeros((n_clusters, T), dtype=np.float32)
    for c in range(n_clusters):
        members = curves[labels_arr == c]
        if len(members) == 0:
            continue
        mean_c[c] = members.mean(axis=0)
        std_c[c] = members.std(axis=0)
    return km, labels_arr, mean_c, std_c


def row_extra_features(full_z_row, window_mask, start, window, index_full):
    """Copies périodiques + saisonnalité propre, calculées sur TOUTE la
    courbe (comme à l'inférence réelle) avec seulement la fenêtre masquée
    synthétiquement retirée -- pas de fuite, cf. docstring du module."""
    masked_full = full_z_row.copy()
    seg = masked_full[start:start + window]
    seg[window_mask] = np.nan
    periodic = feb.periodic_copy_features(masked_full, lags=PERIODIC_LAGS,
                                          normalize=False, include_availability=True)
    periodic = np.nan_to_num(periodic, nan=0.0)  # NaN = source indispo (déjà signalé
                                                  # par le canal availability à 0)
    season = feb.same_curve_seasonality_feature(masked_full, index_full)
    return (periodic[start:start + window].astype(np.float32),
            season[start:start + window].astype(np.float32))


def make_batch_v2(curves, cal_full, cluster_labels, cluster_mean, cluster_std,
                  index_full, lo, hi, window, batch_size, rng):
    n = curves.shape[0]
    rows = rng.randint(0, n, size=batch_size)
    starts = rng.randint(lo, hi - window + 1, size=batch_size)
    x = np.stack([curves[r, s:s + window] for r, s in zip(rows, starts)])
    cal = np.stack([cal_full[s:s + window] for s in starts])
    mask = lb.sample_mask(x.shape, rng)
    x_in = x.copy()
    x_in[mask] = 0.0
    interp = lb.linear_interp_rows(x, mask)

    periodic_list, season_list, cmean_list, cstd_list = [], [], [], []
    for i, (r, s) in enumerate(zip(rows, starts)):
        periodic, season = row_extra_features(curves[r], mask[i], s, window, index_full)
        periodic_list.append(periodic)
        season_list.append(season)
        c = cluster_labels[r]
        cmean_list.append(cluster_mean[c, s:s + window])
        cstd_list.append(cluster_std[c, s:s + window])
    periodic_arr = np.stack(periodic_list)
    season_arr = np.stack(season_list)
    cmean_arr = np.stack(cmean_list).astype(np.float32)
    cstd_arr = np.stack(cstd_list).astype(np.float32)

    feats = np.concatenate([
        x_in[..., None], mask[..., None].astype(np.float32), interp[..., None],
        cal, periodic_arr, season_arr, cmean_arr[..., None], cstd_arr[..., None],
    ], axis=-1).astype(np.float32)
    return (torch.from_numpy(feats), torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(mask), torch.from_numpy(interp.astype(np.float32)))


# ---------------------------------------------------------------------------
# Modèle : lstm / gru / transformer (v1) + convlstm / attn (nouveaux)
# ---------------------------------------------------------------------------

class SeqImputerV2(nn.Module):
    def __init__(self, cell="lstm", hidden=64, layers=2, dropout=.1,
                n_channels=N_CHANNELS_V2, **_):
        super().__init__()
        self.cell = cell
        if cell == "transformer":
            self.proj = nn.Linear(n_channels, hidden)
            enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=4,
                dim_feedforward=4 * hidden, dropout=dropout, batch_first=True)
            self.body = nn.TransformerEncoder(enc, num_layers=layers)
            self.head = nn.Linear(hidden, 1)
        elif cell == "convlstm":
            self.conv = nn.Sequential(
                nn.Conv1d(n_channels, hidden, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.ReLU())
            self.body = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                                bidirectional=True, dropout=dropout if layers > 1 else 0.)
            self.head = nn.Linear(2 * hidden, 1)
        elif cell == "attn":
            self.body = nn.LSTM(n_channels, hidden, num_layers=layers, batch_first=True,
                                bidirectional=True, dropout=dropout if layers > 1 else 0.)
            self.attn = nn.MultiheadAttention(2 * hidden, num_heads=4, dropout=dropout,
                                              batch_first=True)
            self.norm = nn.LayerNorm(2 * hidden)
            self.head = nn.Linear(2 * hidden, 1)
        else:
            rnn = nn.LSTM if cell == "lstm" else nn.GRU
            self.body = rnn(n_channels, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.)
            self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        if self.cell == "transformer":
            h = self.body(self.proj(x))
        elif self.cell == "convlstm":
            c = self.conv(x.transpose(1, 2)).transpose(1, 2)
            h, _ = self.body(c)
        elif self.cell == "attn":
            h, _ = self.body(x)
            a, _ = self.attn(h, h, h)
            h = self.norm(h + a)
        else:
            h, _ = self.body(x)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Entraînement -- sélection du checkpoint sur la RMSE de validation temporelle
# ---------------------------------------------------------------------------

def train_config_v2(cfg, curves, cal_full, cluster_labels, cluster_mean, cluster_std,
                    index_full, max_steps, batch_size=48, eval_every=50, patience=4,
                    verbose=True):
    torch.manual_seed(RANDOM_STATE)
    rng = np.random.RandomState(RANDOM_STATE)
    window = cfg["window"]
    residual = cfg.get("residual", False)
    loss_kind = cfg.get("loss", "l1")

    model = SeqImputerV2(**cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss() if loss_kind == "mse" else nn.L1Loss()

    val_rng = np.random.RandomState(RANDOM_STATE + 1000)
    val_batches = [make_batch_v2(curves, cal_full, cluster_labels, cluster_mean, cluster_std,
                                 index_full, T_SPLIT, curves.shape[1], window, 64, val_rng)
                  for _ in range(6)]

    best_rmse, best_mae_at_best, best_state, bad, history = np.inf, None, None, 0, []
    t0 = time.time()
    for step in range(1, max_steps + 1):
        model.train()
        feats, y, mask, interp = make_batch_v2(curves, cal_full, cluster_labels, cluster_mean,
                                               cluster_std, index_full, 0, T_SPLIT, window,
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
                maes, mses = [], []
                for feats_v, y_v, mask_v, interp_v in val_batches:
                    out_v = model(feats_v)
                    pred_v = interp_v + out_v if residual else out_v
                    diff = pred_v[mask_v] - y_v[mask_v]
                    maes.append(float(diff.abs().mean()))
                    mses.append(float((diff ** 2).mean()))
            v_mae = float(np.mean(maes))
            v_rmse = float(np.sqrt(np.mean(mses)))
            history.append({"step": step, "val_mae_norm": v_mae, "val_rmse_norm": v_rmse})
            if verbose:
                print(f"    step {step:4d}  train={float(loss):.4f}  "
                      f"val_mae={v_mae:.4f}  val_rmse={v_rmse:.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if v_rmse < best_rmse - 1e-4:
                best_rmse, best_mae_at_best, bad = v_rmse, v_mae, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"    early stopping au step {step}", flush=True)
                    break

    model.load_state_dict(best_state)
    return model, best_rmse, best_mae_at_best, history


# ---------------------------------------------------------------------------
# Inférence sur une courbe complète (fenêtres glissantes moyennées)
# ---------------------------------------------------------------------------

def impute_curve_v2(model, values, cal_full, km, cluster_mean, cluster_std, index_full,
                    window, residual, stride=None):
    stride = stride or window // 2
    n = len(values)
    missing = np.isnan(values)
    obs = ~missing
    center, scale = lb.robust_scale(values[obs][None, :].astype(np.float32))
    center, scale = float(center), float(scale)
    z = np.clip((values - center) / scale, -lb.Z_CLIP, lb.Z_CLIP)
    idx = np.arange(n)
    z_interp = z.copy()
    z_interp[missing] = np.interp(idx[missing], idx[obs], z[obs])
    z_in = np.where(missing, 0.0, z)

    periodic_full = feb.periodic_copy_features(z, lags=PERIODIC_LAGS, normalize=False,
                                               include_availability=True).astype(np.float32)
    periodic_full = np.nan_to_num(periodic_full, nan=0.0)
    season_full = feb.same_curve_seasonality_feature(z, index_full).astype(np.float32)

    filled_raw = pd.Series(values, index=index_full).ffill().bfill()
    c = feb.cluster_of(filled_raw, km)
    cmean, cstd = cluster_mean[c], cluster_std[c]

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
                cal_full[sl], periodic_full[sl], season_full[sl],
                cmean[sl, None], cstd[sl, None],
            ], axis=-1).astype(np.float32)[None]
            out = model(torch.from_numpy(feats))[0].numpy()
            pred = z_interp[sl] + out if residual else out
            acc[sl] += pred
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
    index_full = X_train.index
    cal_full = build_calendar_full(index_full)

    print("Ajustement du clustering de forme (consensus voisins bon marché)...", flush=True)
    km, cluster_labels, cluster_mean, cluster_std = fit_cluster_consensus(
        X_train, donor_cols, curves, n_clusters=N_CLUSTERS)
    print(f"{curves.shape[0]} courbes donneuses x {curves.shape[1]} instants ; "
          f"{N_CHANNELS_V2} canaux ; split temporel à t={T_SPLIT} "
          f"({X_train.index[T_SPLIT]}) ; {N_CLUSTERS} clusters de forme", flush=True)

    try:
        results = json.load(open(PROGRESS_PATH))
    except FileNotFoundError:
        results = {}

    for name in ([only] if only else CONFIGS_V2):
        if name in results:
            print(f"=== {name} déjà fait, on saute ===", flush=True)
            continue
        print(f"=== {name} : {CONFIGS_V2[name]} ===", flush=True)
        t0 = time.time()
        model, best_rmse, best_mae, history = train_config_v2(
            CONFIGS_V2[name], curves, cal_full, cluster_labels, cluster_mean, cluster_std,
            index_full, max_steps)
        results[name] = {"config": CONFIGS_V2[name], "best_val_rmse_norm": best_rmse,
                         "val_mae_norm_at_best_rmse": best_mae, "history": history,
                         "train_seconds": time.time() - t0}
        with open(PROGRESS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        torch.save(model.state_dict(), f"lstm_v2_{name}.pt")
        print(f"  -> best val RMSE (normalisée) = {best_rmse:.4f} "
              f"(MAE à ce point = {best_mae:.4f})  ({time.time()-t0:.0f}s)\n", flush=True)
