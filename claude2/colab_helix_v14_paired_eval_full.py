"""Script autonome pour Google Colab : HELIX (mode "toutes les features") vs
TabPFN v14, protocole OFFICIEL APPARIÉ 10 repeats x 10 courbes -- strictement
comparable à `tabpfn_v14_confirm.py` (66,24 +/- 32,45 W) et à
`claude2/helix_v14_paired_eval.py` : mêmes seeds `RANDOM_STATE + i`, même
sous-échantillonnage `rng.choice(val_cols_full, size=10)` des 100 courbes du
split à 10 % (cf. `tabpfn_v14_ablation.run_config`, `n_val_sub=10`). Un seul
lot de 10 courbes par repeat (donc un seul entraînement HELIX par repeat,
10 au total sur tout le run -- pas 100 comme dans la variante précédente qui
validait sur les 100 courbes du split, cf. historique).

Descend de `claude2/helix_v14_paired_eval.py` (voir ce fichier pour la
justification détaillée du design : matrice brute, covariables temporelles
globales, canaux auxiliaires par courbe, blend post-traitement V14, set de
validation interne + visualisation train/val) -- ce script-ci l'exécute sans
dépendance au reste du repo (comme `claude/colab_generate_submission_tabpfn_v14_full.py` :
tout le pipeline TabPFN v14 est réinclus ici), et remplace la sélection
"50-100 courbes corrélées" par une sélection BEAUCOUP plus large
(`MAX_FEATURES`, voir mise en garde mémoire ci-dessous), pensée pour tourner
sur un GPU Colab 80 Go plutôt que le petit GPU local (5,67 Go) qui a fait
planter la version précédente (CUDA OOM à seulement F=97).

MISE EN GARDE MÉMOIRE -- lire avant de lancer
----------------------------------------------
Deux runs réels ont OOM avant ce correctif, et le second (F=778,
window_len=336, batch_size~14, sur un A100 80 Go) a permis d'identifier la
VRAIE cause -- l'estimation précédente de ce docstring ("F_max ~= 850",
supposant un coût linéaire en n_features) était fausse. Le crash a lieu très
précisément ici (cf. traceback PyPOTS) :

    attn_output_weights = attn_output_weights.mean(dim=1)   # torch/nn/functional.py

HELIX appelle `nn.MultiheadAttention(..., need_weights=True)` dans son
encodeur temporel ("time" layer) : ce `need_weights=True` désactive le
kernel fusionné économe en mémoire de PyTorch et force la matérialisation
d'un tenseur complet de poids d'attention, de taille

    batch_size x n_features x n_heads x window_len^2 x 4 octets

(vérifié : `14 x 778 x 4 x 336^2 x 4 ~= 4,58 Gio`, exactement l'allocation
qui a échoué). **`window_len` est donc le facteur DOMINANT, et il est
QUADRATIQUE** -- doubler `window_len` quadruple ce poste mémoire, alors que
doubler `n_features` (`MAX_FEATURES`) ne fait que le doubler. Le calcul
initial "combien de features tiennent en mémoire" était donc le mauvais
calcul : le vrai levier prioritaire est `window_len`, pas `MAX_FEATURES`.

Nouveaux défauts (aucun run réussi n'existe encore -- ce sont des valeurs
volontairement TRÈS prudentes, à faire grossir toi-même une dimension à la
fois une fois qu'un premier repeat aura réussi, plutôt que refaire confiance
à une extrapolation) :

    window_len   336 -> 96   (le levier prioritaire, quadratique)
    MAX_FEATURES 500 -> 150
    batch_size    32 -> 2     (seule dérogation aux défauts PyPOTS 1.5)

L'architecture HELIX reste celle publiée par PyPOTS 1.5 : d_model=256,
n_heads=8, n_layers=2, d_pe=16 et d_feature_embed=1.

Si ça passe, augmente une seule dimension à la fois dans cet ordre de
priorité (`window_len` d'abord, puis `MAX_FEATURES`, puis `batch_size`) et
relance -- le script reprend automatiquement les repeats déjà réussis avec
la même configuration batch2 (dans un checkpoint distinct). Le
script attrape les OOM CUDA proprement (repeat marqué non-fait, pas de perte
de progression).

Installation & utilisation dans Colab
---------------------------------------
1. Active un runtime GPU (Exécution > Modifier le type d'exécution > GPU,
   idéalement A100 pour les 80 Go).
2. Colle ce script dans une cellule (ou `%run`).
3. Mets les 3 CSV du challenge dans DATA_DIR (par défaut "/content/data") :
   upload direct, ou montage Drive si tu veux que le CHECKPOINT survive à
   une déconnexion (pointe alors OUT_DIR sur ton Drive monté).
4. Fournis ta clé TabPFN : soit un secret Colab nommé `TABPFN_API_KEY`
   (icône clé à gauche), soit la variable d'environnement, soit tu seras
   invité à la saisir (getpass, jamais affichée/journalisée).
5. Lance la cellule. Reprend automatiquement où il s'est arrêté si relancé
   (même PROGRESS_PATH) -- y compris après un OOM sur un repeat particulier.
"""

# =============================================================================
# 0. Installation (versions reproductibles ; HELIX est publié dans PyPOTS 1.5)
# =============================================================================
import os
import subprocess
import sys

# Doit être défini avant le premier import de torch pour que l'allocateur
# CUDA utilise des segments extensibles et limite la fragmentation entre
# repeats HELIX successifs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn-client", "matplotlib"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypots==1.5"], check=True)

import re
import gc
import time
import json
import logging
import warnings
import getpass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import tabpfn_client
from tabpfn_client import TabPFNRegressor as CloudTabPFNRegressor
import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif : on sauvegarde des PNG
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

print(f"GPU disponible : {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)}, "
         f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go)"
         if torch.cuda.is_available() else " -- CPU seul, HELIX sera très lent"),
      flush=True)

# =============================================================================
# 1. Configuration -- à adapter
# =============================================================================
DATA_DIR = "/content/data"
OUT_DIR = "/content"                # idéalement un dossier Drive monté pour survivre aux déconnexions
X_TRAIN_FILE = "X_train_78VdSWL.csv"
Y_TRAIN_FILE = "y_train_u0UkKEh.csv"

RUN_TAG = "helix_v14_paired_10x10_pypots15_mae_watts_batch2"
PROGRESS_PATH = os.path.join(OUT_DIR, f"{RUN_TAG}_progress.json")
RESULTS_PATH = os.path.join(OUT_DIR, f"{RUN_TAG}_results.json")
PLOTS_DIR = os.path.join(OUT_DIR, f"{RUN_TAG}_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42
N_REPEATS = 10
VAL_BATCH_SIZE = 10       # protocole 10x10 officiel : les 10 courbes de validation
                          # du repeat forment un seul lot (batch == tout val_cols)
N_CALIBRATION_COLS = 10   # prises exclusivement dans les 90 % de développement
NORMALIZE_HELIX_INPUTS = True
# Toutes les courbes continues, cibles comprises, doivent partager la même
# échelle. Mélanger des cibles en watts bruts et des donneurs normalisés
# déséquilibre fortement les pertes ORT/MIT natives de HELIX.
NORMALIZE_TARGET_CURVES = True
PERIODIC_LAGS = (24, 48, 96, 144, 336)

# Coeur de la demande "toutes les features" : nombre de courbes donneuses
# passées à HELIX en plus des 10 cibles du lot courant. None = littéralement
# TOUTES (~20 000) -- cf. mise en garde mémoire en tête de fichier : le vrai
# facteur mémoire dominant est window_len (quadratique), pas ce nombre --
# mais un F trop grand y contribue quand même (facteur linéaire). Valeur
# volontairement très prudente tant qu'aucun repeat n'a encore réussi ; à
# faire grossir APRÈS window_len une fois qu'un premier run passe.
MAX_FEATURES = 150  # ou None pour vraiment toutes les courbes (RISQUE OOM ÉLEVÉ)

# Config TabPFN v14 = BEST confirmée (README.md, 66,24 +/- 32,45 W / 63,24 W confirmé)
V14_BEST = dict(k=40, k_min=5, corr_threshold=0.15, euclid_weight=0.3, ratio_clip=(0.1, 10.0),
                extra_lags=(24,), vol_weight=0.8, vol_cap=0.6,
                periodic_alpha_strength=1.0, alpha_scale=1.0)
VOL_WEIGHT = V14_BEST["vol_weight"]
PERIODIC_ALPHA_STRENGTH = V14_BEST["periodic_alpha_strength"]

# Poids de l'ensemble post-traitement HELIX x TabPFN (au lieu d'un blend
# HELIX x interpolation seul) -- cf. mise en garde en tête de fichier : le
# prédécesseur SAITS (même famille PyPOTS, même script) perdait 98/100
# courbes face à TabPFN v14 MÊME avec blend alpha-interpolation
# (claude2/saits_v14_paired_v2_results.json : 93,95 W contre 64,73 W). Rien
# ne garantit que HELIX seul batte un jour TabPFN sur ce protocole -- on
# calcule donc EN PLUS un candidat qui mélange HELIX+blend avec la
# prédiction TabPFN elle-même (beaucoup plus forte qu'une simple
# interpolation), pour que HELIX apporte un signal additif plutôt que de
# jouer perdant en compétiteur autonome. 0.25 = poids volontairement prudent
# tant qu'aucun run réel n'a mesuré HELIX ; à retuner (ou pondérer par
# MAE de calibration inverse) une fois les premiers repeats terminés.
HELIX_ENSEMBLE_WEIGHT = 0.25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# window_len = LEVIER MÉMOIRE PRIORITAIRE (quadratique, cf. mise en garde en
# tête de fichier) -- augmente-le en dernier, pas en premier. 96 (1 jour) au
# lieu de 336 (1 semaine) tant qu'aucun repeat n'a réussi.
HELIX_OPTS = dict(
    aux_channels=True, window_len=96, stride=48,
    # Architecture et apprentissage PyPOTS 1.5 officiels. batch_size=2 est
    # l'unique dérogation au défaut 32 : la matrice enrichie et les poids
    # d'attention temporelle matérialisés rendent 32 impraticable sur Colab.
    n_layers=2, d_model=256, n_heads=8, d_pe=16, d_feature_embed=1, dropout=0.1,
    ort_weight=1.0, mit_weight=1.0,
    epochs=100, patience=None, batch_size=2, device=DEVICE,
    plot_training=True,
)


def _validate_tabpfn_api_key(key, source):
    """Nettoie et valide un token sans jamais inclure sa valeur dans une erreur."""
    if not isinstance(key, str):
        raise ValueError(f"{source} n'est pas une chaîne de caractères")

    key = key.strip()
    if not key:
        raise ValueError(f"{source} est vide")
    if not key.isascii():
        raise ValueError(f"{source} contient des caractères non ASCII")
    if any(char.isspace() or not 33 <= ord(char) <= 126 for char in key):
        raise ValueError(f"{source} contient un espace ou un caractère de contrôle")
    return key


def _get_tabpfn_api_key():
    key = os.environ.get("TABPFN_API_KEY")
    if key is not None:
        try:
            return _validate_tabpfn_api_key(key, "La variable d'environnement TABPFN_API_KEY")
        except ValueError as exc:
            raise SystemExit(f"{exc}. Corrigez ou supprimez cette variable puis relancez.") from None

    key = None
    try:
        from google.colab import userdata
        key = userdata.get("TABPFN_API_KEY")
    except Exception:
        pass
    if key is not None:
        try:
            return _validate_tabpfn_api_key(key, "Le secret Colab TABPFN_API_KEY")
        except ValueError as exc:
            raise SystemExit(f"{exc}. Corrigez ou supprimez ce secret puis relancez.") from None

    while True:
        key = getpass.getpass("Clé TABPFN_API_KEY (jamais affichée) : ")
        try:
            return _validate_tabpfn_api_key(key, "La clé saisie")
        except ValueError as exc:
            print(f"Clé refusée : {exc}. Collez uniquement le token TabPFN.", flush=True)


tabpfn_client.set_access_token(_get_tabpfn_api_key())


# =============================================================================
# 2. Pipeline TabPFN v14 (inline, copié/adapté de
#    feature_engineering_bench.py + tabpfn_v14_ablation.py, réduit aux
#    branches actives pour V14_BEST -- même simplification que
#    claude/colab_generate_submission_tabpfn_v14_full.py, mais ici en
#    variante ÉVALUATION (erreur vs y_train) plutôt que génération de
#    soumission, et avec tabpfn_client (cloud, comme le reste du repo) au
#    lieu du package tabpfn local -- pas de contention GPU avec HELIX.)
# =============================================================================

def fill_nan_with_interpolation(column: pd.Series) -> pd.Series:
    return column.interpolate(method="linear", limit_direction="both")


def run_lengths_for(missing_mask):
    n = len(missing_mask)
    run_len = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        if missing_mask[i]:
            j = i
            while j < n and missing_mask[j]:
                j += 1
            run_len[i:j] = j - i
            i = j
        else:
            i += 1
    return run_len


def edge_distance_for(missing_mask):
    n = len(missing_mask)
    edge_dist = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        if missing_mask[i]:
            j = i
            while j < n and missing_mask[j]:
                j += 1
            for p in range(i, j):
                edge_dist[p] = min(p - i, j - 1 - p)
            i = j
        else:
            i += 1
    return edge_dist


def gap_alpha(run_len, adaptive=True):
    if not adaptive:
        return 0.3
    if run_len <= 2:
        return 0.8
    elif run_len <= 6:
        return 0.4
    elif run_len <= 20:
        return 0.15
    else:
        return 0.0


def continuous_alpha(run_len, edge_dist, adaptive=True, edge_tau=1.5):
    base = gap_alpha(run_len, adaptive)
    if not adaptive:
        return base
    edge_boost = np.exp(-edge_dist / edge_tau)
    return np.clip(base + (1 - base) * edge_boost, 0.0, 0.95)


def curve_volatility(values):
    obs = values[~np.isnan(values)]
    if len(obs) < 3:
        return 0.0
    diffs = np.diff(obs)
    level = np.mean(np.abs(obs)) + 1e-9
    return float(np.std(diffs) / level)


def volatility_alpha_mult(vol, vol_weight=0.5, cap=0.6):
    vol_clipped = min(vol, cap) / cap
    return 1.0 - vol_weight * vol_clipped


def time_features_full_seasonality(index, groups=("hour", "dow", "dom", "year")):
    dt = pd.DatetimeIndex(pd.to_datetime(index))
    minute_of_day = dt.hour.to_numpy() * 60.0 + dt.minute.to_numpy()
    day_of_week = dt.dayofweek.to_numpy(dtype=float)
    day_of_month = dt.day.to_numpy(dtype=float) - 1.0
    week_of_year = dt.isocalendar().week.to_numpy(dtype=float) - 1.0
    month_of_year = dt.month.to_numpy(dtype=float) - 1.0
    day_of_year = dt.dayofyear.to_numpy(dtype=float) - 1.0
    annual_position = day_of_year + minute_of_day / 1440.0

    def cyclic(values, period):
        angle = 2.0 * np.pi * values / period
        return np.column_stack([np.sin(angle), np.cos(angle)])

    groups = set(groups)
    available = {
        "hour": cyclic(minute_of_day, 1440.0),
        "dow": cyclic(day_of_week, 7.0),
        "dom": cyclic(day_of_month, 31.0),
        "week": cyclic(week_of_year, 52.1775),
        "month": cyclic(month_of_year, 12.0),
        "year": cyclic(annual_position, 365.2425),
    }
    cols = [available[name] for name in ("hour", "dow", "dom", "week", "month", "year")
            if name in groups]
    cols.append((day_of_week >= 5).astype(float).reshape(-1, 1))
    return np.column_stack(cols)


def same_curve_seasonality_feature(values, index):
    dt_index = pd.to_datetime(index)
    bucket = dt_index.hour.values * 60 + dt_index.minute.values
    observed = ~np.isnan(values)
    df = pd.DataFrame({"bucket": bucket, "val": values})
    med_by_bucket = df.loc[observed].groupby("bucket")["val"].median()
    global_med = float(np.nanmedian(values)) if observed.any() else 0.0
    feat = pd.Series(bucket).map(med_by_bucket).fillna(global_med).values
    return feat.astype(float).reshape(-1, 1)


def periodic_copy_features(values, lags=(48, 96, 144, 336), include_summary=False,
                            normalize=False, include_availability=False):
    values = np.asarray(values, dtype=float)
    n = len(values)
    cols = []
    for lag in lags:
        back = np.full(n, np.nan)
        forward = np.full(n, np.nan)
        back[lag:] = values[:-lag]
        forward[:-lag] = values[lag:]
        cols.extend([back, forward])
    raw = np.column_stack(cols)
    availability = np.isfinite(raw).astype(float)
    if normalize:
        obs = values[np.isfinite(values)]
        center = np.median(obs)
        scale = max(1.4826 * np.median(np.abs(obs - center)),
                    0.02 * (abs(center) + 1.0), 1.0)
        raw = (raw - center) / scale
    if not include_summary:
        return np.column_stack([raw, availability]) if include_availability else raw
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(raw, axis=1)
        mad = np.nanmedian(np.abs(raw - median[:, None]), axis=1)
    count = np.isfinite(raw).sum(axis=1).astype(float)
    result = np.column_stack([raw, median, mad, count])
    return np.column_stack([result, availability]) if include_availability else result


def periodic_agreement(values, lags=(48, 96, 144, 336)):
    raw = periodic_copy_features(values, lags=lags)
    count = np.isfinite(raw).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(raw, axis=1)
        mad = np.nanmedian(np.abs(raw - med[:, None]), axis=1)
    obs = np.asarray(values)[np.isfinite(values)]
    scale = max(1.4826 * np.median(np.abs(obs - np.median(obs))), 1.0)
    return np.clip((count / raw.shape[1]) *
                   np.exp(-np.nan_to_num(mad, nan=np.inf) / scale), 0.0, 1.0)


def neighbor_rolling_features(scaled, window=5):
    consensus = scaled.mean(axis=1)
    s = pd.Series(consensus)
    roll_mean = s.rolling(window, center=True, min_periods=1).mean().values
    roll_std = s.rolling(window, center=True, min_periods=1).std().fillna(0.0).values
    return np.column_stack([roll_mean, roll_std])


def fit_shape_clusters(donor_df, n_clusters=15, random_state=RANDOM_STATE):
    filled = donor_df.ffill().bfill()
    Z = (filled - filled.mean()) / (filled.std() + 1e-9)
    km = KMeans(n_clusters=n_clusters, n_init=4, max_iter=100, random_state=random_state)
    labels = km.fit_predict(Z.T.values)
    labels = pd.Series(labels, index=donor_df.columns)
    return km, labels


def cluster_of(curve_filled, km):
    z = (curve_filled - curve_filled.mean()) / (curve_filled.std() + 1e-9)
    return int(km.predict(z.values.reshape(1, -1))[0])


def adaptive_top_k_corr_euclid_tunable(target_obs, pool_vals_obs, k_min, k_max,
                                        corr_threshold, euclid_weight):
    t_c = target_obs - target_obs.mean()
    d_c = pool_vals_obs - pool_vals_obs.mean(axis=0, keepdims=True)
    num = d_c.T @ t_c
    denom = np.sqrt((d_c ** 2).sum(axis=0) * (t_c ** 2).sum()) + 1e-9
    corr = num / denom
    t_scale = target_obs.std() + 1e-9
    euclid = np.sqrt(((d_c - t_c[:, None]) ** 2).mean(axis=0)) / t_scale
    euclid_score = 1.0 / (1.0 + euclid)
    combined = (1 - euclid_weight) * corr + euclid_weight * euclid_score
    order = np.argsort(-combined)
    above = order[corr[order] >= corr_threshold]
    chosen = above[:k_max] if len(above) >= k_min else order[:k_min]
    return chosen, corr[chosen]


def scale_neighbors_tunable(pool_sel, obs_idx, target_obs, ratio_clip):
    sel_obs = pool_sel[obs_idx, :]
    center = np.median(sel_obs, axis=0)
    floor = 0.02 * (np.abs(center) + 1.0)
    spread = np.maximum(1.4826 * np.median(np.abs(sel_obs - center), axis=0), floor)
    t_center = np.median(target_obs)
    t_spread = max(1.4826 * np.median(np.abs(target_obs - t_center)), 0.02 * (abs(t_center) + 1.0))
    ratio = np.clip(t_spread / spread, ratio_clip[0], ratio_clip[1])
    return (pool_sel - center) * ratio + t_center


def weighted_knn_pred_tunable(pool_vals, obs_idx, miss_idx, target_obs, k,
                               k_min=5, corr_threshold=0.15, euclid_weight=0.3,
                               ratio_clip=(0.1, 10.0)):
    top_idx, corr = adaptive_top_k_corr_euclid_tunable(
        target_obs, pool_vals[obs_idx, :], k_min, k, corr_threshold, euclid_weight)
    weights = np.clip(corr, 1e-3, None)
    weights = weights / weights.sum()
    pool_sel = pool_vals[:, top_idx]
    scaled = scale_neighbors_tunable(pool_sel, obs_idx, target_obs, ratio_clip)
    return (scaled[miss_idx, :] * weights).sum(axis=1), top_idx, scaled


def local_regression_pred(make_regressor, scaled_neighbors, obs_idx, miss_idx, target_obs, extra_feats):
    X_obs = np.hstack([scaled_neighbors[obs_idx, :], extra_feats[obs_idx, :]])
    X_miss = np.hstack([scaled_neighbors[miss_idx, :], extra_feats[miss_idx, :]])
    reg = make_regressor()
    reg.fit(X_obs, target_obs)
    return reg.predict(X_miss)


class MedianTabPFN:
    def __init__(self, random_state=RANDOM_STATE):
        self.reg = CloudTabPFNRegressor(random_state=random_state)

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X, output_type="median")


def build_v14_features(s, pool_vals, km, cluster_col_idx, tfeat, index, o):
    observed = s.notna().to_numpy(); missing = ~observed
    if not missing.any():
        return None
    obs_idx, miss_idx = np.flatnonzero(observed), np.flatnonzero(missing)
    target_obs = s.to_numpy()[obs_idx]
    feat_values = s.to_numpy()
    interp_full = fill_nan_with_interpolation(s)
    cluster = cluster_of(interp_full, km)
    same_idx = cluster_col_idx.get(cluster)
    sub_vals = (pool_vals[:, same_idx]
                if same_idx is not None and len(same_idx) >= max(2 * o["k"], 30)
                else pool_vals)

    knn_pred, _, scaled = weighted_knn_pred_tunable(
        sub_vals, obs_idx, miss_idx, target_obs, o["k"],
        k_min=o["k_min"], corr_threshold=o["corr_threshold"],
        euclid_weight=o["euclid_weight"], ratio_clip=o["ratio_clip"])

    lags = (48, 96, 144, 336) + tuple(o["extra_lags"])
    periodic = periodic_copy_features(feat_values, lags=lags, include_summary=True,
                                       normalize=True, include_availability=True)
    seasonal = same_curve_seasonality_feature(feat_values, index)
    extra = np.hstack([tfeat, seasonal, periodic])
    extra = np.hstack([extra, neighbor_rolling_features(scaled)])  # neighbor_rolling=True dans V14_BEST

    return obs_idx, miss_idx, target_obs, interp_full, knn_pred, scaled, extra, missing, lags


def predict_column_v14(col, s, pool_vals, km, cluster_col_idx, tfeat, index, opts):
    o = opts
    error = None
    built = build_v14_features(s, pool_vals, km, cluster_col_idx, tfeat, index, o)
    if built is None:
        return col, s.copy(), None
    obs_idx, miss_idx, target_obs, interp_full, knn_pred, scaled, extra, missing, lags = built

    pred_neighbors = None
    if len(obs_idx) >= 15:
        for attempt in range(3):
            try:
                pred_neighbors = local_regression_pred(
                    MedianTabPFN, scaled, obs_idx, miss_idx, target_obs, extra)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(2 * (attempt + 1))
    if pred_neighbors is None:
        pred_neighbors = knn_pred

    runs = run_lengths_for(missing)
    edges = edge_distance_for(missing)
    alpha = np.array([continuous_alpha(runs[i], edges[i], True) for i in miss_idx])
    p_agree = periodic_agreement(s.to_numpy(), lags=lags)[miss_idx]
    alpha *= 1.0 - o["periodic_alpha_strength"] * p_agree
    vol = curve_volatility(s.to_numpy())
    alpha *= volatility_alpha_mult(vol, vol_weight=o["vol_weight"], cap=o["vol_cap"])  # volatility_alpha=True dans V14_BEST
    alpha = np.clip(alpha * o["alpha_scale"], 0.0, 0.95)
    pred = alpha * interp_full.to_numpy()[miss_idx] + (1 - alpha) * pred_neighbors

    filled = s.copy()
    filled.iloc[miss_idx] = pred
    return col, filled, error


def complete_column_v14(col, s, y_true_col, pool_vals, km, cluster_col_idx, tfeat, index, opts):
    """Variante évaluation de predict_column_v14 : erreur absolue vs y_train
    aux positions réellement manquantes, au lieu d'une soumission. Renvoie
    aussi la prédiction TabPFN elle-même (mêmes positions valides que
    abs_err) pour permettre l'ensemble HELIX x TabPFN de eval_helix_repeat."""
    observed = s.notna().to_numpy(); missing = ~observed
    if not missing.any():
        return col, None, None, None
    miss_idx = np.flatnonzero(missing)
    _, filled, error = predict_column_v14(col, s, pool_vals, km, cluster_col_idx, tfeat, index, opts)
    pred = filled.to_numpy()[miss_idx]
    y_true = y_true_col.to_numpy()[miss_idx]
    valid = ~np.isnan(y_true)
    abs_err = np.abs(pred[valid] - y_true[valid])
    return col, abs_err, pred[valid], error


def eval_tabpfn_repeat(X_train, y_train, donor_cols, dev_cols, val_cols, opts):
    donor_df = X_train[donor_cols]
    dev_true_df = y_train[dev_cols]
    val_holed_df = X_train[val_cols]

    pool = pd.concat([donor_df, dev_true_df], axis=1)
    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    km, donor_labels = fit_shape_clusters(donor_df, n_clusters=15, random_state=RANDOM_STATE)
    dev_filled = dev_true_df.ffill().bfill()
    Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
    pool_labels = pd.concat([donor_labels, dev_labels])
    pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
    cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
    tfeat = time_features_full_seasonality(val_holed_df.index, ("hour", "dow", "dom", "year"))

    abs_err_by_col = {}
    pred_by_col = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(complete_column_v14, c, val_holed_df[c], y_train[c],
                                    pool_vals, km, cluster_col_idx, tfeat,
                                    val_holed_df.index, opts) for c in val_cols]
        for fut in as_completed(futures):
            col, abs_err, pred, error = fut.result()
            if abs_err is not None:
                abs_err_by_col[col] = abs_err
                pred_by_col[col] = pred
            if error:
                print(f"  [tabpfn] repli sur {col}: {error}", flush=True)
    return abs_err_by_col, pred_by_col


# =============================================================================
# 3. Pipeline HELIX natif (inline, copié de claude2/helix_v14_paired_eval.py
#    -- voir ce fichier pour la justification détaillée de chaque choix de
#    design. Seuls changements : select_donor_features gère MAX_FEATURES=None
#    (toutes les courbes), et eval_helix_repeat attrape les OOM CUDA
#    proprement (repeat marqué non-fait -> juste réduire MAX_FEATURES/
#    window_len/batch_size et relancer, la progression n'est pas perdue).
# =============================================================================

def select_donor_features(X_train, val_cols, donor_cols, max_features,
                            k_per_col=25, corr_threshold=0.15, euclid_weight=0.3):
    """Union, sur les val_cols du lot, des donneurs les plus corrélés,
    plafonnée à max_features. Si max_features est None : renvoie TOUS les
    donneurs directement (pas de classement par corrélation -- inutile et
    coûteux sur ~20 000 colonnes pour au final toutes les garder).

    Allocation en TOURNIQUET (round-robin) sur les colonnes, pas au premier
    arrivé premier servi. Avec l'ancienne version, dès que les premières
    colonnes du lot (dans l'ordre de `val_cols`) remplissaient le quota
    (k_per_col=25 x ~6 colonnes >= max_features=150), la boucle s'arrêtait :
    les 4 dernières des 10 courbes cibles du lot ne contribuaient alors AUCUN
    donneur, et HELIX recevait un pool de features corrélé à une poignée de
    cibles seulement au lieu des 10 -- un biais silencieux qui dépend de
    l'ordre arbitraire (résultat de `make_homogeneous_batches`), pas de la
    pertinence. Ici on prend le meilleur donneur de chaque colonne, puis le
    2e de chaque colonne, etc., jusqu'à max_features : chaque cible du lot
    est représentée aussi équitablement que possible."""
    if max_features is None:
        return list(donor_cols)
    donor_vals = X_train[donor_cols].to_numpy(dtype=np.float64)
    per_col_ranked = []
    for col in val_cols:
        s = X_train[col]
        obs_idx = np.flatnonzero(s.notna().to_numpy())
        if len(obs_idx) < 5:
            per_col_ranked.append([])
            continue
        target_obs = s.to_numpy()[obs_idx]
        pool_obs = donor_vals[obs_idx, :]
        pool_obs = np.where(np.isnan(pool_obs), np.nanmedian(pool_obs, axis=0, keepdims=True), pool_obs)
        idx, _ = adaptive_top_k_corr_euclid_tunable(
            target_obs, pool_obs, k_per_col // 5, k_per_col, corr_threshold, euclid_weight)
        per_col_ranked.append(list(idx))  # déjà trié par score décroissant

    chosen_idx, seen = [], set()
    rank = 0
    while len(chosen_idx) < max_features and any(rank < len(lst) for lst in per_col_ranked):
        for lst in per_col_ranked:
            if rank < len(lst):
                j = lst[rank]
                if j not in seen:
                    seen.add(j)
                    chosen_idx.append(j)
                    if len(chosen_idx) >= max_features:
                        break
        rank += 1
    return [donor_cols[j] for j in chosen_idx[:max_features]]


def global_time_covariates(index):
    """Features calendaires TabPFN v14, partagées et ajoutées une fois."""
    return time_features_full_seasonality(index, ("hour", "dow", "dom", "year"))


def per_curve_aux_channels(raw_curves, keep_idx=None):
    """Canal auxiliaire PAR courbe : accord périodique seul (signal
    dynamique dans le temps, pas un résumé figé par colonne).

    La volatilité (feb.curve_volatility) a été retirée -- coupe de
    priorité n°1 sur 250 features à supprimer (cf. discussion) : c'est un
    scalaire par courbe diffusé CONSTANT sur tous les pas de temps, donc
    la feature la moins informative possible pour une attention
    temporelle (elle "paie" le même coût qu'un signal réellement variable
    pour zéro variation réelle) -- et pour la cible, cette même info est
    de toute façon déjà réinjectée via le blend post-traitement
    (alpha_blend_post), donc redondante de toute façon.

    `keep_idx` (2e vague de coupe) : restreint le calcul aux colonnes
    d'index `keep_idx` (typiquement cibles + calibration) au lieu de
    TOUTES les courbes -- réduit F directement plutôt que de garder des
    colonnes inutiles. L'accord périodique des DONNEURS n'est utilisé nulle
    part en aval (contrairement à celui de la cible, qui alimente déjà
    alpha_blend_post), donc moins prioritaire à exposer explicitement au
    modèle : autant laisser HELIX le redécouvrir lui-même via l'attention
    sur les courbes brutes des donneurs, déjà présentes."""
    if keep_idx is None:
        keep_idx = range(raw_curves.shape[1])
    T = raw_curves.shape[0]
    agree_cols = np.zeros((T, len(keep_idx)))
    for out_j, in_j in enumerate(keep_idx):
        agree_cols[:, out_j] = periodic_agreement(raw_curves[:, in_j])
    return agree_cols


def periodic_lag_channels(raw_curves, lags, availability_lags=None):
    """Valeurs retardées/avancées (past/future), sans remplir les trous.

    Les indicateurs de disponibilité (past_available/future_available) ne
    sont calculés QUE pour `availability_lags` (sous-ensemble de `lags`,
    tous par défaut si None) -- coupe de priorité n°2 sur 250 features à
    supprimer (cf. discussion) : ces indicateurs ne font que dupliquer le
    masque de trous que HELIX perçoit déjà nativement en interne (DMSA) via
    le NaN du canal de valeur lui-même -- l'info est déjà là, le flag
    explicite est redondant. On les garde seulement pour les lags courts
    (24/48/96h, les plus fréquemment décisifs) et on les coupe pour les
    lags longs (144h, 336h) plutôt que de les garder partout.
    """
    if availability_lags is None:
        availability_lags = lags
    availability_lags = set(availability_lags)
    T, F = raw_curves.shape
    blocks = []
    value_masks = []
    for lag in lags:
        past = np.full((T, F), np.nan, dtype=np.float64)
        future = np.full((T, F), np.nan, dtype=np.float64)
        if lag < T:
            past[lag:] = raw_curves[:-lag]
            future[:-lag] = raw_curves[lag:]
        blocks.extend([past, future])
        value_masks.extend([np.ones(F, dtype=bool), np.ones(F, dtype=bool)])
        if lag in availability_lags:
            past_available = np.isfinite(past).astype(np.float64)
            future_available = np.isfinite(future).astype(np.float64)
            blocks.extend([past_available, future_available])
            value_masks.extend([np.zeros(F, dtype=bool), np.zeros(F, dtype=bool)])
    return np.hstack(blocks), np.concatenate(value_masks)


def periodic_summary_channels(raw_curves, lags, include_count=True):
    """Médiane et MAD des copies périodiques, comme TabPFN v14.

    Le sous-bloc "nombre de copies disponibles" (`count`) est optionnel
    (`include_count=False` le retire) -- coupe de priorité (2e vague sur
    250 features à supprimer, cf. discussion) : comme les flags de
    disponibilité des lags et l'ex-volatilité, c'est un signal dérivé du
    masque de trous que HELIX perçoit déjà nativement en interne (DMSA) --
    redondant."""
    T, F = raw_curves.shape
    copies = []
    for lag in lags:
        past = np.full((T, F), np.nan, dtype=np.float64)
        future = np.full((T, F), np.nan, dtype=np.float64)
        if lag < T:
            past[lag:] = raw_curves[:-lag]
            future[:-lag] = raw_curves[lag:]
        copies.extend([past, future])
    stacked = np.stack(copies, axis=1)  # T x (2 * n_lags) x F
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(stacked, axis=1)
        mad = np.nanmedian(np.abs(stacked - median[:, None, :]), axis=1)
    # Ordre par famille pour faciliter les masques: toutes les médianes,
    # tous les MAD, puis (optionnellement) tous les compteurs.
    if include_count:
        count = np.isfinite(stacked).sum(axis=1).astype(np.float64)
        values = np.hstack([median, mad, count])
        normalize_mask = np.r_[np.ones(2 * F, dtype=bool), np.zeros(F, dtype=bool)]
    else:
        values = np.hstack([median, mad])
        normalize_mask = np.ones(2 * F, dtype=bool)
    return values, normalize_mask


def per_curve_seasonality_channels(raw_curves, index):
    """Médiane observée de chaque courbe pour la demi-heure correspondante."""
    return np.hstack([
        same_curve_seasonality_feature(raw_curves[:, j], index)
        for j in range(raw_curves.shape[1])
    ])


def tabpfn_neighbor_summary_channels(priority_curves, donor_curves, k=40,
                                     k_min=5, corr_threshold=0.15,
                                     euclid_weight=0.3, ratio_clip=(0.1, 10.0)):
    """Version mémoire-sûre des voisins TabPFN: consensus + rolling mean/std."""
    donor_curves = np.asarray(donor_curves, dtype=np.float64)
    donor_medians = np.nanmedian(donor_curves, axis=0)
    donor_medians = np.where(np.isfinite(donor_medians), donor_medians, 0.0)
    donor_filled = np.where(np.isfinite(donor_curves), donor_curves, donor_medians[None, :])
    channels = []
    for j in range(priority_curves.shape[1]):
        target = priority_curves[:, j]
        obs_idx = np.flatnonzero(np.isfinite(target))
        if len(obs_idx) < 5 or donor_filled.shape[1] == 0:
            channels.append(np.zeros((len(target), 3), dtype=np.float64))
            continue
        top_idx, corr = adaptive_top_k_corr_euclid_tunable(
            target[obs_idx], donor_filled[obs_idx], k_min,
            min(k, donor_filled.shape[1]), corr_threshold, euclid_weight)
        selected = donor_filled[:, top_idx]
        scaled = scale_neighbors_tunable(selected, obs_idx, target[obs_idx], ratio_clip)
        weights = np.clip(corr, 1e-3, None)
        weights /= weights.sum()
        consensus = scaled @ weights
        consensus_series = pd.Series(consensus)
        rolling_mean = consensus_series.rolling(5, center=True, min_periods=1).mean().to_numpy()
        rolling_std = consensus_series.rolling(5, center=True, min_periods=1).std().fillna(0.0).to_numpy()
        channels.append(np.column_stack([consensus, rolling_mean, rolling_std]))
    return np.hstack(channels)


def robust_scale_matrix(mat, normalize_mask=None):
    center = np.nanmedian(mat, axis=0)
    mad = np.nanmedian(np.abs(mat - center), axis=0)
    scale = np.maximum(1.4826 * mad, 0.02 * (np.abs(center) + 1.0))
    scale = np.where((scale < 1e-6) | ~np.isfinite(scale), 1.0, scale)
    center = np.where(np.isfinite(center), center, 0.0)
    if normalize_mask is not None:
        normalize_mask = np.asarray(normalize_mask, dtype=bool)
        if normalize_mask.shape != (mat.shape[1],):
            raise ValueError("normalize_mask incompatible avec le nombre de features HELIX")
        center = np.where(normalize_mask, center, 0.0)
        scale = np.where(normalize_mask, scale, 1.0)
    return (mat - center) / scale, center, scale


def prepare_helix_matrix(mat, normalize, normalize_mask=None):
    """Prépare les entrées et conserve l'inverse nécessaire aux sorties/MAE."""
    if normalize:
        return robust_scale_matrix(mat, normalize_mask)
    F = mat.shape[1]
    return mat.copy(), np.zeros(F, dtype=float), np.ones(F, dtype=float)


def curve_profile_frame(X, columns):
    """Profils sans y_true utilisés pour composer des lots comparables."""
    rows = []
    for col in columns:
        values = X[col].to_numpy(dtype=np.float64)
        observed = np.isfinite(values)
        finite = values[observed]
        missing = ~observed
        runs = run_lengths_for(missing)
        rows.append([
            np.nanmedian(finite) if finite.size else 0.0,
            1.4826 * np.nanmedian(np.abs(finite - np.nanmedian(finite))) if finite.size else 0.0,
            curve_volatility(values),
            float(missing.mean()),
            float(runs[missing].mean()) if missing.any() else 0.0,
            float(np.nanmean(periodic_agreement(values))),
        ])
    profile = np.asarray(rows, dtype=np.float64)
    med = np.nanmedian(profile, axis=0)
    mad = np.nanmedian(np.abs(profile - med), axis=0)
    denom = np.where(1.4826 * mad > 1e-9, 1.4826 * mad, 1.0)
    profile = (np.where(np.isfinite(profile), profile, med) - med) / denom
    return pd.DataFrame(profile, index=list(columns))


def make_homogeneous_batches(X, val_cols, dev_cols, batch_size, n_calibration, seed):
    """Lots de validation compacts et calibrations proches, sans utiliser y."""
    # Une standardisation commune est indispensable pour que les distances
    # validation-développement soient dans le même repère.
    joint_profile = curve_profile_frame(X, list(val_cols) + list(dev_cols))
    val_profile = joint_profile.loc[list(val_cols)]
    dev_profile = joint_profile.loc[list(dev_cols)]
    remaining = list(np.random.RandomState(seed).permutation(val_cols))
    batches = []
    while remaining:
        anchor = remaining.pop(0)
        anchor_vec = val_profile.loc[anchor].to_numpy()
        nearest = sorted(
            remaining,
            key=lambda col: float(np.sum((val_profile.loc[col].to_numpy() - anchor_vec) ** 2)),
        )[:batch_size - 1]
        batch = [anchor] + nearest
        nearest_set = set(nearest)
        remaining = [col for col in remaining if col not in nearest_set]
        batches.append(batch)

    available_dev = set(dev_cols)
    calibration_batches = []
    for batch in batches:
        centroid = val_profile.loc[batch].to_numpy().mean(axis=0)
        ranked = sorted(
            available_dev,
            key=lambda col: float(np.sum((dev_profile.loc[col].to_numpy() - centroid) ** 2)),
        )
        selected = ranked[:n_calibration]
        if len(selected) != n_calibration:
            raise RuntimeError("Pas assez de courbes de développement pour la calibration")
        calibration_batches.append(selected)
        available_dev.difference_update(selected)
    return batches, calibration_batches


def build_windows(mat, window_len, stride):
    T = mat.shape[0]
    window_len = min(window_len, T)
    starts = list(range(0, T - window_len + 1, stride))
    if not starts or starts[-1] + window_len < T:
        starts.append(T - window_len)
    starts = sorted(set(starts))
    windows = np.stack([mat[s:s + window_len] for s in starts], axis=0)
    return windows, starts


def build_nonoverlapping_windows(mat, window_len):
    """Découpe toute la chronologie une seule fois et padde le dernier bloc."""
    T, F = mat.shape
    windows = []
    for start in range(0, T, window_len):
        block = mat[start:start + window_len]
        if len(block) < window_len:
            padded = np.full((window_len, F), np.nan, dtype=mat.dtype)
            padded[:len(block)] = block
            block = padded
        windows.append(block)
    return np.stack(windows, axis=0)


def make_official_calibration_set(scaled, center, scale, y_train,
                                  raw_cols, calibration_cols, window_len):
    """Crée X/X_ori avec vérité uniquement sur les trous de calibration."""
    X_ori = scaled.copy()
    positions = {col: j for j, col in enumerate(raw_cols)}
    n_truth = 0
    for col in calibration_cols:
        j = positions[col]
        truth = y_train[col].to_numpy(dtype=np.float64)
        holes = np.isnan(scaled[:, j]) & np.isfinite(truth)
        X_ori[holes, j] = (truth[holes] - center[j]) / scale[j]
        n_truth += int(holes.sum())
    if n_truth == 0:
        raise RuntimeError("Aucun vrai trou disponible pour la calibration HELIX")
    return {
        "X": build_nonoverlapping_windows(scaled, window_len),
        "X_ori": build_nonoverlapping_windows(X_ori, window_len),
    }, n_truth


def reassembly_taper(window_len, floor=0.15):
    """Poids triangulaire (1.0 au centre, `floor` aux bords) pour pondérer
    chaque fenêtre lors du réassemblage. L'attention temporelle de HELIX a
    structurellement moins de contexte des deux côtés près des bords d'une
    fenêtre (une position en tout début/fin de fenêtre ne voit le passé/futur
    que d'un seul côté) -- une moyenne uniforme sur les fenêtres qui se
    chevauchent (`stride < window_len`) traite pourtant ces positions bord à
    égalité avec les positions centrales, alors qu'elles sont typiquement
    moins fiables. `floor` > 0 (pas 0) est indispensable : aux deux extrémités
    de la chronologie, certaines positions ne sont couvertes que par UNE
    seule fenêtre (souvent en tant que bord, cf. `build_windows`), un poids
    nul y donnerait un dénominateur nul dans `reassemble`."""
    if window_len <= 1:
        return np.ones(window_len, dtype=np.float64)
    center = (window_len - 1) / 2.0
    dist = np.abs(np.arange(window_len) - center) / center
    return floor + (1.0 - floor) * (1.0 - dist)


def reassemble(imputed_windows, starts, window_len, T, F):
    taper = reassembly_taper(window_len)[:, None]  # (window_len, 1), diffusé sur les F courbes
    out_sum = np.zeros((T, F))
    out_cnt = np.zeros((T, F))
    for w, s in zip(imputed_windows, starts):
        out_sum[s:s + window_len] += w * taper
        out_cnt[s:s + window_len] += taper
    return out_sum / np.maximum(out_cnt, 1e-9)


def save_json_atomic(payload, path, indent=None):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=indent)
    os.replace(tmp_path, path)


def compute_repeat_scores(rec):
    """Calcule les scores appariés d'un repeat complet, uniquement sur ses vrais trous."""
    tb = {c: np.asarray(v, dtype=float) for c, v in rec["tabpfn"].items()}
    hr = {c: np.asarray(v, dtype=float) for c, v in rec["helix_raw"].items()}
    hb = {c: np.asarray(v, dtype=float) for c, v in rec["helix_blend"].items()}
    he = {c: np.asarray(v, dtype=float) for c, v in rec["helix_ensemble"].items()}
    common = [c for c in rec["val_cols"] if c in tb and c in hr and c in hb and c in he]
    if len(common) != len(rec["val_cols"]):
        raise RuntimeError(f"Repeat incomplet : {len(common)}/{len(rec['val_cols'])} courbes appariées")
    wins = {"helix_blend": 0, "tabpfn": 0, "tie": 0}
    wins_ensemble = {"helix_ensemble": 0, "tabpfn": 0, "tie": 0}
    for col in common:
        tab_score, helix_score, ens_score = tb[col].mean(), hb[col].mean(), he[col].mean()
        if helix_score < tab_score - 1e-9:
            wins["helix_blend"] += 1
        elif tab_score < helix_score - 1e-9:
            wins["tabpfn"] += 1
        else:
            wins["tie"] += 1
        if ens_score < tab_score - 1e-9:
            wins_ensemble["helix_ensemble"] += 1
        elif tab_score < ens_score - 1e-9:
            wins_ensemble["tabpfn"] += 1
        else:
            wins_ensemble["tie"] += 1
    return {
        "tabpfn": float(np.concatenate([tb[c] for c in common]).mean()),
        "helix_raw": float(np.concatenate([hr[c] for c in common]).mean()),
        "helix_blend": float(np.concatenate([hb[c] for c in common]).mean()),
        "helix_ensemble": float(np.concatenate([he[c] for c in common]).mean()),
        "wins": wins,
        "wins_ensemble": wins_ensemble,
    }


def compute_tabpfn_validation_score(rec):
    """MAE TabPFN sur les vrais trous des 10 courbes de validation du repeat."""
    missing = [col for col in rec["val_cols"] if col not in rec["tabpfn"]]
    if missing:
        raise RuntimeError(f"Résultat TabPFN incomplet : {len(missing)} courbes manquantes")
    errors = [np.asarray(rec["tabpfn"][col], dtype=float) for col in rec["val_cols"]]
    non_empty = [values for values in errors if values.size]
    if not non_empty:
        raise RuntimeError("Résultat TabPFN vide sur la validation")
    concatenated = np.concatenate(non_empty)
    if not np.isfinite(concatenated).all():
        raise RuntimeError("Résultat TabPFN non fini sur la validation")
    return float(concatenated.mean())


def compute_tabpfn_subset_score(rec, columns):
    """MAE TabPFN sur un sous-ensemble précis des colonnes du repeat."""
    missing = [col for col in columns if col not in rec["tabpfn"]]
    if missing:
        raise RuntimeError(f"Résultat TabPFN absent pour {len(missing)} colonne(s) du lot")
    errors = [np.asarray(rec["tabpfn"][col], dtype=float) for col in columns]
    non_empty = [values for values in errors if values.size]
    if not non_empty:
        raise RuntimeError("Résultat TabPFN vide pour le lot")
    values = np.concatenate(non_empty)
    if not np.isfinite(values).all():
        raise RuntimeError("Résultat TabPFN non fini pour le lot")
    return float(values.mean())


class _LossCapture(logging.Handler):
    _PATTERN = re.compile(
        r"Epoch\s+(\d+)\s*-\s*training loss[^:]*:\s*([0-9.eE+-]+)"
        r"(?:,\s*validation[^:]*:\s*([0-9.eE+-]+))?"
    )

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.epochs, self.train_losses, self.val_losses = [], [], []
        self.interrupted = False
        self.patience_exceeded = False

    def emit(self, record):
        message = record.getMessage()
        if "Training got interrupted by the user" in message:
            self.interrupted = True
        if "Exceeded the training patience" in message:
            self.patience_exceeded = True
        m = self._PATTERN.search(message)
        if m:
            self.epochs.append(int(m.group(1)))
            self.train_losses.append(float(m.group(2)))
            self.val_losses.append(float(m.group(3)) if m.group(3) else None)

    def best_validation(self):
        defined = [(epoch, value) for epoch, value in zip(self.epochs, self.val_losses)
                   if value is not None and np.isfinite(value)]
        return min(defined, key=lambda pair: pair[1]) if defined else (None, None)


def plot_training_curve(train_losses, val_losses, out_path, title):
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, label="train MAE ORT+MIT (normalisée)",
            color="#4C72B0", marker="o", ms=3)
    val_defined = [v for v in val_losses if v is not None]
    if val_defined:
        ax.plot(epochs[:len(val_defined)], val_defined, label="validation MAE non normalisée (W, calibration)",
                 color="#DD8452", marker="o", ms=3)
        best_idx = int(np.argmin(val_defined))
        ax.scatter([best_idx + 1], [val_defined[best_idx]], marker="*", s=130,
                   color="#C44E52", label=f"best val={val_defined[best_idx]:.4g}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train normalisé / validation en W")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def alpha_blend_post(s_series, miss_idx, pred):
    missing = s_series.isna().to_numpy()
    runs = run_lengths_for(missing)
    edges = edge_distance_for(missing)
    alpha = np.array([continuous_alpha(runs[i], edges[i], True) for i in miss_idx])
    p_agree = periodic_agreement(s_series.to_numpy())[miss_idx]
    alpha *= 1.0 - PERIODIC_ALPHA_STRENGTH * p_agree
    vol = curve_volatility(s_series.to_numpy())
    alpha *= volatility_alpha_mult(vol, vol_weight=VOL_WEIGHT, cap=0.6)
    alpha = np.clip(alpha, 0.0, 0.95)
    interp = fill_nan_with_interpolation(s_series).to_numpy()[miss_idx]
    return alpha * interp + (1 - alpha) * pred


def _is_cuda_oom(exc):
    """Reconnaît aussi les OOM que PyPOTS enveloppe dans un RuntimeError."""
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, torch.OutOfMemoryError):
            return True
        message = str(current).lower()
        if "cuda" in message and "out of memory" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _release_helix_cuda(helix, repeat_idx):
    """Supprime toutes les références PyPOTS avant de vider le cache CUDA."""
    if helix is not None:
        try:
            # PyPOTS conserve notamment un optimiseur et une copie du meilleur
            # state_dict. Les détacher explicitement évite de dépendre des
            # cycles de références internes de la version installée.
            helix.optimizer = None
            helix.best_model_dict = None
            if getattr(helix, "model", None) is not None:
                helix.model.to("cpu")
                helix.model = None
        except Exception as cleanup_exc:
            print(f"  [helix] nettoyage du modèle incomplet "
                  f"({type(cleanup_exc).__name__}), poursuite du nettoyage CUDA.", flush=True)

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        allocated = torch.cuda.memory_allocated() / 2 ** 30
        reserved = torch.cuda.memory_reserved() / 2 ** 30
        print(f"  [helix] repeat {repeat_idx + 1}: mémoire après nettoyage CUDA: "
              f"{allocated:.2f} Gio alloués, {reserved:.2f} Gio réservés", flush=True)


def eval_helix_repeat(X_train, y_train, val_cols, calibration_cols, donor_cols,
                      opts, tabpfn_pred_by_col, repeat_idx=0, batch_idx=0):
    from pypots.imputation import HELIX
    from pypots.nn.modules.loss import Criterion, MAE
    from pypots.utils.logging import logger as pypots_logger

    feat_cols = select_donor_features(X_train, val_cols, donor_cols, MAX_FEATURES)
    cols = list(val_cols) + list(calibration_cols) + feat_cols
    n_target = len(val_cols)
    print(f"  [helix] repeat {repeat_idx + 1}, lot {batch_idx + 1}: {len(cols)} courbes brutes "
          f"({n_target} cibles + {len(calibration_cols)} calibration + "
          f"{len(feat_cols)} donneurs)", flush=True)

    raw_curves = X_train[cols].to_numpy(dtype=np.float64)
    tcov = global_time_covariates(X_train.index)
    n_priority_curves = n_target + len(calibration_cols)
    priority_curves = raw_curves[:, :n_priority_curves]
    lag_cols, lag_normalize_mask = periodic_lag_channels(
        priority_curves, PERIODIC_LAGS, availability_lags=(24, 48, 96))
    # include_count=False : 3e vague de coupe (250 features, cf. discussion)
    # -- le "nombre de copies disponibles" est un signal dérivé du masque de
    # trous, déjà perçu nativement par HELIX (DMSA), même raisonnement que
    # les flags de disponibilité des lags et l'ex-volatilité.
    periodic_summary, periodic_summary_mask = periodic_summary_channels(
        priority_curves, PERIODIC_LAGS, include_count=False)
    # seasonal_cols et neighbor_summary SUPPRIMÉS (3e vague) :
    # - seasonal_cols (médiane par créneau horaire) est exactement le type
    #   de feature que benchmark.md §15bis a testé et trouvé À GAIN NUL pour
    #   TabPFN ("semble déjà extraire l'essentiel du signal brut sans qu'un
    #   résumé statistique manuel soit nécessaire") -- HELIX a déjà le
    #   calendrier (tcov) et les copies périodiques brutes (lag_cols) pour
    #   retrouver ce signal lui-même ;
    # - neighbor_summary recalculait à la main un consensus voisins
    #   (type TabPFN) alors que les 150 courbes donneuses BRUTES sont déjà
    #   dans raw_curves -- c'est exactement la contradiction de principe
    #   qu'on voulait éviter dès le départ : HELIX est censé apprendre la
    #   structure inter-courbes lui-même via l'attention, pas la recevoir
    #   pré-résumée.

    # Cible (val_cols) non normalisée si NORMALIZE_TARGET_CURVES=False,
    # calibration + donneurs normalisés comme avant -- même ordre de colonnes
    # que raw_curves (cols = val_cols + calibration_cols + feat_cols).
    raw_curves_normalize_mask = np.concatenate([
        np.full(n_target, NORMALIZE_TARGET_CURVES, dtype=bool),
        np.ones(len(calibration_cols) + len(feat_cols), dtype=bool),
    ])

    if opts["aux_channels"]:
        # keep_idx=cibles+calibration seulement (3e vague) : l'accord
        # périodique des DONNEURS n'alimente rien en aval (contrairement à
        # celui de la cible, déjà réinjecté par alpha_blend_post) -- moins
        # prioritaire à exposer explicitement, HELIX peut le redécouvrir
        # lui-même sur les courbes brutes des donneurs.
        agree_cols = per_curve_aux_channels(raw_curves, keep_idx=range(n_priority_curves))
        mat = np.hstack([raw_curves, lag_cols, periodic_summary, tcov, agree_cols])
        normalize_mask = np.concatenate([
            raw_curves_normalize_mask, lag_normalize_mask,
            periodic_summary_mask,
            np.zeros(tcov.shape[1], dtype=bool),
            np.ones(agree_cols.shape[1], dtype=bool),
        ])
    else:
        mat = np.hstack([raw_curves, lag_cols, periodic_summary, tcov])
        normalize_mask = np.concatenate([
            raw_curves_normalize_mask, lag_normalize_mask,
            periodic_summary_mask,
            np.zeros(tcov.shape[1], dtype=bool),
        ])

    scaled, center, scale = prepare_helix_matrix(
        mat, NORMALIZE_HELIX_INPUTS, normalize_mask=normalize_mask)
    T, F = scaled.shape
    if len(normalize_mask) != F:
        raise RuntimeError(f"Masque de normalisation incohérent: {len(normalize_mask)} != F={F}")
    window_len = opts["window_len"]
    windows, starts = build_windows(scaled, window_len, opts["stride"])
    train_set = {"X": windows}  # ajustement sur toute la chronologie du lot
    val_set, n_calibration_truth = make_official_calibration_set(
        scaled, center, scale, y_train, cols, calibration_cols, window_len)
    print(f"  [helix] features: brut={raw_curves.shape[1]}, lags={lag_cols.shape[1]}, "
          f"résumés périodiques (sans compte)={periodic_summary.shape[1]}, "
          f"saisonnalité=0 (supprimée), voisins agrégés=0 (supprimé), calendrier={tcov.shape[1]}, "
          f"auxiliaires (accord périodique cibles+calibration seulement)="
          f"{agree_cols.shape[1] if opts['aux_channels'] else 0}",
          flush=True)
    print(f"  [helix] repeat {repeat_idx + 1}, lot {batch_idx + 1}: F={F}, "
          f"train_windows={len(windows)}, val_windows={len(val_set['X'])}, "
          f"trous_calibration={n_calibration_truth}, window_len={window_len}, "
          f"Adam PyPOTS par défaut (lr=1e-3, sans weight decay), "
          f"ORT/MIT={opts['ort_weight']:g}/{opts['mit_weight']:g}",
          flush=True)

    class CalibrationWattMAE(Criterion):
        """MAE dénormalisée, calculée uniquement sur les vrais trous de calibration."""

        def __init__(self, feature_scale):
            super().__init__()
            self.register_buffer(
                "feature_scale", torch.as_tensor(feature_scale, dtype=torch.float32)
            )

        def forward(self, logits, targets, masks=None):
            if masks is None:
                masks = torch.ones_like(logits)
            errors_watts = (
                torch.abs(logits - targets)
                * self.feature_scale.view(1, 1, -1)
                * masks
            )
            return errors_watts.sum() / masks.sum().clamp_min(1.0)

    validation_metric = CalibrationWattMAE(scale)

    # helix initialisé à None avant le try : garantit que le `finally` peut
    # toujours détacher modèle, optimiseur et meilleur state_dict CUDA, que
    # la construction/le fit réussisse ou non. torch.cuda.empty_cache() SEUL
    # ne suffit pas tant que ces objets sont encore référencés.
    helix = None
    imputed_windows = None
    try:
        helix = HELIX(
            n_steps=window_len, n_features=F, d_pe=opts["d_pe"], d_feature_embed=opts["d_feature_embed"],
            d_model=opts["d_model"], n_heads=opts["n_heads"], n_layers=opts["n_layers"], dropout=opts["dropout"],
            ORT_weight=opts["ort_weight"], MIT_weight=opts["mit_weight"],
            epochs=opts["epochs"], patience=opts["patience"],
            batch_size=min(opts["batch_size"], len(windows)),
            device=opts["device"],
            # ORT/MIT sont explicitement optimisées en MAE normalisée. La
            # validation est une MAE dénormalisée en watts sur les vrais
            # trous des courbes de calibration.
            training_loss=MAE,
            validation_metric=validation_metric,
            verbose=True,
        )

        capture = _LossCapture()
        pypots_logger.addHandler(capture)
        try:
            helix.fit(train_set, val_set)
        finally:
            pypots_logger.removeHandler(capture)

        out_path = None
        if opts.get("plot_training") and capture.train_losses:
            tag = f"repeat{repeat_idx + 1}_batch{batch_idx + 1}"
            out_path = os.path.join(PLOTS_DIR, f"helix_{tag}.png")
            plot_training_curve(capture.train_losses, capture.val_losses, out_path,
                                 title=f"HELIX -- {tag} (validation officielle calibration en W)")

        best_epoch, best_val = capture.best_validation()
        last_epoch = capture.epochs[-1] if capture.epochs else 0
        last_val = next((v for v in reversed(capture.val_losses) if v is not None), None)
        if capture.interrupted:
            stop_reason = "INTERRUPTION UTILISATEUR/COLAB"
        elif capture.patience_exceeded:
            stop_reason = f"patience={opts['patience']} sans amélioration"
        elif last_epoch >= opts["epochs"]:
            stop_reason = f"limite de {opts['epochs']} époques atteinte"
        else:
            stop_reason = "fin signalée par PyPOTS"
        print(f"  [helix] repeat {repeat_idx + 1}, lot {batch_idx + 1}: "
              f"arrêt={stop_reason}; dernière époque={last_epoch}"
              + (f", dernière MAE calibration={last_val:.2f} W" if last_val is not None else "")
              + (f"; BEST époque={best_epoch}, MAE calibration={best_val:.2f} W"
                 if best_val is not None else "")
              + (f" -> {out_path}" if out_path is not None else ""), flush=True)

        if capture.interrupted:
            print("  [helix] lot interrompu : aucune prédiction ni sauvegarde. "
                  "Relancez la cellule pour recommencer ce lot.", flush=True)
            raise KeyboardInterrupt("Entraînement HELIX interrompu ; lot non sauvegardé")

        imputed_windows = helix.impute({"X": windows})
    except RuntimeError as exc:
        if _is_cuda_oom(exc):
            print(f"  [helix] repeat {repeat_idx + 1}, lot {batch_idx + 1}: "
                  f"CUDA OOM avec F={F}, window_len={window_len}, "
                  f"batch_size={opts['batch_size']}. window_len est le facteur QUADRATIQUE dominant "
                  f"(cf. mise en garde en tête de fichier) -- réduis-le en premier, puis MAX_FEATURES, "
                  f"puis batch_size, et relance -- ce repeat n'est PAS marqué comme fait, "
                  f"la progression précédente est conservée.", flush=True)
        else:
            raise
    finally:
        _release_helix_cuda(helix, repeat_idx)
        helix = None

    if imputed_windows is None:
        return None, None, None
    full_scaled = reassemble(imputed_windows, starts, window_len, T, F)
    full = full_scaled * scale + center

    raw_err_by_col, blend_err_by_col, ensemble_err_by_col = {}, {}, {}
    for j, col in enumerate(val_cols):
        s = X_train[col]
        missing = s.isna().to_numpy()
        miss_idx = np.flatnonzero(missing)
        if len(miss_idx) == 0:
            continue
        pred_raw = full[miss_idx, j]
        pred_blend = alpha_blend_post(s, miss_idx, pred_raw)
        y_true = y_train[col].to_numpy()[miss_idx]
        valid = ~np.isnan(y_true)
        raw_err_by_col[col] = np.abs(pred_raw[valid] - y_true[valid])
        blend_err_by_col[col] = np.abs(pred_blend[valid] - y_true[valid])

        # Ensemble HELIX+blend x TabPFN (idée #8, cf. mise en garde SAITS en
        # tête de fichier) : combine avec la prédiction TabPFN elle-même
        # plutôt qu'avec la seule interpolation, pour que HELIX puisse
        # aider TabPFN au lieu de simplement le concurrencer. tabpfn_pred_by_col
        # a été calculé par eval_tabpfn_repeat AVANT le blend HELIX, sur les
        # mêmes positions valides (miss_idx filtré par y_true non-NaN) --
        # une taille incohérente signale un bug de correspondance de colonnes
        # (ex. lot recalculé sur un split différent) et ne doit jamais être
        # mélangée silencieusement.
        tabpfn_pred_valid = np.asarray(tabpfn_pred_by_col.get(col, []), dtype=float)
        if tabpfn_pred_valid.shape[0] == int(valid.sum()):
            pred_ensemble = (HELIX_ENSEMBLE_WEIGHT * pred_blend[valid]
                              + (1.0 - HELIX_ENSEMBLE_WEIGHT) * tabpfn_pred_valid)
            ensemble_err_by_col[col] = np.abs(pred_ensemble - y_true[valid])
        else:
            print(f"  [helix] ensemble ignoré pour {col}: tailles TabPFN/HELIX "
                  f"incohérentes ({tabpfn_pred_valid.shape[0]} vs {int(valid.sum())})",
                  flush=True)
    return raw_err_by_col, blend_err_by_col, ensemble_err_by_col


# =============================================================================
# 4. Chargement des données
# =============================================================================
print("Chargement de X_train, y_train...", flush=True)
X_train = pd.read_csv(os.path.join(DATA_DIR, X_TRAIN_FILE), index_col=0)
y_train = pd.read_csv(os.path.join(DATA_DIR, Y_TRAIN_FILE), index_col=0)
donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]
holed_cols = list(y_train.columns)
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, "
      f"donneurs: {len(donor_cols)}, holed: {len(holed_cols)}", flush=True)
print(f"MAX_FEATURES={MAX_FEATURES} ({'TOUTES les courbes -- RISQUE OOM ELEVE' if MAX_FEATURES is None else f'{MAX_FEATURES} donneurs les plus corrélés par repeat'})",
      flush=True)
print(f"NORMALIZE_HELIX_INPUTS={NORMALIZE_HELIX_INPUTS} "
      f"({'robust scaling actif' if NORMALIZE_HELIX_INPUTS else 'toutes les entrées HELIX restent brutes'})",
      flush=True)
print(f"NORMALIZE_TARGET_CURVES={NORMALIZE_TARGET_CURVES} "
      f"({'courbes cibles normalisées comme le reste' if NORMALIZE_TARGET_CURVES else 'courbes cibles en W bruts, calibration/donneurs normalisés'})",
      flush=True)
print(f"HELIX PyPOTS 1.5 + features TabPFN agrégées: lags={PERIODIC_LAGS}, "
      f"architecture d_model={HELIX_OPTS['d_model']}, heads={HELIX_OPTS['n_heads']}, "
      f"layers={HELIX_OPTS['n_layers']}, ORT/MIT="
      f"{HELIX_OPTS['ort_weight']:g}/{HELIX_OPTS['mit_weight']:g}, "
      f"Adam, train MAE normalisée, validation MAE en W, "
      f"batch Colab={HELIX_OPTS['batch_size']}", flush=True)

# =============================================================================
# 5. Boucle officielle 10 repeats x 10 courbes (protocole apparié standard du
#    dépôt, comparable à tabpfn_v14_confirm.py : 66,24 +/- 32,45 W)
# =============================================================================
progress_signature = {
    "schema_version": 9,  # 9: train MAE explicite + validation MAE dénormalisée en watts
    "protocol": "paired_10x10_pypots15_mae_watts_batch2",
    "pypots_version": "1.5",
    "random_state": RANDOM_STATE,
    "n_repeats": N_REPEATS,
    "val_batch_size": VAL_BATCH_SIZE,
    "n_calibration_cols": N_CALIBRATION_COLS,
    "normalize_helix_inputs": NORMALIZE_HELIX_INPUTS,
    "normalize_target_curves": NORMALIZE_TARGET_CURVES,
    "periodic_lags": PERIODIC_LAGS,
    "helix_ensemble_weight": HELIX_ENSEMBLE_WEIGHT,
    "max_features": MAX_FEATURES,
    "helix_opts": HELIX_OPTS,
}
if os.path.exists(PROGRESS_PATH):
    with open(PROGRESS_PATH) as handle:
        progress = json.load(handle)
    if progress.get("signature") != progress_signature:
        raise RuntimeError("Le checkpoint existe avec une configuration différente")
else:
    progress = {"signature": progress_signature, "repeats": {}}

for i in range(N_REPEATS):
    seed_i = RANDOM_STATE + i
    # Étape 1, IDENTIQUE au protocole TabPFN confirmé (tabpfn_v14_ablation.run_config) :
    # split 90 % dev / 10 % val (1000 courbes -> 100 val_cols_full).
    dev_cols, val_cols_full = train_test_split(holed_cols, test_size=.1, random_state=seed_i)
    dev_cols, val_cols_full = list(dev_cols), list(val_cols_full)
    if len(val_cols_full) != 100:
        raise RuntimeError(f"Le repeat {i + 1} contient {len(val_cols_full)} courbes de split à 10 %, attendu 100")
    # Étape 2, IDENTIQUE à run_config(n_val_sub=10) : ré-échantillonne 10
    # courbes parmi ces 100 -- même RandomState(seed_i) que TabPFN, donc les
    # 10 courbes évaluées ici sont EXACTEMENT celles de tabpfn_v14_confirm.py
    # au même repeat -- comparaison appariée légitime (pas seulement des
    # graines identiques : les mêmes trous, sur les mêmes courbes).
    rng = np.random.RandomState(seed_i)
    val_cols = list(rng.choice(val_cols_full, size=min(VAL_BATCH_SIZE, len(val_cols_full)), replace=False))
    if len(val_cols) != VAL_BATCH_SIZE:
        raise RuntimeError(f"Le repeat {i + 1} contient {len(val_cols)} courbes de validation, attendu {VAL_BATCH_SIZE}")
    # make_homogeneous_batches avec batch_size == len(val_cols) ne produit
    # qu'UN SEUL lot (toutes les 10 courbes) -- ne sert plus qu'à choisir les
    # N_CALIBRATION_COLS courbes de dev les plus proches en profil, comme avant.
    val_batches, calibration_batches = make_homogeneous_batches(
        X_train, val_cols, dev_cols, VAL_BATCH_SIZE, N_CALIBRATION_COLS,
        seed=seed_i + 10_000)
    ordered_val_cols = [col for batch in val_batches for col in batch]
    if set(ordered_val_cols) != set(val_cols) or len(ordered_val_cols) != len(val_cols):
        raise RuntimeError(f"Lots homogènes invalides au repeat {i + 1}")

    rec = progress["repeats"].setdefault(str(i), {
        "val_cols": ordered_val_cols,
        "val_batches": val_batches,
        "calibration_batches": calibration_batches,
        "tabpfn": {}, "tabpfn_pred": {},
        "helix_raw": {}, "helix_blend": {}, "helix_ensemble": {},
        "completed_batches": [], "tabpfn_complete": False,
    })
    if (rec["val_cols"] != ordered_val_cols or rec.get("val_batches") != val_batches
            or rec["calibration_batches"] != calibration_batches):
        raise RuntimeError(f"Split sauvegardé incohérent pour le repeat {i + 1}")

    tabpfn_reused = rec.get("tabpfn_complete", False)
    if not tabpfn_reused:
        t0 = time.time()
        tabpfn_err, tabpfn_pred = eval_tabpfn_repeat(X_train, y_train, donor_cols, dev_cols, val_cols, V14_BEST)
        rec["tabpfn"] = {c: e.tolist() for c, e in tabpfn_err.items()}
        rec["tabpfn_pred"] = {c: p.tolist() for c, p in tabpfn_pred.items()}
        rec["tabpfn_complete"] = True
        save_json_atomic(progress, PROGRESS_PATH)
        print(f"  [repeat {i + 1}] tabpfn v14, {VAL_BATCH_SIZE} courbes : {time.time() - t0:.1f}s", flush=True)

    tabpfn_mae = compute_tabpfn_validation_score(rec)
    if rec.get("tabpfn_mae") != tabpfn_mae:
        rec["tabpfn_mae"] = tabpfn_mae
        save_json_atomic(progress, PROGRESS_PATH)
    source = "checkpoint repris" if tabpfn_reused else "calculé"
    print(f"\n=== REPEAT {i + 1}/{N_REPEATS} -- VALIDATION TABPFN COMPLETE ===\n"
          f"MAE TabPFN v14 ({source}, {VAL_BATCH_SIZE} courbes) : {tabpfn_mae:.2f} W\n",
          flush=True)

    completed = set(rec["completed_batches"])
    repeat_oom = False
    for batch_idx, (batch_cols, calibration_cols) in enumerate(zip(val_batches, calibration_batches)):
        if batch_idx in completed:
            print(f"repeat {i + 1}/{N_REPEATS}, lot {batch_idx + 1}/{len(val_batches)} déjà fait, skip",
                  flush=True)
            continue
        t0 = time.time()
        helix_raw_err, helix_blend_err, helix_ensemble_err = eval_helix_repeat(
            X_train, y_train, batch_cols, calibration_cols, donor_cols, HELIX_OPTS,
            rec["tabpfn_pred"], repeat_idx=i, batch_idx=batch_idx)
        print(f"  [repeat {i + 1}, lot {batch_idx + 1}] helix : {time.time() - t0:.1f}s", flush=True)
        if helix_raw_err is None:
            print(f"repeat {i + 1}, lot {batch_idx + 1} NON sauvegardé (OOM). "
                  "La reprise repartira de ce lot.\n", flush=True)
            repeat_oom = True
            break

        rec["helix_raw"].update({c: e.tolist() for c, e in helix_raw_err.items()})
        rec["helix_blend"].update({c: e.tolist() for c, e in helix_blend_err.items()})
        rec["helix_ensemble"].update({c: e.tolist() for c, e in helix_ensemble_err.items()})
        rec["completed_batches"].append(batch_idx)
        save_json_atomic(progress, PROGRESS_PATH)
        batch_raw = float(np.concatenate(list(helix_raw_err.values())).mean())
        batch_blend = float(np.concatenate(list(helix_blend_err.values())).mean())
        batch_ensemble = (float(np.concatenate(list(helix_ensemble_err.values())).mean())
                           if helix_ensemble_err else float("nan"))
        batch_tabpfn = compute_tabpfn_subset_score(rec, batch_cols)
        print(f"\n=== REPEAT {i + 1}, LOT {batch_idx + 1}/{len(val_batches)} "
              f"-- MÊMES {len(batch_cols)} CIBLES / MÊMES TROUS ===\n"
              f"MAE TabPFN v14 (lot) :       {batch_tabpfn:.2f} W\n"
              f"MAE HELIX brute officielle : {batch_raw:.2f} W "
              f"(delta {batch_raw - batch_tabpfn:+.2f} W)\n"
              f"MAE HELIX blend officielle : {batch_blend:.2f} W "
              f"(delta {batch_blend - batch_tabpfn:+.2f} W)\n"
              f"MAE HELIX+TabPFN ensemble ({HELIX_ENSEMBLE_WEIGHT:g}/{1 - HELIX_ENSEMBLE_WEIGHT:g}) : "
              f"{batch_ensemble:.2f} W (delta {batch_ensemble - batch_tabpfn:+.2f} W)\n"
              f"MAE TabPFN v14 (repeat, {VAL_BATCH_SIZE} courbes) : {tabpfn_mae:.2f} W\n"
              "Note: la validation interne HELIX affichée pendant le fit concerne "
              "les courbes de calibration, pas ces cibles officielles.\n",
              flush=True)

    if repeat_oom:
        continue
    if len(set(rec["completed_batches"])) != len(val_batches):
        raise RuntimeError(f"Lots manquants après le repeat {i + 1}")
    repeat_scores = compute_repeat_scores(rec)
    repeat_wins = repeat_scores["wins"]
    repeat_wins_ens = repeat_scores["wins_ensemble"]
    print(f"\n=== REPEAT {i + 1}/{N_REPEATS} COMPLET -- {VAL_BATCH_SIZE} COURBES ===\n"
          f"TabPFN v14 :  {repeat_scores['tabpfn']:.2f} W\n"
          f"HELIX brut :  {repeat_scores['helix_raw']:.2f} W "
          f"(delta vs TabPFN {repeat_scores['helix_raw'] - repeat_scores['tabpfn']:+.2f} W)\n"
          f"HELIX+blend : {repeat_scores['helix_blend']:.2f} W "
          f"(delta vs TabPFN {repeat_scores['helix_blend'] - repeat_scores['tabpfn']:+.2f} W)\n"
          f"HELIX+TabPFN ensemble : {repeat_scores['helix_ensemble']:.2f} W "
          f"(delta vs TabPFN {repeat_scores['helix_ensemble'] - repeat_scores['tabpfn']:+.2f} W)\n"
          f"Victoires colonnes -- HELIX+blend: {repeat_wins['helix_blend']}, "
          f"TabPFN: {repeat_wins['tabpfn']}, égalités: {repeat_wins['tie']}\n"
          f"Victoires colonnes -- HELIX+TabPFN ensemble: {repeat_wins_ens['helix_ensemble']}, "
          f"TabPFN: {repeat_wins_ens['tabpfn']}, égalités: {repeat_wins_ens['tie']}\n",
          flush=True)

# =============================================================================
# 6. Rapport apparié (repeat par repeat ET colonne par colonne)
# =============================================================================
rows = []
all_tabpfn, all_helix_raw, all_helix_blend, all_helix_ensemble = [], [], [], []
wins = {"helix_blend": 0, "tabpfn": 0, "tie": 0}
wins_ensemble = {"helix_ensemble": 0, "tabpfn": 0, "tie": 0}
for i in range(N_REPEATS):
    if str(i) not in progress["repeats"]:
        continue
    rec = progress["repeats"][str(i)]
    expected_batches = (len(rec["val_cols"]) + VAL_BATCH_SIZE - 1) // VAL_BATCH_SIZE
    if len(set(rec.get("completed_batches", []))) != expected_batches:
        continue
    tb = {c: np.array(v) for c, v in rec["tabpfn"].items()}
    hr = {c: np.array(v) for c, v in rec["helix_raw"].items()}
    hb = {c: np.array(v) for c, v in rec["helix_blend"].items()}
    he = {c: np.array(v) for c, v in rec.get("helix_ensemble", {}).items()}
    common = [c for c in rec["val_cols"] if c in tb and c in hr and c in hb and c in he]
    if len(common) != len(rec["val_cols"]):
        print(f"repeat {i + 1} exclu du rapport : {len(common)}/{VAL_BATCH_SIZE} courbes appariées", flush=True)
        continue
    tb_cat = np.concatenate([tb[c] for c in common])
    hr_cat = np.concatenate([hr[c] for c in common])
    hb_cat = np.concatenate([hb[c] for c in common])
    he_cat = np.concatenate([he[c] for c in common])
    all_tabpfn.append(tb_cat)
    all_helix_raw.append(hr_cat)
    all_helix_blend.append(hb_cat)
    all_helix_ensemble.append(he_cat)
    for c in common:
        m_t, m_b, m_e = tb[c].mean(), hb[c].mean(), he[c].mean()
        if m_b < m_t - 1e-9:
            wins["helix_blend"] += 1
        elif m_t < m_b - 1e-9:
            wins["tabpfn"] += 1
        else:
            wins["tie"] += 1
        if m_e < m_t - 1e-9:
            wins_ensemble["helix_ensemble"] += 1
        elif m_t < m_e - 1e-9:
            wins_ensemble["tabpfn"] += 1
        else:
            wins_ensemble["tie"] += 1
    rows.append((i + 1, float(tb_cat.mean()), float(hr_cat.mean()), float(hb_cat.mean()), float(he_cat.mean())))

print("\n| Repeat | TabPFN v14 | HELIX brut | HELIX+blend | Δ (blend-tabpfn) | HELIX+TabPFN ens. | Δ (ens.-tabpfn) |")
print("|---:|---:|---:|---:|---:|---:|---:|")
for r, t, hr, hb, he in rows:
    print(f"| {r} | {t:.2f} | {hr:.2f} | {hb:.2f} | {hb - t:+.2f} | {he:.2f} | {he - t:+.2f} |")

if all_tabpfn:
    mae_tabpfn = float(np.concatenate(all_tabpfn).mean())
    mae_helix_raw = float(np.concatenate(all_helix_raw).mean())
    mae_helix_blend = float(np.concatenate(all_helix_blend).mean())
    mae_helix_ensemble = float(np.concatenate(all_helix_ensemble).mean())
    print(f"\nMAE TabPFN v14 (pool):             {mae_tabpfn:.2f} W")
    print(f"MAE HELIX brut (pool):             {mae_helix_raw:.2f} W")
    print(f"MAE HELIX + blend (pool):          {mae_helix_blend:.2f} W")
    print(f"MAE HELIX + TabPFN ensemble (pool): {mae_helix_ensemble:.2f} W")
    print(f"gain relatif HELIX+blend vs TabPFN v14: {100 * (1 - mae_helix_blend / mae_tabpfn):.1f}%")
    print(f"gain relatif HELIX+TabPFN ensemble vs TabPFN v14 seul: "
          f"{100 * (1 - mae_helix_ensemble / mae_tabpfn):.1f}%")
    print(f"colonnes gagnées par HELIX+blend: {wins['helix_blend']} / par TabPFN v14: {wins['tabpfn']} "
          f"/ égalité: {wins['tie']} / total: {sum(wins.values())}")
    print(f"colonnes gagnées par HELIX+TabPFN ensemble: {wins_ensemble['helix_ensemble']} / "
          f"par TabPFN v14: {wins_ensemble['tabpfn']} / égalité: {wins_ensemble['tie']} / "
          f"total: {sum(wins_ensemble.values())}")

    out = {
        "protocol": progress_signature,
        "evaluated_columns_per_repeat": VAL_BATCH_SIZE,
        "max_features": MAX_FEATURES, "helix_opts": HELIX_OPTS,
        "helix_ensemble_weight": HELIX_ENSEMBLE_WEIGHT,
        "rows": [{"repeat": r, "tabpfn": t, "helix_raw": hr, "helix_blend": hb, "delta": hb - t,
                  "helix_ensemble": he, "delta_ensemble": he - t}
                 for r, t, hr, hb, he in rows],
        "mae_tabpfn": mae_tabpfn, "mae_helix_raw": mae_helix_raw, "mae_helix_blend": mae_helix_blend,
        "mae_helix_ensemble": mae_helix_ensemble,
        "wins_helix_blend": wins["helix_blend"], "wins_tabpfn": wins["tabpfn"], "ties": wins["tie"],
        "wins_helix_ensemble": wins_ensemble["helix_ensemble"],
        "wins_tabpfn_ensemble": wins_ensemble["tabpfn"], "ties_ensemble": wins_ensemble["tie"],
    }
    save_json_atomic(out, RESULTS_PATH, indent=2)
    print(f"\nrésultats écrits dans {RESULTS_PATH}", flush=True)
else:
    print("\nAucun repeat complété (tout a OOM ?) -- réduis MAX_FEATURES et relance.", flush=True)

try:
    from google.colab import files
    files.download(RESULTS_PATH)
except ImportError:
    pass  # exécuté hors Colab
except Exception as exc:
    print(f"Téléchargement auto impossible ({type(exc).__name__}: {exc}). "
          f"Fichiers sur disque : {RESULTS_PATH}, {PROGRESS_PATH}, {PLOTS_DIR}/", flush=True)
