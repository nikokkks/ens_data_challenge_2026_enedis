"""Script autonome pour Google Colab : génère la soumission TabPFN V14 en
fittant sur TOUTES les données annotées disponibles :

    pool de donneurs = colonnes non-"holed_*" de X_train (20 000, complètes)
                      + y_train (1000, vérité terrain des holed_* de X_train)
                      + colonnes non-"holed_*" de X_test (37 140, complètes)

Autrement dit : toutes les colonnes de X_train ET de X_test qui n'ont AUCUN
NaN (donneuses), plus toutes les valeurs de y_train, sont concaténées en un
seul pool de voisins. On prédit ensuite sur le reste des données à
prédire : les positions manquantes (NaN) des 1000 colonnes "holed_*" de
X_test -- ce sont les valeurs à soumettre au challenge.

C'est le pipeline validé du repo (TabPFN V14, MAE 63.24 +/- 30.46 W sur le
protocole 10x10, cf. generate_submission_tabpfn_v14.py / tabpfn_v14_ablation.py),
recopié ici en version autonome (aucune dépendance au reste du repo) pour
tourner directement dans une session Colab : tout le pipeline (features,
sélection de voisins, blend alpha, TabPFN) est inclus dans ce seul fichier --
inutile d'uploader feature_engineering_bench.py / evaluate.py /
tabpfn_v14_ablation.py.

Ajout vs la version précédente : features SVD/PCA (top-N_SVD facteurs
temporels du pool, cf. claude/svd_matrix_completion_diagnostics.py et
claude/tabpfn_v14_svd_paired_eval.py) concaténées aux features existantes.
Validé en test apparié TabPFN (même repeat, mêmes 100 colonnes val) : MAE
58.42 W sans SVD -> 57.49 W avec SVD (+1.60%, 61 colonnes gagnées / 33
perdues / 6 égalité) -- gain modeste mais réel sur un seul repeat.

Modèle local (GPU Colab), pas l'API cloud
------------------------------------------
Ce script utilise le package `tabpfn` (modèle exécuté localement, sur le GPU
de la session Colab) et non `tabpfn_client` (API cloud) -- donc pas de clé
API/token, pas de quota journalier. En contrepartie, un seul GPU ne peut pas
fitter 20 TabPFN en parallèle comme le faisaient les appels réseau cloud
(risque d'OOM) : la concurrence est donc volontairement réduite (cf.
MAX_WORKERS ci-dessous). Le premier fit télécharge les poids pré-entraînés
du modèle (une fois, mis en cache) -- nécessite un accès internet initial.

Utilisation dans Colab
-----------------------
1. Active un runtime GPU (Exécution > Modifier le type d'exécution > GPU).
2. Colle ce script dans une cellule (ou exécute-le tel quel avec `%run`).
3. Mets les 3 CSV du challenge dans DATA_DIR (par défaut "/content/data") :
   - upload direct (icône dossier à gauche, ou `google.colab.files.upload()`)
   - ou monte ton Drive (`from google.colab import drive; drive.mount(...)`)
     et pointe DATA_DIR dessus si tu veux que le CHECKPOINT survive à une
     déconnexion (le script reprend automatiquement où il s'est arrêté si tu
     le relances avec le même CHECKPOINT_PATH).
4. Lance la cellule. Le fichier final est écrit dans OUT_PATH et proposé
   au téléchargement à la fin.
"""

# =============================================================================
# 0. Installation (Colab n'a pas tabpfn par défaut)
# =============================================================================
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn==8.4.0"], check=True)

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from tabpfn import TabPFNRegressor

warnings.filterwarnings("ignore")

print(f"GPU disponible : {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " -- CPU seul, ce sera lent"),
      flush=True)

# =============================================================================
# 1. Configuration -- à adapter
# =============================================================================
DATA_DIR = "/content/data"          # dossier contenant les 3 CSV du challenge
OUT_DIR = "/content"                # dossier de sortie (idéalement sur Drive monté)
X_TRAIN_FILE = "X_train_78VdSWL.csv"
Y_TRAIN_FILE = "y_train_u0UkKEh.csv"
X_TEST_FILE = "X_test_XKVc4no.csv"
SAMPLE_FILE = "new_output_sample.csv"  # optionnel, juste pour valider le format

OUT_PATH = os.path.join(OUT_DIR, "submission_tabpfn_v14_svd_full.csv")
CHECKPOINT_PATH = os.path.join(OUT_DIR, "submission_tabpfn_v14_svd_full_partial.csv")

N_CLUSTERS, RANDOM_STATE = 15, 42
N_SVD = 50
DEVICE = "auto"  # "auto" choisit le GPU si dispo (torch.cuda.is_available()), sinon CPU
# Un seul GPU Colab (T4/L4/A100 partagé) ne supporte pas 20 fits TabPFN
# simultanés comme le faisait l'API cloud (risque d'OOM) : concurrence
# réduite ici. Augmente MAX_WORKERS prudemment si ton GPU a beaucoup de VRAM
# libre (le fit_shape_clusters et le chargement des données restent, eux,
# aussi rapides qu'avant).
BATCH_SIZE, MAX_WORKERS = 10, 2
MAX_RETRIES = 3

# Config V14 validée (README.md du repo) : poids alpha réduit quand la
# courbe est bruitée (volatility_alpha) ou que les copies périodiques
# s'accordent (periodic_alpha_strength), + copie périodique à ±24h
# (extra_lags) + tendance locale des voisins (neighbor_rolling).
OPTS = dict(
    k=40, k_min=5, corr_threshold=0.15, euclid_weight=0.3, ratio_clip=(0.1, 10.0),
    extra_lags=(24,), vol_weight=0.8, vol_cap=0.6,
    periodic_alpha_strength=1.0, alpha_scale=1.0,
)


# =============================================================================
# 2. Pipeline V14 (copié depuis feature_engineering_bench.py /
#    tabpfn_v14_ablation.py, réduit aux branches réellement actives pour la
#    config OPTS ci-dessus -- comportement identique au pipeline validé)
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


def fit_shape_clusters(donor_df, n_clusters=15, random_state=42):
    filled = donor_df.ffill().bfill()
    Z = (filled - filled.mean()) / (filled.std() + 1e-9)
    km = KMeans(n_clusters=n_clusters, n_init=4, max_iter=100, random_state=random_state)
    labels = km.fit_predict(Z.T.values)
    labels = pd.Series(labels, index=donor_df.columns)
    return km, labels


def compute_svd_factors(pool_df, n_components=20, random_state=42):
    filled = pool_df.ffill().bfill().to_numpy(dtype=np.float64)
    c = filled - filled.mean(axis=0, keepdims=True)
    U, s, Vt = randomized_svd(c, n_components=n_components, random_state=random_state)
    return U * s


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
        self.reg = TabPFNRegressor(random_state=random_state, device=DEVICE)

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X, output_type="median")


def build_v14_features(s, pool_vals, km, cluster_col_idx, tfeat, index, o, svd_feat=None):
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
    extra = np.hstack([extra, neighbor_rolling_features(scaled)])
    if svd_feat is not None:
        extra = np.hstack([extra, svd_feat])

    return obs_idx, miss_idx, target_obs, interp_full, knn_pred, scaled, extra, missing, lags


def predict_column_v14(col, s, pool_vals, km, cluster_col_idx, tfeat, index, opts, svd_feat=None):
    o = opts
    error = None
    built = build_v14_features(s, pool_vals, km, cluster_col_idx, tfeat, index, o, svd_feat)
    if built is None:
        return col, s.copy(), None
    obs_idx, miss_idx, target_obs, interp_full, knn_pred, scaled, extra, missing, lags = built

    pred_neighbors = None
    if len(obs_idx) >= 15:
        for attempt in range(MAX_RETRIES):
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
    alpha *= volatility_alpha_mult(vol, vol_weight=o["vol_weight"], cap=o["vol_cap"])
    alpha = np.clip(alpha * o["alpha_scale"], 0.0, 0.95)
    pred = alpha * interp_full.to_numpy()[miss_idx] + (1 - alpha) * pred_neighbors

    filled = s.copy()
    filled.iloc[miss_idx] = pred
    return col, filled, error


# =============================================================================
# 3. Chargement des données (modèle local : pas d'authentification à faire)
# =============================================================================
print("Chargement de X_train, y_train, X_test...", flush=True)
X_train = pd.read_csv(os.path.join(DATA_DIR, X_TRAIN_FILE), index_col=0)
y_train = pd.read_csv(os.path.join(DATA_DIR, Y_TRAIN_FILE), index_col=0)
X_test = pd.read_csv(os.path.join(DATA_DIR, X_TEST_FILE), index_col=0)

# Toutes les colonnes SANS AUCUN NaN (X_train + X_test) + toute la vérité
# terrain de y_train -> pool de donneurs maximal, sans jamais utiliser les
# positions manquantes des colonnes holed_* elles-mêmes.
train_donors = [c for c in X_train.columns if not str(c).startswith("holed_")]
test_donors = [c for c in X_test.columns if not str(c).startswith("holed_")]
holes = [c for c in X_test.columns if str(c).startswith("holed_")]  # à prédire/soumettre

pool = pd.concat([X_test[test_donors], X_train[train_donors], y_train], axis=1)
print(f"Pool de donneurs : {pool.shape[1]} colonnes "
      f"({len(test_donors)} X_test + {len(train_donors)} X_train + {y_train.shape[1]} y_train), "
      f"{len(holes)} colonnes holed_* à compléter", flush=True)

# =============================================================================
# 4. Reprise depuis un checkpoint éventuel
# =============================================================================
if os.path.exists(CHECKPOINT_PATH):
    done = pd.read_csv(CHECKPOINT_PATH, index_col=0)
    done.index = X_test.index
else:
    done = pd.DataFrame(index=X_test.index)
remaining = [c for c in holes if c not in done]
print(f"Reprise : {len(done.columns)} colonnes déjà faites, {len(remaining)} restantes", flush=True)

pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
print("Clustering de forme sur le pool complet...", flush=True)
km, labels = fit_shape_clusters(pool, N_CLUSTERS, RANDOM_STATE)
positions = pd.Series(np.arange(len(labels)), index=labels.index)
cluster_idx = {cl: positions[idx].to_numpy()
               for cl, idx in labels.groupby(labels).groups.items()}
tfeat = time_features_full_seasonality(X_test.index, groups=("hour", "dow", "dom", "year"))
print("Calcul des facteurs SVD/PCA du pool...", flush=True)
svd_feat = compute_svd_factors(pool, N_SVD, RANDOM_STATE)

# =============================================================================
# 5. Boucle principale : complétion par lots, en parallèle, checkpointée
# =============================================================================
fallbacks = 0
started = time.time()
for start in range(0, len(remaining), BATCH_SIZE):
    batch = remaining[start:start + BATCH_SIZE]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(predict_column_v14, c, X_test[c], pool_vals, km,
                                    cluster_idx, tfeat, X_test.index, OPTS, svd_feat)
                   for c in batch]
        for future in as_completed(futures):
            col, filled, error = future.result()
            done[col] = filled
            if error:
                fallbacks += 1
                print(f"Repli KNN pour {col}: {error}", flush=True)
    done.to_csv(CHECKPOINT_PATH)
    completed = start + len(batch)
    elapsed = time.time() - started
    eta = elapsed / completed * (len(remaining) - completed) / 60 if completed else 0.0
    rate = completed / elapsed * 60 if elapsed else 0.0
    print(f"[{len(done.columns)}/1000] lot={len(batch)} parallèle={MAX_WORKERS} "
          f"temps={elapsed/60:.1f} min vitesse={rate:.1f} col/min "
          f"ETA={eta:.1f} min replis={fallbacks}", flush=True)

# =============================================================================
# 6. Validation + écriture finale + téléchargement
# =============================================================================
final = done[holes].clip(lower=0)
final.index.name = "Horodate"

sample_path = os.path.join(DATA_DIR, SAMPLE_FILE)
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path, index_col=0)
    assert final.shape == sample.shape == (1057, 1000)
    assert list(final.columns) == list(sample.columns)
else:
    print(f"({SAMPLE_FILE} absent de {DATA_DIR}, validation de forme sautée)", flush=True)
assert final.isna().sum().sum() == 0
assert np.isfinite(final.to_numpy()).all()

final.to_csv(OUT_PATH)
print(f"Soumission écrite : {OUT_PATH}; replis KNN={fallbacks}", flush=True)

try:
    from google.colab import files
    files.download(OUT_PATH)
except ImportError:
    pass  # exécuté hors Colab : pas de téléchargement automatique
except Exception as exc:
    # files.download() a besoin d'un kernel IPython actif (comm_manager) --
    # ça échoue silencieusement (AttributeError) si ce script est lancé en
    # sous-processus externe (`!python ce_script.py`) plutôt que collé/exécuté
    # dans une cellule. Le CSV est déjà écrit sur disque dans les deux cas :
    # ce n'est qu'un raté du téléchargement automatique, pas de la génération.
    print(f"Téléchargement auto impossible ({type(exc).__name__}: {exc}). "
          f"Le fichier est bien sur disque à {OUT_PATH} -- récupère-le avec "
          f"le panneau de fichiers Colab, ou dans une cellule :\n"
          f"  from google.colab import files; files.download('{OUT_PATH}')", flush=True)
