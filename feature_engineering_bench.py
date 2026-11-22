"""
Benchmark progressif de feature engineering pour l'imputation des courbes Enedis.

Réutilise le harness de evaluate.py (repeated_holdout_eval, official_mae, baseline),
avec EXACTEMENT les mêmes paramètres (n_repeats=5, test_size=0.10, random_state=42)
pour que chaque étape soit comparable aux précédentes.

Étapes (cumulatives) :
  V1 : modèle KNN inter-courbes (déjà construit précédemment) — point de départ
  V2 : V1 + clustering de formes (KMeans) -> voisins restreints au cluster
  V3 : V2 + pondération adaptative interpolation/voisins selon la longueur du trou
  V4 : V3 + régression locale (Ridge) avec features temporelles (heure sin/cos, weekday)
  V5 : V4 + mise à l'échelle robuste des voisins (médiane/MAD au lieu de moyenne/écart-type)
  V6 : V5 + stacking (méta-modèle Ridge combinant interp / cluster-KNN / ridge-local)
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")  # Ridge ill-conditioned warnings on near-collinear neighbors, harmless

from evaluate import (
    load_data, official_mae, fill_nan_with_interpolation,
    repeated_holdout_eval, baseline_model,
)

RANDOM_STATE = 42
N_REPEATS = 5
TEST_SIZE = 0.10

# ---------------------------------------------------------------------------
# Helpers partagés
# ---------------------------------------------------------------------------

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


def gap_alpha(run_len, adaptive):
    """Poids donné à l'interpolation linéaire (1-alpha = poids voisins/régression)."""
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


def curve_volatility(values):
    """Bruit local d'une courbe : écart-type des différences premières entre
    points connus consécutifs, ramené au niveau moyen de la courbe (mesure
    auto-normalisée, comparable entre courbes de niveaux très différents)."""
    obs = values[~np.isnan(values)]
    if len(obs) < 3:
        return 0.0
    diffs = np.diff(obs)
    level = np.mean(np.abs(obs)) + 1e-9
    return float(np.std(diffs) / level)


def volatility_alpha_mult(vol, vol_weight=0.5, cap=0.6):
    """Courbe bruyante (vol élevée) -> l'interpolation linéaire lisse de vraies
    fluctuations qu'elle n'aurait pas dû lisser -> on réduit alpha (moins de
    poids à l'interpolation, plus au modèle) via un multiplicateur < 1."""
    vol_clipped = min(vol, cap) / cap
    return 1.0 - vol_weight * vol_clipped


def edge_distance_for(missing_mask):
    """Distance (en points) de chaque point manquant au bord le plus proche de
    SON trou (0 = juste à côté d'un point connu, max = au milieu d'un long trou).
    Motivé par l'EDA (figures/04) : l'interpolation linéaire est quasi exacte
    tout près des bords même d'un trou long, et se dégrade vers le centre."""
    n = len(missing_mask)
    edge_dist = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        if missing_mask[i]:
            j = i
            while j < n and missing_mask[j]:
                j += 1
            run_len = j - i
            for p in range(i, j):
                edge_dist[p] = min(p - i, j - 1 - p)
            i = j
        else:
            i += 1
    return edge_dist


def continuous_alpha(run_len, edge_dist, adaptive, edge_tau=1.5):
    """gap_alpha(run_len) de base, boosté en confiance-interpolation près des
    bords du trou (où l'interpolation est quasi exacte, même dans un trou
    pleine-journée) -> transition continue plutôt qu'un seul palier par trou."""
    base = gap_alpha(run_len, adaptive)
    if not adaptive:
        return base
    edge_boost = np.exp(-edge_dist / edge_tau)
    return np.clip(base + (1 - base) * edge_boost, 0.0, 0.95)


def nearest_known_features(values):
    """Pour chaque point : valeur et distance du point connu le plus proche
    avant / après (features de type "lag direct", idée #11 de benchmark.md).
    values : array 1D (une courbe), avec NaN aux positions manquantes."""
    n = len(values)
    observed = ~np.isnan(values)
    last_val = np.full(n, np.nan)
    last_dist = np.full(n, np.nan)
    cur_val, cur_idx = np.nan, None
    for i in range(n):
        if observed[i]:
            cur_val, cur_idx = values[i], i
        last_val[i] = cur_val
        last_dist[i] = (i - cur_idx) if cur_idx is not None else n  # bounded: no point found -> full length, not ~1e9
    next_val = np.full(n, np.nan)
    next_dist = np.full(n, np.nan)
    cur_val, cur_idx = np.nan, None
    for i in range(n - 1, -1, -1):
        if observed[i]:
            cur_val, cur_idx = values[i], i
        next_val[i] = cur_val
        next_dist[i] = (cur_idx - i) if cur_idx is not None else n
    fallback = np.nanmean(values)
    last_val = np.where(np.isnan(last_val), fallback, last_val)
    next_val = np.where(np.isnan(next_val), fallback, next_val)
    # normalise les distances sur [0,1] (comparable aux autres features engineered
    # sin/cos/weekend) plutôt que de laisser Ridge voir une feature à l'échelle brute ~n
    last_dist = last_dist / n
    next_dist = next_dist / n
    return np.column_stack([last_val, last_dist, next_val, next_dist])


_CLUSTER_CACHE = {}


def fit_shape_clusters(donor_df, n_clusters=25, random_state=RANDOM_STATE):
    # keyed by the actual donor set (not just "any cluster fit done so far") so
    # that switching donor_df (e.g. X_train donors -> X_test donors within the
    # same process) doesn't silently reuse a stale, mismatched clustering.
    key = (tuple(donor_df.columns), n_clusters, random_state)
    if key in _CLUSTER_CACHE:
        return _CLUSTER_CACHE[key]
    filled = donor_df.ffill().bfill()
    Z = (filled - filled.mean()) / (filled.std() + 1e-9)
    km = KMeans(n_clusters=n_clusters, n_init=4, max_iter=100, random_state=random_state)
    labels = km.fit_predict(Z.T.values)
    labels = pd.Series(labels, index=donor_df.columns)
    _CLUSTER_CACHE[key] = (km, labels)
    return km, labels


def cluster_of(curve_filled, km):
    z = (curve_filled - curve_filled.mean()) / (curve_filled.std() + 1e-9)
    return int(km.predict(z.values.reshape(1, -1))[0])


def top_k_corr(target_obs, pool_vals_obs, k):
    t_c = target_obs - target_obs.mean()
    d_c = pool_vals_obs - pool_vals_obs.mean(axis=0, keepdims=True)
    num = d_c.T @ t_c
    denom = np.sqrt((d_c ** 2).sum(axis=0) * (t_c ** 2).sum()) + 1e-9
    corr = num / denom
    kk = min(k, corr.shape[0])
    top_idx = np.argpartition(-corr, kk - 1)[:kk]
    return top_idx, corr[top_idx]


def adaptive_top_k_corr(target_obs, pool_vals_obs, k_min=5, k_max=25, corr_threshold=0.15):
    """K variable par courbe (idée #7) : prend tous les voisins au-dessus d'un
    seuil de corrélation (jusqu'à k_max), ou repli sur les k_min meilleurs si
    aucun ne franchit le seuil -> évite de diluer un bon signal avec k_max
    voisins fixes quand seuls 3-4 sont vraiment corrélés (ou l'inverse)."""
    t_c = target_obs - target_obs.mean()
    d_c = pool_vals_obs - pool_vals_obs.mean(axis=0, keepdims=True)
    num = d_c.T @ t_c
    denom = np.sqrt((d_c ** 2).sum(axis=0) * (t_c ** 2).sum()) + 1e-9
    corr = num / denom
    order = np.argsort(-corr)
    above = order[corr[order] >= corr_threshold]
    chosen = above[:k_max] if len(above) >= k_min else order[:k_min]
    return chosen, corr[chosen]


def adaptive_top_k_corr_euclid(target_obs, pool_vals_obs, k_min=5, k_max=25,
                                corr_threshold=0.15, euclid_weight=0.3):
    """Comme adaptive_top_k_corr, mais le classement des voisins mélange la
    corrélation de Pearson (forme) avec une proximité de NIVEAU/AMPLITUDE brute
    (deux courbes peuvent avoir la même forme, corr=1, mais des échelles très
    différentes que la corrélation seule ne pénalise pas). Le seuil de
    sélection reste basé sur la corrélation brute (la proximité de niveau ne
    fait que réordonner / départager parmi les voisins déjà corrélés).

    NB : une v1 de cette fonction normalisait aussi le pool par son PROPRE
    écart-type avant de calculer la distance euclidienne -- mathématiquement,
    sur deux séries centrées-réduites par leur propre écart-type,
    distance² = 2n(1-corr) est une fonction déterministe de la corrélation
    (identité algébrique), donc ce "score complémentaire" ne faisait que
    reformuler la corrélation (résultats bit-identiques en ablation, cf.
    benchmark.md § V9). Ici on ne renormalise QUE par l'échelle de la cible
    (pas du donneur), pour que la distance reste sensible à un donneur de
    même forme mais de niveau différent -- une info que la corrélation ignore."""
    t_c = target_obs - target_obs.mean()
    d_c = pool_vals_obs - pool_vals_obs.mean(axis=0, keepdims=True)
    num = d_c.T @ t_c
    denom = np.sqrt((d_c ** 2).sum(axis=0) * (t_c ** 2).sum()) + 1e-9
    corr = num / denom

    t_scale = target_obs.std() + 1e-9
    euclid = np.sqrt(((d_c - t_c[:, None]) ** 2).mean(axis=0)) / t_scale
    euclid_score = 1.0 / (1.0 + euclid)  # -> similarité dans (0,1], 1 = identique

    combined = (1 - euclid_weight) * corr + euclid_weight * euclid_score
    order = np.argsort(-combined)
    above = order[corr[order] >= corr_threshold]
    chosen = above[:k_max] if len(above) >= k_min else order[:k_min]
    return chosen, corr[chosen]


def scale_neighbors(pool_sel, obs_idx, target_obs, robust):
    sel_obs = pool_sel[obs_idx, :]
    if robust:
        center = np.median(sel_obs, axis=0)
        # floor the spread relative to the neighbor's own level: flat/repeated
        # segments (very common at night for consumption curves) can give a
        # MAD of ~0, which would otherwise blow up the t_spread/spread ratio.
        floor = 0.02 * (np.abs(center) + 1.0)
        spread = np.maximum(1.4826 * np.median(np.abs(sel_obs - center), axis=0), floor)
        t_center = np.median(target_obs)
        t_spread = max(1.4826 * np.median(np.abs(target_obs - t_center)), 0.02 * (abs(t_center) + 1.0))
    else:
        center = sel_obs.mean(axis=0)
        spread = sel_obs.std(axis=0) + 1e-9
        t_center = target_obs.mean()
        t_spread = target_obs.std() + 1e-9
    # cap the scaling ratio itself: even with the floor above, a target that's
    # far more/less volatile than a given neighbor shouldn't be amplified without bound.
    ratio = np.clip(t_spread / spread, 0.1, 10.0)
    return (pool_sel - center) * ratio + t_center


def weighted_knn_pred(pool_vals, obs_idx, miss_idx, target_obs, k, robust,
                       adaptive_k=False, use_euclid=False, corr_threshold=0.15):
    # BUG CORRIGE : adaptive_top_k_corr(_euclid) ignoraient silencieusement le
    # `k` de l'appelant et utilisaient toujours leur k_max=25 par défaut ->
    # tout appel avec use_euclid=True (donc tout make_v9_model, cf. V9) et un
    # k != 25 testait en réalité exactement k_max=25 sans le signaler (2
    # configs bit-identiques a X0 dans xgboost_bench.py: T7_k15, T8_k35,
    # cf. benchmark.md). k est maintenant bien transmis comme k_max.
    # corr_threshold : exposé pour la piste Gemini #3 (filtrage donneurs plus
    # strict) -- défaut 0.15 inchangé, donc rétro-compatible avec V8/V9/V10.
    if use_euclid:
        top_idx, corr = adaptive_top_k_corr_euclid(target_obs, pool_vals[obs_idx, :], k_max=k, corr_threshold=corr_threshold)
    elif adaptive_k:
        top_idx, corr = adaptive_top_k_corr(target_obs, pool_vals[obs_idx, :], k_max=k, corr_threshold=corr_threshold)
    else:
        top_idx, corr = top_k_corr(target_obs, pool_vals[obs_idx, :], k)
    weights = np.clip(corr, 1e-3, None)
    weights = weights / weights.sum()
    pool_sel = pool_vals[:, top_idx]
    scaled = scale_neighbors(pool_sel, obs_idx, target_obs, robust)
    return (scaled[miss_idx, :] * weights).sum(axis=1), top_idx, scaled


def ridge_local_pred(scaled_neighbors, obs_idx, miss_idx, target_obs, extra_feats, ridge_alpha=1.0):
    X_obs = np.hstack([scaled_neighbors[obs_idx, :], extra_feats[obs_idx, :]])
    X_miss = np.hstack([scaled_neighbors[miss_idx, :], extra_feats[miss_idx, :]])
    reg = Ridge(alpha=ridge_alpha)
    reg.fit(X_obs, target_obs)
    return reg.predict(X_miss)


def time_features(index):
    dt_index = pd.to_datetime(index)
    hour = dt_index.hour.values + dt_index.minute.values / 60.0
    sin_h = np.sin(2 * np.pi * hour / 24)
    cos_h = np.cos(2 * np.pi * hour / 24)
    weekend = (dt_index.dayofweek.values >= 5).astype(float)
    return np.column_stack([sin_h, cos_h, weekend])


# ---------------------------------------------------------------------------
# V1 : modèle KNN inter-courbes de départ (repris de evaluate.py)
# ---------------------------------------------------------------------------

def make_v1(k=20, blend_interp=0.3):
    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        pool_vals = pool.ffill().bfill().values.astype(np.float64)
        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            observed = s.notna().values
            missing = ~observed
            if not missing.any():
                preds[col] = s
                continue
            obs_idx, miss_idx = np.where(observed)[0], np.where(missing)[0]
            target_obs = s.values[obs_idx]
            knn_pred, _, _ = weighted_knn_pred(pool_vals, obs_idx, miss_idx, target_obs, k, robust=False)
            interp = fill_nan_with_interpolation(s).values[miss_idx]
            pred = blend_interp * interp + (1 - blend_interp) * knn_pred
            filled = s.copy()
            filled.iloc[miss_idx] = pred
            preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model


# ---------------------------------------------------------------------------
# V2..V5 : un seul modèle paramétrable, activé en cascade
# ---------------------------------------------------------------------------

def make_stepped_model(k=20, use_cluster=False, adaptive_alpha=False,
                        use_ridge=False, robust_scale=False, n_clusters=25):
    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        # ffill/bfill is per-column independent -> safe to fill ONCE globally
        # and reuse (numpy) slices for any column subset (e.g. cluster subsets)
        # instead of re-filling a pandas subset from scratch for every target.
        pool_vals = pool.ffill().bfill().values.astype(np.float64)

        km = cluster_col_idx = None
        if use_cluster:
            km, donor_labels = fit_shape_clusters(donor_df, n_clusters=n_clusters)
            if dev_true_df.shape[1] > 0:
                dev_filled = dev_true_df.ffill().bfill()
                Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
                dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
                pool_labels = pd.concat([donor_labels, dev_labels])
            else:
                pool_labels = donor_labels
            # precompute, once per repeat, the column *positions* belonging to
            # each cluster (numpy int arrays) so per-target lookup is O(1) slicing
            pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
            cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}

        tfeat = time_features(val_holed_df.index)

        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            observed = s.notna().values
            missing = ~observed
            if not missing.any():
                preds[col] = s
                continue
            obs_idx, miss_idx = np.where(observed)[0], np.where(missing)[0]
            target_obs = s.values[obs_idx]

            if use_cluster:
                interp_full = fill_nan_with_interpolation(s)
                cl = cluster_of(interp_full, km)
                same_idx = cluster_col_idx.get(cl)
                if same_idx is not None and len(same_idx) >= max(2 * k, 30):
                    sub_vals = pool_vals[:, same_idx]
                else:
                    sub_vals = pool_vals
            else:
                sub_vals = pool_vals

            knn_pred, top_idx, scaled = weighted_knn_pred(
                sub_vals, obs_idx, miss_idx, target_obs, k, robust=robust_scale
            )

            if use_ridge and len(obs_idx) >= 15:
                pred_neighbors = ridge_local_pred(scaled, obs_idx, miss_idx, target_obs, tfeat)
            else:
                pred_neighbors = knn_pred

            interp_vals = fill_nan_with_interpolation(s).values
            interp = interp_vals[miss_idx]

            run_len = run_lengths_for(missing)[miss_idx]
            alphas = np.array([gap_alpha(rl, adaptive_alpha) for rl in run_len])
            pred = alphas * interp + (1 - alphas) * pred_neighbors

            filled = s.copy()
            filled.iloc[miss_idx] = pred
            preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model


# ---------------------------------------------------------------------------
# V6 : stacking - méta-modèle Ridge combinant interp / cluster-KNN / ridge-local
# ---------------------------------------------------------------------------

def make_v6(k=20, n_clusters=25, dev_subsample=150, random_state=RANDOM_STATE):
    rng = np.random.RandomState(random_state)

    def base_preds_for_col(s, pool_vals, cluster_col_idx, self_pos, km, tfeat, k):
        observed = s.notna().values
        missing = ~observed
        if not missing.any():
            return None
        obs_idx, miss_idx = np.where(observed)[0], np.where(missing)[0]
        target_obs = s.values[obs_idx]

        interp_full = fill_nan_with_interpolation(s)
        interp = interp_full.values[miss_idx]

        cl = cluster_of(interp_full, km)
        same_idx = cluster_col_idx.get(cl)
        if self_pos is not None and same_idx is not None:
            same_idx = same_idx[same_idx != self_pos]
        if same_idx is not None and len(same_idx) >= max(2 * k, 30):
            sub_vals = pool_vals[:, same_idx]
        elif self_pos is not None:
            keep = np.arange(pool_vals.shape[1]) != self_pos
            sub_vals = pool_vals[:, keep]
        else:
            sub_vals = pool_vals

        knn_pred, top_idx, scaled = weighted_knn_pred(sub_vals, obs_idx, miss_idx, target_obs, k, robust=True)

        if len(obs_idx) >= 15:
            ridge_pred = ridge_local_pred(scaled, obs_idx, miss_idx, target_obs, tfeat)
        else:
            ridge_pred = knn_pred

        return miss_idx, interp, knn_pred, ridge_pred

    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        # filled ONCE globally (ffill/bfill is per-column independent, so this
        # is equivalent to filling any column subset separately, but O(pool) not O(n_targets * pool))
        pool_vals = pool.ffill().bfill().values.astype(np.float64)

        km, donor_labels = fit_shape_clusters(donor_df, n_clusters=n_clusters)
        dev_filled = dev_true_df.ffill().bfill()
        Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
        dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
        pool_labels = pd.concat([donor_labels, dev_labels])
        pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
        cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
        col_pos = {c: i for i, c in enumerate(pool.columns)}

        tfeat_dev = time_features(dev_holed_df.index)
        tfeat_val = time_features(val_holed_df.index)

        # --- fit stacking meta-model on a subsample of dev columns ---
        dev_cols = list(dev_holed_df.columns)
        sub_cols = rng.choice(dev_cols, size=min(dev_subsample, len(dev_cols)), replace=False)
        rows_X, rows_y = [], []
        for col in sub_cols:
            s = dev_holed_df[col]
            # exclude this dev curve from its own donor pool (avoid leakage)
            out = base_preds_for_col(s, pool_vals, cluster_col_idx, col_pos[col], km, tfeat_dev, k)
            if out is None:
                continue
            miss_idx, interp, knn_pred, ridge_pred = out
            y_true = dev_true_df[col].values[miss_idx]
            rows_X.append(np.column_stack([interp, knn_pred, ridge_pred]))
            rows_y.append(y_true)
        X_meta = np.vstack(rows_X)
        y_meta = np.concatenate(rows_y)
        meta = Ridge(alpha=1.0, positive=True)
        meta.fit(X_meta, y_meta)

        # --- apply to val columns ---
        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            out = base_preds_for_col(s, pool_vals, cluster_col_idx, None, km, tfeat_val, k)
            if out is None:
                preds[col] = s
                continue
            miss_idx, interp, knn_pred, ridge_pred = out
            X_stack = np.column_stack([interp, knn_pred, ridge_pred])
            pred = meta.predict(X_stack)
            filled = s.copy()
            filled.iloc[miss_idx] = pred
            preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model


# ---------------------------------------------------------------------------
# V8 : V5 + 2 améliorations retenues sur 3 testées, parmi les 15 pistes de
# benchmark.md (ablation sur 1 split, cf. _ablation_debug.py) :
#   - #12 position relative dans le trou (bord vs milieu) -> alpha continu   : 140.9 -> 138.6 W  [gardée]
#   - #7  K adaptatif par seuil de corrélation (plutôt que top-k fixe)      : 140.9 -> 140.4 W  [gardée]
#   - #11 lags directs (valeur/distance au point connu avant/après)        : 140.9 -> 178.7 W  [REJETÉE]
# combinées (les 2 gardées) : 140.9 -> 138.1 W.
# Les lags directs dégradent nettement : sur les points observés (entraînement),
# la distance au point connu le plus proche est presque toujours ~1 (voisin
# immédiat) -> Ridge apprend une relation quasi triviale qui extrapole mal aux
# grandes distances des vrais trous longs. Piste abandonnée plutôt que masquée.
#
# Régresseur local injectable (make_regressor) pour pouvoir brancher Ridge OU
# TabPFNRegressor sur EXACTEMENT le même pipeline de features -> comparaison
# tête-à-tête équitable entre tabpfn_bench.py et une évaluation Ridge.
# ---------------------------------------------------------------------------

def local_regression_pred(make_regressor, scaled_neighbors, obs_idx, miss_idx, target_obs, extra_feats):
    X_obs = np.hstack([scaled_neighbors[obs_idx, :], extra_feats[obs_idx, :]])
    X_miss = np.hstack([scaled_neighbors[miss_idx, :], extra_feats[miss_idx, :]])
    reg = make_regressor()
    reg.fit(X_obs, target_obs)
    return reg.predict(X_miss)


def make_v8_model(k=20, n_clusters=25, adaptive_k=True, make_regressor=None, random_state=RANDOM_STATE):
    if make_regressor is None:
        make_regressor = lambda: Ridge(alpha=1.0)

    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        pool_vals = pool.ffill().bfill().values.astype(np.float64)

        km, donor_labels = fit_shape_clusters(donor_df, n_clusters=n_clusters, random_state=random_state)
        if dev_true_df.shape[1] > 0:
            dev_filled = dev_true_df.ffill().bfill()
            Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
            dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
            pool_labels = pd.concat([donor_labels, dev_labels])
        else:
            pool_labels = donor_labels
        pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
        cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}

        tfeat = time_features(val_holed_df.index)

        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            observed = s.notna().values
            missing = ~observed
            if not missing.any():
                preds[col] = s
                continue
            obs_idx, miss_idx = np.where(observed)[0], np.where(missing)[0]
            target_obs = s.values[obs_idx]

            interp_full = fill_nan_with_interpolation(s)
            cl = cluster_of(interp_full, km)
            same_idx = cluster_col_idx.get(cl)
            if same_idx is not None and len(same_idx) >= max(2 * k, 30):
                sub_vals = pool_vals[:, same_idx]
            else:
                sub_vals = pool_vals

            knn_pred, top_idx, scaled = weighted_knn_pred(
                sub_vals, obs_idx, miss_idx, target_obs, k, robust=True, adaptive_k=adaptive_k
            )

            # NB: les lags directs (nearest_known_features, idée #11) ont été testés
            # et REJETÉS ici (cf. benchmark.md) : sur les points observés utilisés pour
            # l'entraînement, le point connu le plus proche est presque toujours à
            # distance ~1, donc Ridge apprend une relation quasi triviale qui
            # extrapole mal aux grandes distances des vrais trous longs (-27% MAE
            # en test). On garde uniquement les 2 idées qui ont tenu à l'ablation.
            extra = tfeat

            if len(obs_idx) >= 15:
                pred_neighbors = local_regression_pred(make_regressor, scaled, obs_idx, miss_idx, target_obs, extra)
            else:
                pred_neighbors = knn_pred

            interp = interp_full.values[miss_idx]
            run_len_full = run_lengths_for(missing)
            edge_dist_full = edge_distance_for(missing)
            alphas = np.array([
                continuous_alpha(run_len_full[m], edge_dist_full[m], adaptive=True) for m in miss_idx
            ])
            pred = alphas * interp + (1 - alphas) * pred_neighbors

            filled = s.copy()
            filled.iloc[miss_idx] = pred
            preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model


# ---------------------------------------------------------------------------
# V9 : V8 + 1 piste retenue sur 2 candidates génériques testées par ablation
# (cf. _ablation_v9.py ; les 3 autres candidates de la sélection de 5 sont
# spécifiques TabPFN, testées séparément dans tabpfn_bench2.py) :
#   - #1 distance euclidienne complémentaire à la corrélation (voisins) : 93.1 -> 92.2 W  [gardée]
#   - #2 alpha modulé par la volatilité locale de la courbe             : 93.1 -> 95.2 W  [REJETÉE]
# v1 de la distance euclidienne normalisait aussi par l'écart-type du DONNEUR
# -> résultats bit-identiques à la corrélation seule (identité algébrique
# distance²=2n(1-corr) sur deux séries centrées-réduites par leur propre
# écart-type : aucune information nouvelle). Corrigé en ne renormalisant que
# par l'échelle de la CIBLE (cf. adaptive_top_k_corr_euclid).
# Même régresseur injectable que V8 (Ridge ou TabPFN).
# ---------------------------------------------------------------------------

def make_v9_model(k=20, n_clusters=25, adaptive_k=True, use_euclid=True,
                   use_volatility=False, make_regressor=None, random_state=RANDOM_STATE,
                   use_lag_feats=False, use_cluster_mean_feat=False):
    """use_lag_feats / use_cluster_mean_feat : features optionnelles pour le
    régresseur local, testées ici pour X0+ (XGBoost) -- use_lag_feats avait
    été rejetée pour Ridge (cf. V8, extrapolation linéaire hors plage), mais
    un modèle à base d'arbres ne extrapole pas de la même façon -> vaut le
    coup d'être re-testée indépendamment par régresseur plutôt que supposée
    mauvaise pour tous."""
    if make_regressor is None:
        make_regressor = lambda: Ridge(alpha=1.0)

    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        pool_vals = pool.ffill().bfill().values.astype(np.float64)

        km, donor_labels = fit_shape_clusters(donor_df, n_clusters=n_clusters, random_state=random_state)
        if dev_true_df.shape[1] > 0:
            dev_filled = dev_true_df.ffill().bfill()
            Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
            dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
            pool_labels = pd.concat([donor_labels, dev_labels])
        else:
            pool_labels = donor_labels
        pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
        cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}

        tfeat = time_features(val_holed_df.index)

        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            observed = s.notna().values
            missing = ~observed
            if not missing.any():
                preds[col] = s
                continue
            obs_idx, miss_idx = np.where(observed)[0], np.where(missing)[0]
            target_obs = s.values[obs_idx]

            interp_full = fill_nan_with_interpolation(s)
            cl = cluster_of(interp_full, km)
            same_idx = cluster_col_idx.get(cl)
            if same_idx is not None and len(same_idx) >= max(2 * k, 30):
                sub_vals = pool_vals[:, same_idx]
            else:
                sub_vals = pool_vals

            knn_pred, top_idx, scaled = weighted_knn_pred(
                sub_vals, obs_idx, miss_idx, target_obs, k, robust=True,
                adaptive_k=adaptive_k, use_euclid=use_euclid,
            )

            extra = tfeat
            if use_cluster_mean_feat:
                cluster_mean_col = scaled.mean(axis=1, keepdims=True)
                extra = np.hstack([extra, cluster_mean_col])
            if use_lag_feats:
                lag_feats = nearest_known_features(s.values)
                extra = np.hstack([extra, lag_feats])

            if len(obs_idx) >= 15:
                pred_neighbors = local_regression_pred(make_regressor, scaled, obs_idx, miss_idx, target_obs, extra)
            else:
                pred_neighbors = knn_pred

            interp = interp_full.values[miss_idx]
            run_len_full = run_lengths_for(missing)
            edge_dist_full = edge_distance_for(missing)
            alphas = np.array([
                continuous_alpha(run_len_full[m], edge_dist_full[m], adaptive=True) for m in miss_idx
            ])
            if use_volatility:
                vol = curve_volatility(s.values)
                alphas = alphas * volatility_alpha_mult(vol)
            pred = alphas * interp + (1 - alphas) * pred_neighbors

            filled = s.copy()
            filled.iloc[miss_idx] = pred
            preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model


# ---------------------------------------------------------------------------
# Round 4 (E1-E8) : nouvelles pistes de feature engineering issues de
# recherche (PMM/MICE, "time aggregation features", décomposition
# saisonnière légère), construites sur V9 + LightGBM (meilleur modèle a ce
# stade, cf. xgboost_bench_round2_results.json R1_lightgbm). Chaque piste
# activable indépendamment pour ablation avant combinaison éventuelle,
# même logique honnête que V8/V9 (garder seulement ce qui tient).
# ---------------------------------------------------------------------------

def same_curve_seasonality_feature(values, index):
    """Médiane des valeurs OBSERVEES de la MEME courbe au même "bucket"
    horaire (heure:minute) -- version légère d'une décomposition
    saisonnière journalière (idée R9 différée, cf.
    xgboost_bench_round2.py), sans utiliser aucune autre courbe (pas de
    risque de fuite inter-courbes, uniquement les points réellement
    observés de la courbe elle-même)."""
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
    """Observed values at the same half-hour on nearby days/weeks.

    Unlike ``nearest_known_features`` rejected in V8, these features have
    exactly the same train/predict semantics: a source is present only when
    that timestamp is genuinely observed. NaNs are left for tree models.
    """
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
    """Moyenne/écart-type glissants du consensus des voisins (moyenne des
    voisins scalés) autour de chaque position -- capture une tendance
    locale plutôt qu'un seul point isolé (motivé par les "time aggregation
    features" pour XGBoost, cf. recherche arXiv 2601.10019)."""
    consensus = scaled.mean(axis=1)
    s = pd.Series(consensus)
    roll_mean = s.rolling(window, center=True, min_periods=1).mean().values
    roll_std = s.rolling(window, center=True, min_periods=1).std().fillna(0.0).values
    return np.column_stack([roll_mean, roll_std])


def neighbor_consensus_std_feature(scaled):
    """Écart-type inter-voisins à chaque position : mesure de désaccord
    local entre voisins (feature de confiance/bruit potentiellement
    exploitable par un modèle en arbres)."""
    return scaled.std(axis=1).reshape(-1, 1)


def time_features_v2(index):
    """time_features() + jour de semaine cyclique (sin/cos) au lieu du seul
    binaire week-end -- capture une différence Lun/Mer/Ven que le binaire
    week-end/semaine ignore complètement."""
    dt_index = pd.to_datetime(index)
    hour = dt_index.hour.values + dt_index.minute.values / 60.0
    sin_h = np.sin(2 * np.pi * hour / 24)
    cos_h = np.cos(2 * np.pi * hour / 24)
    dow = dt_index.dayofweek.values
    sin_d = np.sin(2 * np.pi * dow / 7)
    cos_d = np.cos(2 * np.pi * dow / 7)
    weekend = (dow >= 5).astype(float)
    return np.column_stack([sin_h, cos_h, sin_d, cos_d, weekend])


def time_features_full_seasonality(index, groups=None):
    """Encodages cycliques complets du calendrier.

    Les couples sinus/cosinus évitent les discontinuités artificielles entre
    23h30 et 00h00, dimanche et lundi, décembre et janvier, etc. La
    saisonnalité "année" est représentée par la position dans l'année
    (jour + fraction de journée), et non par le numéro civil 2023/2024 qui
    constitue une tendance et non un cycle.
    """
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

    groups = set(groups or ("hour", "dow", "dom", "week", "month", "year"))
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


def pchip_interp_feature(s):
    """Interpolation monotone (PCHIP, shape-preserving, sans oscillation)
    comme feature COMPLEMENTAIRE à l'interpolation linéaire (pas en
    remplacement) -- laisse le modèle arbitrer entre les deux plutôt que
    d'imposer un alpha fixe basé sur une seule méthode d'interpolation."""
    from scipy.interpolate import PchipInterpolator
    values = s.values.astype(float)
    observed = ~np.isnan(values)
    x_obs = np.where(observed)[0]
    if len(x_obs) < 3:
        return fill_nan_with_interpolation(s).values
    interp = PchipInterpolator(x_obs, values[x_obs], extrapolate=True)
    return interp(np.arange(len(values)))


def pmm_snap(pred, candidates, blend=0.3):
    """Predictive Mean Matching (technique standard en imputation multiple,
    cf. MICE/mixgb) : rapproche la prédiction continue du modèle (que le
    boosting tend à lisser) de la valeur voisine RÉELLEMENT observée la
    plus proche parmi les candidats à cette position -- préserve mieux la
    texture/distribution réelle des données qu'une prédiction moyenne
    seule. `blend` pondère la part de snapping (1.0 = snap complet)."""
    diffs = np.abs(candidates - pred[:, None])
    idx = np.argmin(diffs, axis=1)
    snapped = candidates[np.arange(len(pred)), idx]
    return blend * snapped + (1 - blend) * pred


def make_v10_model(k=20, n_clusters=25, adaptive_k=True, use_euclid=True,
                    make_regressor=None, random_state=RANDOM_STATE,
                    use_seasonality_feat=False, use_neighbor_rolling_feat=False,
                    use_neighbor_consensus_feat=False, use_weekday_cyclic=False,
                    use_pchip_feat=False, use_pmm=False, pmm_blend=0.3,
                    use_periodic_copy_feat=False,
                    use_periodic_consensus_feat=False,
                    use_full_calendar_feat=False, calendar_groups=None,
                    normalize_periodic_feat=False,
                    include_periodic_availability=False,
                    periodic_alpha_strength=0.0):
    """V9 + 6 nouvelles pistes de feature engineering / post-traitement
    (round 4), chacune activable indépendamment pour ablation avant
    combinaison. Même structure que make_v9_model."""
    if make_regressor is None:
        make_regressor = lambda: Ridge(alpha=1.0)

    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        pool_vals = pool.ffill().bfill().values.astype(np.float64)

        km, donor_labels = fit_shape_clusters(donor_df, n_clusters=n_clusters, random_state=random_state)
        if dev_true_df.shape[1] > 0:
            dev_filled = dev_true_df.ffill().bfill()
            Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
            dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
            pool_labels = pd.concat([donor_labels, dev_labels])
        else:
            pool_labels = donor_labels
        pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
        cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}

        if use_full_calendar_feat:
            tfeat = time_features_full_seasonality(val_holed_df.index, calendar_groups)
        elif use_weekday_cyclic:
            tfeat = time_features_v2(val_holed_df.index)
        else:
            tfeat = time_features(val_holed_df.index)

        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            observed = s.notna().values
            missing = ~observed
            if not missing.any():
                preds[col] = s
                continue
            obs_idx, miss_idx = np.where(observed)[0], np.where(missing)[0]
            target_obs = s.values[obs_idx]

            interp_full = fill_nan_with_interpolation(s)
            cl = cluster_of(interp_full, km)
            same_idx = cluster_col_idx.get(cl)
            if same_idx is not None and len(same_idx) >= max(2 * k, 30):
                sub_vals = pool_vals[:, same_idx]
            else:
                sub_vals = pool_vals

            knn_pred, top_idx, scaled = weighted_knn_pred(
                sub_vals, obs_idx, miss_idx, target_obs, k, robust=True,
                adaptive_k=adaptive_k, use_euclid=use_euclid,
            )

            extra = tfeat
            if use_seasonality_feat:
                extra = np.hstack([extra, same_curve_seasonality_feature(s.values, val_holed_df.index)])
            if use_neighbor_rolling_feat:
                extra = np.hstack([extra, neighbor_rolling_features(scaled)])
            if use_neighbor_consensus_feat:
                extra = np.hstack([extra, neighbor_consensus_std_feature(scaled)])
            if use_pchip_feat:
                extra = np.hstack([extra, pchip_interp_feature(s).reshape(-1, 1)])
            if use_periodic_copy_feat:
                extra = np.hstack([extra, periodic_copy_features(
                    s.values, include_summary=use_periodic_consensus_feat,
                    normalize=normalize_periodic_feat,
                    include_availability=include_periodic_availability)])

            if len(obs_idx) >= 15:
                pred_neighbors = local_regression_pred(make_regressor, scaled, obs_idx, miss_idx, target_obs, extra)
            else:
                pred_neighbors = knn_pred

            if use_pmm:
                pred_neighbors = pmm_snap(pred_neighbors, scaled[miss_idx, :], blend=pmm_blend)

            interp = interp_full.values[miss_idx]
            run_len_full = run_lengths_for(missing)
            edge_dist_full = edge_distance_for(missing)
            alphas = np.array([
                continuous_alpha(run_len_full[m], edge_dist_full[m], adaptive=True) for m in miss_idx
            ])
            if periodic_alpha_strength > 0:
                alphas *= 1.0 - periodic_alpha_strength * periodic_agreement(s.values)[miss_idx]
            pred = alphas * interp + (1 - alphas) * pred_neighbors

            filled = s.copy()
            filled.iloc[miss_idx] = pred
            preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model


# ---------------------------------------------------------------------------
# Round 5 (G1-G6) : 6 pistes proposées par un second modèle (Gemini) sur la
# base du meilleur pipeline V10 -- interpolation PCHIP, alpha forcé à 1.0 sur
# les trous courts, seuil de corrélation resserré, ensemble TabPFN+CatBoost,
# plafonnement supérieur post-hoc, output_type median explicite. Chacune
# testée individuellement sur make_v11_model (V10 + ces options), même
# logique d'ablation honnête que les rounds précédents.
# ---------------------------------------------------------------------------

def pchip_fill(s):
    """Remplace l'interpolation linéaire par PCHIP (spline monotone,
    préserve la courbure naturelle des pics/creux qu'une interpolation
    linéaire "coupe" en ligne droite -- piste Gemini #1). Fallback sur
    l'interpolation linéaire si trop peu de points observés (< 3) pour
    ajuster un spline de façon fiable."""
    from scipy.interpolate import PchipInterpolator
    values = s.values.astype(float)
    observed = ~np.isnan(values)
    obs_idx = np.where(observed)[0]
    if len(obs_idx) < 3:
        return fill_nan_with_interpolation(s)
    pchip = PchipInterpolator(obs_idx, values[obs_idx], extrapolate=True)
    out = s.copy()
    out.iloc[:] = pchip(np.arange(len(values)))
    return out


def cap_upper_predictions(pred, observed_values, margin=1.2):
    """Plafonne les prédictions au-dessus d'une borne dérivée des valeurs
    RÉELLEMENT observées de la courbe cible (piste Gemini #5) -- évite qu'un
    pic aberrant isolé pénalise lourdement la MAE. `margin` laisse une marge
    au-dessus du quantile 99.5% observé (pas un simple max, plus robuste à
    un seul point extrême déjà présent dans les données observées)."""
    obs = observed_values[~np.isnan(observed_values)]
    if len(obs) == 0:
        return pred
    cap = np.quantile(obs, 0.995) * margin
    return np.minimum(pred, cap)


def make_v11_model(k=20, n_clusters=25, adaptive_k=True, use_euclid=True,
                    make_regressor=None, random_state=RANDOM_STATE,
                    use_seasonality_feat=False, interp_method="linear",
                    corr_threshold=0.15, force_full_interp_below=None,
                    upper_cap_margin=None):
    """V10 + 4 pistes Gemini directement intégrables au pipeline de features
    (l'ensemble TabPFN+CatBoost et le output_type median explicite se
    branchent via make_regressor, testés à part dans tabpfn_bench_gemini.py) :
      - interp_method="pchip" : interpolation PCHIP au lieu de linéaire,
        comme baseline ET comme composante du blend alpha (piste #1)
      - force_full_interp_below=N : alpha forcé à 1.0 (100% interpolation,
        pas de modèle) pour les trous de longueur <= N (piste #2)
      - corr_threshold : seuil de corrélation resserré pour la sélection des
        voisins, avec repli automatique sur k_min voisins si aucun ne passe
        le seuil (piste #3, cf. adaptive_top_k_corr(_euclid))
      - upper_cap_margin : plafonnement post-hoc des prédictions (piste #5)
    """
    if make_regressor is None:
        make_regressor = lambda: Ridge(alpha=1.0)
    interp_fn = pchip_fill if interp_method == "pchip" else fill_nan_with_interpolation

    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        pool_vals = pool.ffill().bfill().values.astype(np.float64)

        km, donor_labels = fit_shape_clusters(donor_df, n_clusters=n_clusters, random_state=random_state)
        if dev_true_df.shape[1] > 0:
            dev_filled = dev_true_df.ffill().bfill()
            Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
            dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
            pool_labels = pd.concat([donor_labels, dev_labels])
        else:
            pool_labels = donor_labels
        pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
        cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}

        tfeat = time_features(val_holed_df.index)

        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            observed = s.notna().values
            missing = ~observed
            if not missing.any():
                preds[col] = s
                continue
            obs_idx, miss_idx = np.where(observed)[0], np.where(missing)[0]
            target_obs = s.values[obs_idx]

            interp_full = interp_fn(s)
            # clustering toujours base sur l'interp lineaire (le KMeans a ete
            # calibre dessus) -- reste stable quel que soit interp_method,
            # seul le PRED final / le blend alpha change avec interp_method
            cl = cluster_of(fill_nan_with_interpolation(s), km)
            same_idx = cluster_col_idx.get(cl)
            if same_idx is not None and len(same_idx) >= max(2 * k, 30):
                sub_vals = pool_vals[:, same_idx]
            else:
                sub_vals = pool_vals

            knn_pred, top_idx, scaled = weighted_knn_pred(
                sub_vals, obs_idx, miss_idx, target_obs, k, robust=True,
                adaptive_k=adaptive_k, use_euclid=use_euclid, corr_threshold=corr_threshold,
            )

            extra = tfeat
            if use_seasonality_feat:
                extra = np.hstack([extra, same_curve_seasonality_feature(s.values, val_holed_df.index)])

            if len(obs_idx) >= 15:
                pred_neighbors = local_regression_pred(make_regressor, scaled, obs_idx, miss_idx, target_obs, extra)
            else:
                pred_neighbors = knn_pred

            interp = interp_full.values[miss_idx]
            run_len_full = run_lengths_for(missing)
            edge_dist_full = edge_distance_for(missing)
            alphas = np.array([
                continuous_alpha(run_len_full[m], edge_dist_full[m], adaptive=True) for m in miss_idx
            ])
            if force_full_interp_below is not None:
                short_gap = run_len_full[miss_idx] <= force_full_interp_below
                alphas = np.where(short_gap, 1.0, alphas)
            pred = alphas * interp + (1 - alphas) * pred_neighbors

            if upper_cap_margin is not None:
                pred = cap_upper_predictions(pred, s.values, margin=upper_cap_margin)

            filled = s.copy()
            filled.iloc[miss_idx] = pred
            preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model


# ---------------------------------------------------------------------------
# Harness partagé : N hold-out 90/10 INDEPENDANTS (pas de k-fold), une graine
# différente par repeat (random_state + i). Utilisé par ce module ET par
# tabpfn_bench.py pour garantir des évaluations strictement comparables
# (mêmes données, mêmes colonnes dev/val, mêmes graines).
# ---------------------------------------------------------------------------

def evaluate_repeated_splits(X_train, y_train, donor_cols, model_fn,
                              n_repeats=10, test_size=0.10, random_state=RANDOM_STATE,
                              n_val_sub=None, save_path=None, verbose=True,
                              start_repeat=0):
    from sklearn.model_selection import train_test_split
    holed_cols = list(y_train.columns)
    scores = []
    for i in range(start_repeat, n_repeats):
        seed_i = random_state + i
        dev_cols, val_cols_full = train_test_split(holed_cols, test_size=test_size, random_state=seed_i)
        if n_val_sub is not None and n_val_sub < len(val_cols_full):
            sub_rng = np.random.RandomState(seed_i)
            val_cols = list(sub_rng.choice(val_cols_full, size=n_val_sub, replace=False))
        else:
            val_cols = val_cols_full

        t0 = time.time()
        y_pred_val = model_fn(
            donor_df=X_train[donor_cols],
            dev_holed_df=X_train[dev_cols],
            dev_true_df=y_train[dev_cols],
            val_holed_df=X_train[val_cols],
        )
        mask = X_train[val_cols].isna()
        mae = official_mae(y_train[val_cols], y_pred_val, mask)
        dt = time.time() - t0
        scores.append(mae)
        if verbose:
            print(f"  repeat {i + 1}/{n_repeats}: MAE = {mae:8.2f} W "
                  f"(n_val_cols={len(val_cols)}/{len(val_cols_full)}, {dt:.1f}s)", flush=True)
        if save_path:
            import json
            with open(save_path, "w") as f:
                json.dump({"scores_so_far": scores, "n_repeats_done": i + 1, "n_repeats_target": n_repeats}, f)

    scores = np.array(scores)
    if verbose:
        print(f"  -> Mean MAE over {n_repeats} repeats: {scores.mean():.2f} +/- {scores.std():.2f} W\n")
    return scores


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Chargement des données...")
    X_train, y_train, X_test = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    steps = [
        ("V0_baseline_interp", baseline_model),
        ("V1_knn_correlation", make_v1(k=20, blend_interp=0.3)),
        ("V2_+cluster_knn", make_stepped_model(k=20, use_cluster=True)),
        ("V3_+adaptive_gap_alpha", make_stepped_model(k=20, use_cluster=True, adaptive_alpha=True)),
        ("V4_+ridge_local_time_feats", make_stepped_model(k=20, use_cluster=True, adaptive_alpha=True, use_ridge=True)),
        ("V5_+robust_scaling", make_stepped_model(k=20, use_cluster=True, adaptive_alpha=True, use_ridge=True, robust_scale=True)),
        ("V6_+stacking_meta_model", make_v6(k=20, dev_subsample=80)),
    ]

    results = {}
    for name, model_fn in steps:
        print(f"=== {name} ===")
        t0 = time.time()
        scores = repeated_holdout_eval(
            X_train, y_train, donor_cols, model_fn,
            n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )
        results[name] = scores
        print(f"  total time: {time.time()-t0:.1f}s\n")

    print("\n=== TABLEAU RECAPITULATIF ===")
    print(f"{'Etape':30s} {'MAE moy (W)':>12s} {'std (W)':>10s} {'gain vs V0':>12s} {'gain vs prec.':>14s}")
    prev = None
    base = results["V0_baseline_interp"].mean()
    for name, scores in results.items():
        m, sd = scores.mean(), scores.std()
        gain_v0 = 100 * (1 - m / base)
        gain_prev = 100 * (1 - m / prev) if prev is not None else 0.0
        print(f"{name:30s} {m:12.2f} {sd:10.2f} {gain_v0:11.1f}% {gain_prev:13.1f}%")
        prev = m

    import json
    with open("benchmark_results.json", "w") as f:
        json.dump({k: v.tolist() for k, v in results.items()}, f, indent=2)
    print("\nRésultats sauvegardés dans benchmark_results.json")
