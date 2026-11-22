"""Plus de 50 features supplémentaires construites sur le pool de courbes
donneuses (20 000+) et sur la courbe cible elle-même, ajoutées au meilleur
modèle local documenté (LightGBM + imputation itérative 2 passes, A4,
68.89 W -- cf. benchmark.md § 7bis) plutôt qu'à TabPFN (dont l'apprentissage
en contexte semble déjà extraire ce signal, cf. §15bis-15quater : 10 idées
similaires toutes rejetées côté TabPFN). Catalogue organisé en 6 groupes,
inspiré de tsfresh/catch22 (statistiques canoniques de série temporelle) et
de la littérature PMM/donor-pool (cf. recherche documentée dans la
conversation) :

  A. Statistiques du pool de voisins par position (12 features)
  B. Statistiques du pool de voisins, niveau courbe (5 features)
  C. Statistiques propres à la courbe cible (6 features)
  D. Composition/diversité du pool de voisins (6 features)
  E. Confiance/incertitude et interactions (6 features)
  F. Day-type, tendance locale, tarif (déjà prototypées pour TabPFN
     §15bis-15quater, réutilisées ici pour un régresseur différent, 15+ features)

Total : 50+ colonnes ajoutées à l'extra existant de V12/A4.
"""
import warnings

import numpy as np
import pandas as pd

import feature_engineering_bench as feb

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# A. Statistiques du pool de voisins par position (12 features)
# ---------------------------------------------------------------------------

def neighbor_pool_stats(scaled):
    """mean, median, std, MAD, min, max, range, Q25, Q75, IQR, skew, kurtosis
    du pool de voisins déjà remis à l'échelle, à CHAQUE position -- catalogue
    tsfresh/catch22 appliqué au pool plutôt qu'à la courbe elle-même."""
    with np.errstate(all="ignore"):
        mean = np.nanmean(scaled, axis=1, keepdims=True)
        median = np.nanmedian(scaled, axis=1, keepdims=True)
        std = np.nanstd(scaled, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(scaled - median), axis=1, keepdims=True)
        mn = np.nanmin(scaled, axis=1, keepdims=True)
        mx = np.nanmax(scaled, axis=1, keepdims=True)
        rng = mx - mn
        q25 = np.nanpercentile(scaled, 25, axis=1).reshape(-1, 1)
        q75 = np.nanpercentile(scaled, 75, axis=1).reshape(-1, 1)
        iqr = q75 - q25
        centered = scaled - mean
        m2 = np.nanmean(centered ** 2, axis=1, keepdims=True)
        m3 = np.nanmean(centered ** 3, axis=1, keepdims=True)
        m4 = np.nanmean(centered ** 4, axis=1, keepdims=True)
        skew = m3 / np.maximum(m2 ** 1.5, 1e-6)
        kurt = m4 / np.maximum(m2 ** 2, 1e-6) - 3.0
    out = np.hstack([mean, median, std, mad, mn, mx, rng, q25, q75, iqr, skew, kurt])
    return np.nan_to_num(out)


def neighbor_interval_width(scaled, lo=10, hi=90):
    p_lo = np.nanpercentile(scaled, lo, axis=1)
    p_hi = np.nanpercentile(scaled, hi, axis=1)
    return np.nan_to_num((p_hi - p_lo)).reshape(-1, 1)


def neighbor_trimmed_mean(scaled, trim=0.1):
    k = scaled.shape[1]
    n_trim = max(1, int(k * trim))
    sorted_vals = np.sort(scaled, axis=1)
    trimmed = sorted_vals[:, n_trim:-n_trim] if k > 2 * n_trim else sorted_vals
    return np.nan_to_num(np.nanmean(trimmed, axis=1)).reshape(-1, 1)


def neighbor_exp_weighted_mean(scaled, corr, half_life_rank=10):
    """Moyenne à décroissance exponentielle selon le RANG de corrélation
    (pas la valeur brute de corr) -- schéma de pondération alternatif au
    poids linéaire par corrélation déjà utilisé pour knn_pred."""
    order = np.argsort(-corr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(corr))
    w = np.exp(-ranks / max(half_life_rank, 1))
    w = w / w.sum()
    return np.nan_to_num((scaled * w).sum(axis=1)).reshape(-1, 1)


def neighbor_gradient_agreement(scaled):
    """Gradient moyen inter-voisins (différence t vs t-1) -- capture la
    tendance locale instantanée que la fenêtre glissante (neighbor_rolling,
    déjà en place) moyenne sur plusieurs points."""
    grad = np.diff(scaled, axis=0, prepend=scaled[:1, :])
    return np.nan_to_num(np.nanmean(grad, axis=1)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# B. Statistiques du pool de voisins, niveau courbe (scalaires diffusés) (5)
# ---------------------------------------------------------------------------

def neighbor_corr_level_stats(corr):
    mean_corr = float(np.mean(corr))
    std_corr = float(np.std(corr))
    n_lo = float(((corr >= 0.15) & (corr < 0.3)).sum())
    n_mid = float(((corr >= 0.3) & (corr < 0.5)).sum())
    n_hi = float((corr >= 0.5).sum())
    return np.array([mean_corr, std_corr, n_lo, n_mid, n_hi])


def best_single_neighbor(scaled, corr):
    """Valeur du voisin le plus corrélé, seul (ancre extrême, sans lissage)."""
    idx = int(np.argmax(corr))
    return np.nan_to_num(scaled[:, idx]).reshape(-1, 1)


# ---------------------------------------------------------------------------
# C. Statistiques propres à la courbe cible (6)
# ---------------------------------------------------------------------------

def own_curve_stats(values):
    obs = values[~np.isnan(values)]
    if len(obs) < 3:
        return np.zeros(6)
    mean, std = obs.mean(), obs.std() + 1e-9
    skew = float(np.mean(((obs - mean) / std) ** 3))
    kurt = float(np.mean(((obs - mean) / std) ** 4) - 3.0)
    scale = max(np.median(np.abs(obs - np.median(obs))), 1e-6)
    zero_frac = float((np.abs(obs) < 0.05 * scale).mean())
    diffs = np.diff(obs)
    n_flat_runs = float((np.abs(diffs) < 1e-9).sum()) / max(len(diffs), 1)
    cv = float(std / (abs(mean) + 1e-6))
    missing_frac = float(np.isnan(values).mean())
    return np.array([skew, kurt, zero_frac, n_flat_runs, cv, missing_frac])


def own_autocorr(values, lag=48):
    obs_idx = np.flatnonzero(~np.isnan(values))
    if len(obs_idx) < 10:
        return 0.0
    d = {i: v for i, v in zip(obs_idx, values[obs_idx])}
    pairs = [(d[i], d[i - lag]) for i in obs_idx if (i - lag) in d]
    if len(pairs) < 5:
        return 0.0
    a, b = np.array(pairs).T
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# D. Composition/diversité du pool + multi-échelle (jusqu'à 6)
# ---------------------------------------------------------------------------

def multiscale_consensus_disagreement(pool_vals, obs_idx, miss_idx, target_obs, k_small=10, k_large=40):
    """Désaccord entre le consensus à k=10 et à k=40 voisins -- signal de
    confiance : un grand écart signifie que l'ajout de voisins moins
    corrélés change beaucoup la prédiction (pool hétérogène, peu fiable)."""
    pred_small, _, _ = feb.weighted_knn_pred(pool_vals, obs_idx, miss_idx, target_obs,
                                              k_small, robust=True, adaptive_k=True, use_euclid=True)
    pred_large, _, _ = feb.weighted_knn_pred(pool_vals, obs_idx, miss_idx, target_obs,
                                              k_large, robust=True, adaptive_k=True, use_euclid=True)
    return np.nan_to_num(pred_large - pred_small).reshape(-1, 1), pred_small


def periodic_knn_disagreement(periodic_median, knn_pred_full):
    return np.nan_to_num(periodic_median - knn_pred_full).reshape(-1, 1)


# ---------------------------------------------------------------------------
# E. Confiance/incertitude et features "exposées" à l'explicit du régresseur
#    (déjà calculées ailleurs pour piloter alpha, mais jamais vues du
#    régresseur directement -- ajout "gratuit", pas de calcul nouveau) (3)
# ---------------------------------------------------------------------------

def alpha_context_features(missing, values):
    runs = feb.run_lengths_for(missing).astype(float)
    edges = feb.edge_distance_for(missing).astype(float)
    vol = feb.curve_volatility(values)
    return np.column_stack([runs, edges, np.full(len(missing), vol)])


# ---------------------------------------------------------------------------
# F. Day-type / tendance locale / tarif (déjà prototypées pour TabPFN,
#    réutilisées ici pour un régresseur d'arbres -- hypothèse de travail :
#    un arbre, contrairement à TabPFN, n'a pas d'apprentissage en contexte
#    et peut donc tirer parti de résumés explicites que TabPFN ignorait.
# ---------------------------------------------------------------------------

def weekday_seasonality_feature(values, index):
    dt_index = pd.to_datetime(index)
    bucket = dt_index.hour.values * 60 + dt_index.minute.values
    is_weekend = (dt_index.dayofweek.values >= 5).astype(int)
    observed = ~np.isnan(values)
    df = pd.DataFrame({"bucket": bucket, "weekend": is_weekend, "val": values})
    obs_df = df.loc[observed]
    med_by_daytype = obs_df.groupby(["weekend", "bucket"])["val"].median()
    med_by_bucket = obs_df.groupby("bucket")["val"].median()
    count_by_daytype = obs_df.groupby(["weekend", "bucket"])["val"].count()
    global_med = float(np.nanmedian(values)) if observed.any() else 0.0
    feat = np.full(len(values), global_med, dtype=float)
    for i in range(len(values)):
        key = (is_weekend[i], bucket[i])
        if key in med_by_daytype.index and count_by_daytype.get(key, 0) >= 2:
            feat[i] = med_by_daytype[key]
        elif bucket[i] in med_by_bucket.index:
            feat[i] = med_by_bucket[bucket[i]]
    return feat.reshape(-1, 1)


def peak_offpeak_flag(index):
    dt_index = pd.to_datetime(index)
    hour = dt_index.hour.values
    return ((hour >= 22) | (hour < 6)).astype(float).reshape(-1, 1)


def local_trend_feature(values, window=72):
    n = len(values)
    observed = ~np.isnan(values)
    obs_idx_all = np.flatnonzero(observed)
    obs_vals_all = values[obs_idx_all]
    global_med = float(np.nanmedian(values)) if observed.any() else 0.0
    slope = np.zeros(n, dtype=float)
    local_level_dev = np.zeros(n, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        mask = (obs_idx_all >= lo) & (obs_idx_all < hi)
        idx_local, val_local = obs_idx_all[mask], obs_vals_all[mask]
        if len(idx_local) >= 5:
            t = idx_local.astype(float) - i
            t_c = t - t.mean()
            denom = (t_c ** 2).sum()
            if denom > 1e-9:
                slope[i] = float((t_c * (val_local - val_local.mean())).sum() / denom)
            local_level_dev[i] = float(np.median(val_local) - global_med)
    scale = max(1.4826 * np.median(np.abs(obs_vals_all - global_med)), 1.0) if observed.any() else 1.0
    return np.column_stack([slope / scale, local_level_dev / scale])


def daytype_match_flags(index, lags=(48, 96, 144, 336)):
    """Flags 0/1 de correspondance semaine/week-end pour chaque copie
    périodique back/forward -- motivé par l'EDA (§15bis) : les lags
    ±96h/±144h tombent sur le mauvais type de jour ~50-55% du temps. Le
    masquage dur avait été rejeté pour TabPFN ; ici on garde la variante
    "douce" (flag additif, TabPFN §15bis idée #40) qui était la moins
    mauvaise des 4 variantes testées côté TabPFN -- à revalider pour
    LightGBM qui pourrait exploiter ce signal différemment."""
    dt_index = pd.to_datetime(index)
    is_weekend = (dt_index.dayofweek.values >= 5)
    n = len(is_weekend)
    cols = []
    for lag in lags:
        back_match = np.zeros(n, dtype=float)
        back_match[lag:] = (is_weekend[lag:] == is_weekend[:-lag]).astype(float)
        forward_match = np.zeros(n, dtype=float)
        forward_match[:-lag] = (is_weekend[:-lag] == is_weekend[lag:]).astype(float)
        cols.extend([back_match, forward_match])
    return np.column_stack(cols)


def distance_to_nearest_observed(missing):
    """Distance (en points) au point observé le plus proche (avant OU
    après) -- feature de contexte de trou minimale, jamais exposée
    explicitement au régresseur jusqu'ici (seule sa forme dérivée,
    edge_distance, alimente alpha)."""
    n = len(missing)
    observed = ~missing
    obs_idx = np.flatnonzero(observed)
    if len(obs_idx) == 0:
        return np.zeros(n).reshape(-1, 1)
    all_idx = np.arange(n)
    pos = np.searchsorted(obs_idx, all_idx)
    pos = np.clip(pos, 1, len(obs_idx) - 1)
    left = np.abs(all_idx - obs_idx[pos - 1])
    right = np.abs(obs_idx[pos] - all_idx)
    return np.minimum(left, right).astype(float).reshape(-1, 1)


def nearest_known_anchor_values(values):
    """Valeur brute du dernier point connu avant / après (ancre, PAS la
    distance -- déjà testée et rejetée en V8 combinée à la distance à cause
    d'une extrapolation Ridge trop naïve, cf. benchmark.md ; ici isolée de la
    distance, sans Ridge, pour un arbre qui n'extrapole pas linéairement)."""
    n = len(values)
    observed = ~np.isnan(values)
    last_val = np.full(n, np.nan)
    cur = np.nan
    for i in range(n):
        if observed[i]:
            cur = values[i]
        last_val[i] = cur
    next_val = np.full(n, np.nan)
    cur = np.nan
    for i in range(n - 1, -1, -1):
        if observed[i]:
            cur = values[i]
        next_val[i] = cur
    fallback = np.nanmedian(values) if observed.any() else 0.0
    last_val = np.where(np.isnan(last_val), fallback, last_val)
    next_val = np.where(np.isnan(next_val), fallback, next_val)
    return np.column_stack([last_val, next_val])


def periodic_total_availability(values, lags=(48, 96, 144, 336)):
    """Nombre total de copies périodiques disponibles (somme sur tous les
    lags/directions) -- résumé scalaire de la richesse d'information
    périodique à cette position, distinct du "count" déjà inclus dans
    periodic_copy_features (qui lui n'est ajouté que si include_summary)."""
    n = len(values)
    total = np.zeros(n, dtype=float)
    for lag in lags:
        total[lag:] += np.isfinite(values[:-lag]).astype(float)
        total[:-lag] += np.isfinite(values[lag:]).astype(float)
    return total.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Assemblage complet : >50 colonnes au total
# ---------------------------------------------------------------------------

def build_mega_extra(s, index, obs_idx, miss_idx, target_obs, scaled, corr,
                      pool_vals, missing, values_for_feats, periodic_median_full):
    """Construit le bloc complet de features supplémentaires (groupes A-F)
    sur TOUTE la longueur de la courbe (n points), comme le reste du
    pipeline (tfeat/periodic/seasonal) -- le régresseur s'entraîne sur
    obs_idx et prédit sur miss_idx, donc CHAQUE colonne doit porter un
    signal non constant aux deux endroits (bug corrigé : une première
    version ne calculait ces features qu'à miss_idx, les laissant à zéro
    partout à l'entraînement -> aucune information exploitable, régression
    bit-identique avec/sans le bloc mega). Retourne un tableau (n, 53)."""
    n = len(values_for_feats)

    pool_stats = neighbor_pool_stats(scaled)                                 # 12
    interval = neighbor_interval_width(scaled)                               # 1
    trimmed = neighbor_trimmed_mean(scaled)                                  # 1
    exp_w = neighbor_exp_weighted_mean(scaled, corr)                         # 1
    grad = neighbor_gradient_agreement(scaled)                               # 1

    corr_level = np.tile(neighbor_corr_level_stats(corr), (n, 1))            # 5
    best_neigh = best_single_neighbor(scaled, corr)                          # 1

    own_stats = np.tile(own_curve_stats(values_for_feats), (n, 1))           # 6
    autocorr_48 = np.full((n, 1), own_autocorr(values_for_feats, 48))        # 1

    # multiscale_consensus_disagreement ne connait que miss_idx (elle réutilise
    # weighted_knn_pred, qui prédit exactement sur miss_idx) -> on la calcule
    # sur TOUTES les positions en passant all_idx = arange(n) comme "miss_idx"
    all_idx = np.arange(n)
    disagree_all, pred_small_all = multiscale_consensus_disagreement(
        pool_vals, obs_idx, all_idx, target_obs)
    period_disagree_all = periodic_knn_disagreement(periodic_median_full, pred_small_all)

    alpha_ctx = alpha_context_features(missing, values_for_feats)            # 3

    weekday_season = weekday_seasonality_feature(values_for_feats, index)    # 1
    peak_offpeak = peak_offpeak_flag(index)                                  # 1
    trend = local_trend_feature(values_for_feats)                           # 2
    dist_nearest = distance_to_nearest_observed(missing)                     # 1
    daytype_flags = daytype_match_flags(index)                               # 8
    pchip = feb.pchip_interp_feature(s).reshape(-1, 1)                       # 1
    rolling = feb.neighbor_rolling_features(scaled)                          # 2
    anchors = nearest_known_anchor_values(values_for_feats)                  # 2
    periodic_avail = periodic_total_availability(values_for_feats)           # 1

    full = np.hstack([
        pool_stats, interval, trimmed, exp_w, grad,          # 16
        corr_level, best_neigh,                              # 6
        own_stats, autocorr_48,                               # 7
        disagree_all, period_disagree_all,                    # 2
        alpha_ctx,                                            # 3
        weekday_season, peak_offpeak, trend, dist_nearest,    # 5
        daytype_flags, pchip, rolling,                        # 11
        anchors, periodic_avail,                              # 3
    ])  # total = 16+6+7+2+3+5+11+3 = 53 nouvelles colonnes
    return full
