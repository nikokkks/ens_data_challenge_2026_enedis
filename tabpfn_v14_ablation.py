"""V14 : criblage de 20 idées candidates sur le pipeline TabPFN V12
(complete_column de generate_submission_tabpfn_v12.py), chacune activable
indépendamment via `opts`, testée en écran apparié 3x10 (mêmes graines que
la table V12 documentée : repeat1=122.386, repeat2=65.000, repeat3=50.348)
avant combinaison et confirmation 10x10.
"""
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tabpfn_client
from tabpfn_client import TabPFNRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import PowerTransformer
from scipy.stats import rankdata

from evaluate import load_data, official_mae, fill_nan_with_interpolation
import feature_engineering_bench as feb

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
RESULTS_PATH = "tabpfn_v14_ablation_results.json"
MAX_WORKERS = 20
MAX_RETRIES = 3
K_DEFAULT = 40


class MedianTabPFN:
    def __init__(self, random_state=RANDOM_STATE):
        self.reg = TabPFNRegressor(random_state=random_state)

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X, output_type="median")


class MeanMedianBlendTabPFN:
    """Idée #15 : moyenne mean/median TabPFN plutôt que median pur."""
    def __init__(self, random_state=RANDOM_STATE, mean_weight=0.5):
        self.reg = TabPFNRegressor(random_state=random_state)
        self.mean_weight = mean_weight

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        med = self.reg.predict(X, output_type="median")
        mean = self.reg.predict(X, output_type="mean")
        return self.mean_weight * mean + (1 - self.mean_weight) * med


class SeedEnsembleTabPFN:
    """Idée #14 : médiane de 2 seeds TabPFN indépendants."""
    def __init__(self, seeds=(42, 43)):
        self.regs = [TabPFNRegressor(random_state=s, thinking_mode=True, thinking_metric="mean_absolute_error") for s in seeds]

    def fit(self, X, y):
        for r in self.regs:
            r.fit(X, y)
        return self

    def predict(self, X):
        preds = np.vstack([r.predict(X, output_type="median") for r in self.regs])
        return np.median(preds, axis=0)


def same_curve_seasonality_weekday_feature(values, index):
    """Comme feb.same_curve_seasonality_feature, mais bucket = (weekend,
    demi-heure) au lieu de demi-heure seule -- motivé par l'EDA : ~20% de
    différence médiane semaine/week-end sur 64% des courbes échantillonnées,
    que le bucket unique (feb.same_curve_seasonality_feature) dilue en
    mélangeant les deux régimes dans une seule médiane par créneau.
    Repli sur le médian toutes-semaines si le bucket week-end est trop
    creux (peu de samedis/dimanches observés sur 3 semaines), puis sur la
    médiane globale."""
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


def periodic_copy_features_daytype(values, index, lags=(48, 96, 144, 336),
                                    include_summary=False, normalize=False,
                                    include_availability=False):
    """Comme feb.periodic_copy_features, mais masque (NaN) les copies dont le
    type de jour (semaine/week-end) NE correspond PAS à celui de la position
    cible -- motivé par l'EDA : les lags ±96h/±144h tombent sur le mauvais
    type de jour ~50-55% du temps (ex: vendredi+3j=lundi), alors que le
    combo médiane/agreement actuel les traite comme des copies fiables au
    même titre que ±48h (28.5% de décalage) ou ±336h (0%, toujours le même
    jour de semaine)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    dt_index = pd.to_datetime(index)
    is_weekend = (dt_index.dayofweek.values >= 5)
    cols = []
    for lag in lags:
        back = np.full(n, np.nan)
        forward = np.full(n, np.nan)
        back[lag:] = values[:-lag]
        forward[:-lag] = values[lag:]
        # back[i] provient de i-lag ; forward[i] provient de i+lag
        back_match = np.full(n, False)
        back_match[lag:] = is_weekend[lag:] == is_weekend[:-lag]
        forward_match = np.full(n, False)
        forward_match[:-lag] = is_weekend[:-lag] == is_weekend[lag:]
        back = np.where(back_match, back, np.nan)
        forward = np.where(forward_match, forward, np.nan)
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


def periodic_agreement_daytype(values, index, lags=(48, 96, 144, 336)):
    raw = periodic_copy_features_daytype(values, index, lags=lags)
    count = np.isfinite(raw).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(raw, axis=1)
        mad = np.nanmedian(np.abs(raw - med[:, None]), axis=1)
    obs = np.asarray(values)[np.isfinite(values)]
    scale = max(1.4826 * np.median(np.abs(obs - np.median(obs))), 1.0)
    return np.clip((count / raw.shape[1]) *
                   np.exp(-np.nan_to_num(mad, nan=np.inf) / scale), 0.0, 1.0)


def daytype_match_flag_features(index, lags=(48, 96, 144, 336)):
    """Variante douce de periodic_copy_features_daytype : au lieu de masquer
    (retirer) les copies dont le type de jour ne correspond pas, on garde
    TOUTES les valeurs et on ajoute juste un flag 0/1 de correspondance par
    lag (back/forward) -- signal additionnel plutôt que suppression
    d'information, laissant TabPFN décider lui-même du poids à donner à
    chaque copie plutôt que de le lui imposer (le masquage dur a été
    testé et rejeté : cf. benchmark.md, moins bon que la référence)."""
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


def local_trend_feature(values, window=72):
    """Dérive locale propre à la courbe : pour chaque position, régression
    robuste (moindres carrés simples sur points observés) de la valeur en
    fonction du temps dans une fenêtre de +/-`window` points (~1,5 jour à
    72 points), donnant une pente + un niveau local -- motivé par l'EDA :
    52% des courbes échantillonnées montrent >20% de dérive de niveau entre
    1ere et 2e moitié des 3 semaines, information qu'aucune feature actuelle
    ne capture directement (les copies périodiques comparent au jour/semaine
    précédent·e, pas de tendance locale ; la position annuelle globale est
    trop lente pour bouger en 3 semaines)."""
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
        else:
            local_level_dev[i] = 0.0
    scale = max(1.4826 * np.median(np.abs(obs_vals_all - global_med)), 1.0) if observed.any() else 1.0
    return np.column_stack([slope / scale, local_level_dev / scale])


def neighbor_mad_feature(scaled):
    """MAD (median absolute deviation) inter-voisins à chaque position --
    pendant de neighbor_consensus_std_feature (écart-type, déjà testé) mais
    robuste aux voisins aberrants isolés, jamais testé isolément."""
    med = np.median(scaled, axis=1, keepdims=True)
    return np.median(np.abs(scaled - med), axis=1).reshape(-1, 1)


def neighbor_interval_width_feature(scaled, lo=10, hi=90):
    """Largeur de l'intervalle [lo, hi] percentile inter-voisins à chaque
    position -- signal de confiance explicite (large = voisins en
    désaccord, donc prédiction incertaine), distinct du MAD/std (sensible à
    la queue de distribution, pas juste à la dispersion centrale)."""
    p_lo = np.percentile(scaled, lo, axis=1)
    p_hi = np.percentile(scaled, hi, axis=1)
    return (p_hi - p_lo).reshape(-1, 1)


def peak_offpeak_flag(index):
    """Flag heures pleines/creuses (approximation du tarif EDF standard,
    heures creuses ~22h-6h) -- feature calendaire française spécifique,
    jamais testée ici (seul heure/jour-de-semaine génériques existent)."""
    dt_index = pd.to_datetime(index)
    hour = dt_index.hour.values
    is_offpeak = ((hour >= 22) | (hour < 6)).astype(float)
    return is_offpeak.reshape(-1, 1)


def self_rolling_features(interp_values, windows=(12, 24, 48, 96)):
    """Statistiques glissantes (moyenne/écart-type/min/max) de la courbe
    interpolée elle-même à plusieurs échelles (6h/12h/24h/48h à 30 min de
    pas) -- contexte de tendance/volatilité multi-échelle qu'aucune feature
    actuelle ne capture directement (les copies périodiques comparent un
    point isolé au jour/semaine précédent·e, pas une fenêtre de contexte
    autour du point courant). Normalisé par l'échelle robuste propre à la
    courbe (médiane/MAD), comme les autres features "niveau" du pipeline."""
    obs = np.asarray(interp_values, dtype=float)
    center = np.median(obs)
    scale = max(1.4826 * np.median(np.abs(obs - center)), 0.02 * (abs(center) + 1.0), 1.0)
    s = pd.Series(obs)
    cols = []
    for w in windows:
        roll = s.rolling(w, center=True, min_periods=1)
        cols.append((roll.mean().to_numpy() - center) / scale)
        cols.append(roll.std().fillna(0.0).to_numpy() / scale)
        cols.append((roll.min().to_numpy() - center) / scale)
        cols.append((roll.max().to_numpy() - center) / scale)
    return np.column_stack(cols)


def neighbor_rolling_multiscale_features(scaled, windows=(3, 12, 24, 48, 96)):
    """Généralisation de feb.neighbor_rolling_features (fenêtre unique = 5)
    à plusieurs échelles temporelles -- capture la tendance du consensus
    voisin aussi bien à court terme (3 points, ~1h30) qu'à moyen terme
    (96 points, ~2 jours), jamais testé à plusieurs échelles simultanées."""
    consensus = scaled.mean(axis=1)
    s = pd.Series(consensus)
    cols = []
    for w in windows:
        roll = s.rolling(w, center=True, min_periods=1)
        cols.append(roll.mean().to_numpy())
        cols.append(roll.std().fillna(0.0).to_numpy())
    return np.column_stack(cols)


def neighbor_quantile_features(scaled, qs=(10, 25, 75, 90)):
    """Quantiles inter-voisins à chaque position (au-delà de la
    médiane/MAD/std/intervalle déjà disponibles) + asymétrie et étendue --
    capture la FORME de la distribution des voisins (une distribution
    étalée vers le haut n'est pas la même information qu'un simple écart-
    type symétrique)."""
    q = np.percentile(scaled, qs, axis=1).T  # (n, len(qs))
    mean = scaled.mean(axis=1)
    std = scaled.std(axis=1) + 1e-9
    median = np.median(scaled, axis=1)
    skew = (mean - median) / std
    mn = scaled.min(axis=1)
    mx = scaled.max(axis=1)
    return np.column_stack([q, skew, mn, mx])


def gap_context_features(missing):
    """Contexte du trou (longueur du run, distance au bord) donné
    DIRECTEMENT au régresseur -- jusqu'ici cette information ne pilotait
    que le poids alpha heuristique (continuous_alpha) en aval, jamais vue
    par le modèle lui-même. Hypothèse motivée par benchmark.md §11 (le
    blend alpha plafonne la contribution du régresseur, alpha moyen 0,39) :
    un modèle qui SAIT qu'il est au centre d'un trou de 400 points peut
    apprendre à s'appuyer différemment sur les copies périodiques qu'au
    bord d'un trou de 2 points, plutôt que de ne recevoir qu'un mélange
    post-hoc uniforme après coup."""
    n = len(missing)
    runs = feb.run_lengths_for(missing).astype(float)
    edges = feb.edge_distance_for(missing).astype(float)
    alpha_heur = np.array([feb.continuous_alpha(runs[i], edges[i], True) for i in range(n)])
    frac_through = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        if missing[i]:
            j = i
            while j < n and missing[j]:
                j += 1
            run_len = j - i
            if run_len > 1:
                frac_through[i:j] = (np.abs(np.arange(run_len) - (run_len - 1) / 2.0)
                                      / (run_len / 2.0))
            i = j
        else:
            i += 1
    return np.column_stack([runs, edges, alpha_heur, frac_through,
                             np.log1p(runs), np.log1p(edges)])


def trend_slope_multiscale_features(interp_values, scaled, windows=(12, 24, 48, 96)):
    """Pente de régression linéaire locale (moindres carrés glissants) à
    plusieurs échelles, sur (a) la courbe interpolée elle-même et (b) le
    consensus des voisins -- généralise local_trend_feature (fenêtre unique
    72, boucle Python par position) à plusieurs échelles via un calcul
    vectorisé pandas (rolling cov/var), motivé par la même observation EDA
    (dérive de niveau significative sur une partie des courbes échantillon-
    nées) mais sans imposer une seule granularité au modèle."""
    t = pd.Series(np.arange(len(interp_values), dtype=float))

    def rolling_slope(values, window):
        sv = pd.Series(np.asarray(values, dtype=float))
        cov = sv.rolling(window, center=True, min_periods=3).cov(t)
        var = t.rolling(window, center=True, min_periods=3).var()
        return (cov / var.replace(0, np.nan)).fillna(0.0).to_numpy()

    obs = np.asarray(interp_values, dtype=float)
    center = np.median(obs)
    scale = max(1.4826 * np.median(np.abs(obs - center)), 1.0)
    consensus = scaled.mean(axis=1)
    cols = []
    for w in windows:
        cols.append(rolling_slope(interp_values, w) / scale)
        cols.append(rolling_slope(consensus, w))
    return np.column_stack(cols)


def difference_features(interp_values, day_lag=48):
    """Dérivées discrètes (différence 1er ordre, 2e ordre) et écart au même
    horaire hier/demain sur la courbe interpolée -- complète les copies
    périodiques (valeurs ABSOLUES à ±48h) par leur VARIATION plutôt que
    leur niveau, information distincte pour un modèle en arbres (un même
    niveau peut correspondre à une pente montante ou descendante)."""
    v = np.asarray(interp_values, dtype=float)
    n = len(v)
    diff1 = np.zeros(n); diff1[1:] = v[1:] - v[:-1]
    diff2 = np.zeros(n); diff2[1:-1] = v[2:] - 2 * v[1:-1] + v[:-2]
    diff_prevday = np.full(n, np.nan)
    diff_prevday[day_lag:] = v[day_lag:] - v[:-day_lag]
    diff_nextday = np.full(n, np.nan)
    diff_nextday[:-day_lag] = v[:-day_lag] - v[day_lag:]
    center = np.median(v)
    scale = max(1.4826 * np.median(np.abs(v - center)), 1.0)
    return np.column_stack([diff1, diff2, diff_prevday, diff_nextday]) / scale


def time_bucket_features(index):
    """Créneaux horaires génériques (nuit/matin/après-midi/soir) encodés en
    one-hot -- redondant en information avec heure sin/cos pour un modèle
    linéaire, mais donne des SEUILS directement exploitables par les
    splits d'un modèle en arbres (un arbre doit sinon recombiner sin/cos
    sur plusieurs niveaux de profondeur pour retrouver la même frontière)."""
    dt_index = pd.to_datetime(index)
    hour = dt_index.hour.values
    night = ((hour >= 0) & (hour < 6)).astype(float)
    morning = ((hour >= 6) & (hour < 12)).astype(float)
    afternoon = ((hour >= 12) & (hour < 18)).astype(float)
    evening = ((hour >= 18) & (hour < 24)).astype(float)
    return np.column_stack([night, morning, afternoon, evening])


def interpolation_method_features(s, methods=("linear", "pchip", "quadratic")):
    """Courbe complète (toutes positions, observées ET manquantes) reconstruite
    par plusieurs méthodes d'interpolation intra-courbe, une colonne par
    méthode -- données comme features COMPLEMENTAIRES au régresseur, pas en
    remplacement de l'interpolation linéaire du blend alpha (celui-ci reste
    inchangé, cf. predict_column_v14). Chaque méthode encode une hypothèse
    de forme différente entre points connus (linéaire = segment droit,
    PCHIP = monotone sans oscillation, quadratique = courbure locale) ;
    laisse TabPFN arbitrer/pondérer plutôt que d'imposer un choix unique
    dans le blend. `pchip_interp_feature` (déjà dans feature_engineering_
    bench.py) avait été testé seul sur le pipeline V10 (Ridge, pas TabPFN) :
    108,49 W sur 5x100, rejeté (cf. benchmark.md §7) -- jamais revalidé sur
    le pipeline V14/TabPFN actuel, ni combiné à d'autres méthodes.

    Garde-fous : repli sur linéaire si <3 points observés (spline/PCHIP mal
    posés), extrapolate=False + repli linéaire aux bords hors plage, clip
    aux bornes [min, max] observées -- même piège que l'échec "PCHIP en
    remplacement du blend" de ce dépôt (1808 W sur 10x10, cf. benchmark.md
    §7 : PchipInterpolator(extrapolate=True) sans clip peut exploser sur les
    bords de trou). Normalisées médiane/MAD comme les autres blocs de
    `extra` (periodic_copy_features)."""
    values = np.asarray(s.values, dtype=float)
    n = len(values)
    observed = ~np.isnan(values)
    obs_idx = np.where(observed)[0]
    if len(obs_idx) < 2:
        flat = np.full(n, float(np.nanmedian(values)) if observed.any() else 0.0)
        return np.tile(flat.reshape(-1, 1), (1, len(methods)))
    obs_val = values[obs_idx]
    v_min, v_max = obs_val.min(), obs_val.max()
    center = np.median(obs_val)
    scale = max(1.4826 * np.median(np.abs(obs_val - center)),
                0.02 * (abs(center) + 1.0), 1.0)
    linear_fallback = None

    def _linear():
        nonlocal linear_fallback
        if linear_fallback is None:
            linear_fallback = s.interpolate(method="linear", limit_direction="both").to_numpy()
        return linear_fallback

    cols = []
    for method in methods:
        if method == "linear":
            filled = _linear().copy()
        elif method in ("pchip", "quadratic") and len(obs_idx) < 3:
            filled = _linear().copy()
        elif method == "pchip":
            from scipy.interpolate import PchipInterpolator
            interp = PchipInterpolator(obs_idx, obs_val, extrapolate=False)
            filled = interp(np.arange(n))
            oor = np.isnan(filled)
            if oor.any():
                filled[oor] = _linear()[oor]
        elif method == "quadratic":
            try:
                filled = s.interpolate(method="quadratic", limit_direction="both").to_numpy()
            except Exception:
                filled = _linear().copy()
            oor = np.isnan(filled)
            if oor.any():
                filled = np.where(oor, _linear(), filled)
        else:
            raise ValueError(f"méthode d'interpolation inconnue: {method}")
        filled = np.clip(filled, v_min, v_max)
        cols.append((filled - center) / scale)
    return np.column_stack(cols)


def spearman_powertransform_neighbor_features(pool_vals, obs_idx, target_obs,
                                               k=20, k_min=5, corr_threshold=0.1):
    """Second bloc de voisins, COMPLEMENTAIRE (pas en remplacement) à
    `weighted_knn_pred_tunable`/`adaptive_top_k_corr_euclid_tunable` (qui
    sélectionnent par Pearson+euclidien) : sélection par corrélation de
    SPEARMAN (sur les rangs, robuste aux pics/valeurs aberrantes de
    consommation qui peuvent fausser Pearson -- jamais testé dans ce dépôt,
    cf. discussion sur la sélection de voisins), puis les courbes retenues
    sont PowerTransformées (Yeo-Johnson) avant d'être données en feature.

    Point mathématique important (même piège déjà documenté dans ce dépôt
    pour "z-score + euclidien == Pearson", cf. adaptive_top_k_corr_euclid) :
    PowerTransform est une fonction MONOTONE, et Spearman ne dépend QUE du
    rang des valeurs -- donc Spearman(PowerTransform(x), PowerTransform(y))
    == Spearman(x, y) EXACTEMENT, quelle que soit la courbe. Power-
    transformer AVANT de calculer Spearman ne changerait donc RIEN au
    classement des voisins retenus. Le PowerTransform est appliqué ici
    seulement sur les VALEURS des voisins déjà choisis par Spearman (pas
    sur le critère de sélection lui-même) : il donne au modèle une
    représentation différente (distribution reshapée vers une forme plus
    gaussienne, cf. discussion PowerTransformer sur la cible) des mêmes
    courbes -- complémentaire au bloc de voisins Pearson+euclidien/robuste-
    médiane-MAD existant (`scaled`), pas redondant avec lui.

    pool_vals : (T, n_pool) déjà rempli (ffill/bfill), comme dans
    weighted_knn_pred_tunable -- toutes les courbes du pool sont complètes,
    seul le rang de la courbe CIBLE (via obs_idx/target_obs) est partiel.
    """
    pool_obs = pool_vals[obs_idx, :]
    target_rank = rankdata(target_obs)
    pool_rank = rankdata(pool_obs, axis=0)

    t_c = target_rank - target_rank.mean()
    d_c = pool_rank - pool_rank.mean(axis=0, keepdims=True)
    num = d_c.T @ t_c
    denom = np.sqrt((d_c ** 2).sum(axis=0) * (t_c ** 2).sum()) + 1e-9
    spearman_corr = num / denom

    order = np.argsort(-spearman_corr)
    above = order[spearman_corr[order] >= corr_threshold]
    chosen = above[:k] if len(above) >= k_min else order[:k_min]

    pool_sel = pool_vals[:, chosen]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pt_vals = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(pool_sel)
        if not np.all(np.isfinite(pt_vals)):
            raise ValueError("PowerTransformer non fini")
    except Exception:
        # Repli robuste médiane/MAD (même convention que scale_neighbors_tunable)
        # si Yeo-Johnson échoue/diverge (ex. colonne quasi constante).
        center = np.median(pool_sel, axis=0, keepdims=True)
        scale = np.maximum(1.4826 * np.median(np.abs(pool_sel - center), axis=0, keepdims=True), 1.0)
        pt_vals = (pool_sel - center) / scale

    return pt_vals, chosen, spearman_corr[chosen]


def collective_consumption_index(pool_vals, sample_size=3000, random_state=42):
    """Indice de consommation collective : médiane robuste (centrée-réduite)
    d'un large échantillon de TOUTES les courbes du pool à chaque instant --
    signal agrégé (météo/type de jour implicites) distinct de la sélection
    de voisins corrélés à UNE courbe cible ; calculé UNE FOIS par split,
    partagé par toutes les colonnes du batch."""
    rng = np.random.RandomState(random_state)
    n_cols = pool_vals.shape[1]
    idx = rng.choice(n_cols, size=min(sample_size, n_cols), replace=False)
    sample = pool_vals[:, idx]
    # normalise chaque courbe par sa propre échelle avant d'agréger, sinon
    # quelques courbes à forte amplitude dominent la médiane
    center = np.median(sample, axis=0, keepdims=True)
    scale = np.maximum(1.4826 * np.median(np.abs(sample - center), axis=0, keepdims=True), 1.0)
    normed = (sample - center) / scale
    idx_med = np.median(normed, axis=1)
    idx_center = np.median(idx_med)
    idx_scale = max(1.4826 * np.median(np.abs(idx_med - idx_center)), 1.0)
    return (idx_med - idx_center) / idx_scale


def compute_svd_factors(pool_vals, n_components=10, random_state=RANDOM_STATE):
    """Facteurs temporels latents du pool (une valeur par instant, pas par
    courbe) via SVD tronquée -- U*sigma des n_components premières
    composantes de la matrice pool (instants x courbes) centrée par colonne.
    Repris de compute_svd_factors dans
    claude/colab_generate_submission_tabpfn_v14_full.py (mêmes maths), avec
    n_components par défaut recalé sur le rang optimal trouvé empiriquement
    dans le notebook externe dempxl/ChallengeData-ENEDIS (section 3.4,
    imputation SVD directe sur 200 courbes : rang 10 -> MAE 111,58 W,
    meilleur que rang 20 (116,08), 50 (121,14) et 100 (127,78) -- au-delà de
    ~10 composantes, la SVD sur-apprend/reconstruit du bruit sur ce jeu de
    données plutôt que du signal utile) au lieu du N_SVD=50 non confirmé du
    script colab (validé sur 1 seul repeat, pas le protocole 10x10 standard
    de ce dépôt). Usage ici DIFFERENT du notebook : pas une méthode
    d'imputation directe, mais des features supplémentaires concaténées à
    la matrice passée à TabPFN (comme les autres blocs de `extra`)."""
    c = pool_vals - pool_vals.mean(axis=0, keepdims=True)
    U, s, _ = randomized_svd(c, n_components=n_components, random_state=random_state)
    return U * s


def lagged_eof_features(pool_vals, lags=(0, 48, 96, 144), n_components=8, random_state=RANDOM_STATE):
    """ST-PCA / EOF (Empirical Orthogonal Functions) via matrice de décalage
    temporel -- LE standard scientifique/industriel de réduction de
    dimension spatio-temporelle linéaire sans réseau de neurones (source :
    demande explicite de l'utilisateur, comparaison de 4 transformateurs
    classiques : ST-PCA/EOF, Tucker/HOSVD, Spatial PCA/Moran, Kernel PCA
    spatio-temporel). Les 3 autres options nécessitent une structure que ce
    jeu de données n'a pas : Tucker demande un vrai tenseur 3D (courbes ×
    temps × VARIABLES, on a une seule variable par courbe), Spatial PCA/
    Moran et le noyau spatio-temporel demandent des coordonnées géo-
    graphiques entre courbes (identifiants aléatoires ici, aucune notion
    d'espace) -- seule l'EOF s'applique.

    Différence avec compute_svd_factors (déjà dans ce pipeline, option
    n_svd) : celle-ci fait une SVD UNIQUE sur le pool brut (covariance
    INSTANTANEE entre courbes). Ici, une SVD tronquée SEPAREE est calculée
    à CHAQUE décalage temporel du pool (lag=0 = brut, +48/+96/+144h =
    pool décalé de ce nombre de pas), puis les facteurs (U*sigma) de
    chaque lag sont concaténés -- capture la covariance inter-courbes
    DECALEE dans le temps (une courbe peut ressembler à une autre avec un
    décalage de phase), pas seulement instantanée. Approximation d'une
    vraie "Extended/Multichannel EOF" (SVD jointe sur une matrice
    (T, n_courbes*n_lags) concaténée) par N SVD séquentielles sur des
    matrices de la même taille que le pool -- évite l'explosion du nombre
    de colonnes (n_courbes*n_lags atteindrait des dizaines/centaines de
    milliers de colonnes sur le pool complet) pour un coût comparable à
    len(lags) appels de compute_svd_factors.

    Décalage borné en amont (repli sur la 1ère valeur valide, cohérent
    avec le ffill/bfill déjà appliqué au pool)."""
    blocks = []
    for lag in lags:
        if lag == 0:
            shifted = pool_vals
        else:
            shifted = pd.DataFrame(pool_vals).shift(lag).bfill().to_numpy()
        c = shifted - shifted.mean(axis=0, keepdims=True)
        U, s, _ = randomized_svd(c, n_components=n_components, random_state=random_state)
        blocks.append(U * s)
    return np.hstack(blocks)


def daily_profile(values, index, n_buckets=48):
    """Profil journalier médian (n_buckets valeurs, une par demi-heure),
    normalisé par la propre échelle de la courbe -- représentation compacte
    de la FORME typique d'une courbe plutôt que de ses 1057 points bruts.
    Un vrai DTW sur 20000 x 1057 points étant hors de portée en temps
    raisonnable (des milliards d'opérations), ce profil sert de proxy léger
    pour re-clusteriser par forme journalière plutôt que par valeurs brutes
    z-scorées (qui restent sensibles au bruit/aux trous position par
    position)."""
    dt_index = pd.to_datetime(index)
    minute_of_day = dt_index.hour.values * 60 + dt_index.minute.values
    bucket_size = 1440 // n_buckets
    bucket = minute_of_day // bucket_size
    observed = ~np.isnan(values)
    df = pd.DataFrame({"bucket": bucket, "val": values})
    med_by_bucket = df.loc[observed].groupby("bucket")["val"].median()
    global_med = float(np.nanmedian(values)) if observed.any() else 0.0
    profile = np.full(n_buckets, global_med, dtype=float)
    for b, v in med_by_bucket.items():
        if 0 <= b < n_buckets:
            profile[b] = v
    center = np.median(profile)
    scale = max(1.4826 * np.median(np.abs(profile - center)), 0.02 * (abs(center) + 1.0), 1.0)
    return (profile - center) / scale


def daily_profiles_batch(values_matrix, index, n_buckets=48):
    """Version vectorisée de daily_profile pour une matrice (T, n_curves) --
    une seule boucle sur les n_buckets créneaux (pas sur les courbes), tout
    le reste opère sur toutes les courbes à la fois. La boucle par courbe
    (20000 x pandas groupby) prenait plus de 2 minutes ; cette version
    vectorisée ramène le clustering par profil à un coût comparable au
    clustering existant sur courbe brute."""
    dt_index = pd.to_datetime(index)
    minute_of_day = dt_index.hour.values * 60 + dt_index.minute.values
    bucket_size = 1440 // n_buckets
    bucket = minute_of_day // bucket_size
    n_curves = values_matrix.shape[1]
    profiles = np.full((n_curves, n_buckets), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_med = np.nanmedian(values_matrix, axis=0)
        for b in range(n_buckets):
            mask = bucket == b
            if not mask.any():
                profiles[:, b] = global_med
                continue
            med = np.nanmedian(values_matrix[mask, :], axis=0)
            profiles[:, b] = np.where(np.isnan(med), global_med, med)
    center = np.median(profiles, axis=1, keepdims=True)
    scale = np.maximum(1.4826 * np.median(np.abs(profiles - center), axis=1, keepdims=True),
                        np.maximum(0.02 * (np.abs(center) + 1.0), 1.0))
    return (profiles - center) / scale


_PROFILE_CLUSTER_CACHE = {}


def fit_shape_clusters_profile(donor_df, index, n_clusters=15, random_state=42):
    key = (tuple(donor_df.columns), n_clusters, random_state)
    if key in _PROFILE_CLUSTER_CACHE:
        return _PROFILE_CLUSTER_CACHE[key]
    from sklearn.cluster import KMeans
    filled = donor_df.ffill().bfill().to_numpy()
    profiles = daily_profiles_batch(filled, index)
    km = KMeans(n_clusters=n_clusters, n_init=4, max_iter=100, random_state=random_state)
    labels = km.fit_predict(profiles)
    labels = pd.Series(labels, index=donor_df.columns)
    _PROFILE_CLUSTER_CACHE[key] = (km, labels)
    return km, labels


def cluster_of_profile(values, index, km):
    return int(km.predict(daily_profile(values, index).reshape(1, -1))[0])


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


def weighted_neighbors_native_nan(pool_vals, obs_idx, target_obs, k,
                                  k_min=5, corr_threshold=0.15,
                                  euclid_weight=0.3, ratio_clip=(0.1, 10.0),
                                  min_common=15):
    """Sélection et mise à l'échelle de voisins sans aucune imputation.

    Chaque donneur est comparé à la cible uniquement aux timestamps où les
    deux sont observés. Les NaN du donneur restent des NaN dans les features
    retournées afin que le régresseur choisi les traite lui-même.
    """
    pool_obs = pool_vals[obs_idx, :]
    n_pool = pool_obs.shape[1]
    corr = np.full(n_pool, -np.inf)
    euclid_score = np.zeros(n_pool)
    target_obs = np.asarray(target_obs, dtype=np.float64)
    target_scale = np.nanstd(target_obs) + 1e-9
    for j in range(n_pool):
        donor = pool_obs[:, j]
        valid = np.isfinite(donor) & np.isfinite(target_obs)
        if valid.sum() < min_common:
            continue
        target_common, donor_common = target_obs[valid], donor[valid]
        target_centered = target_common - target_common.mean()
        donor_centered = donor_common - donor_common.mean()
        denom = np.sqrt(np.sum(target_centered ** 2) * np.sum(donor_centered ** 2))
        if denom > 1e-9:
            corr[j] = float(np.dot(target_centered, donor_centered) / denom)
        distance = np.sqrt(np.mean((donor_centered - target_centered) ** 2)) / target_scale
        euclid_score[j] = 1.0 / (1.0 + distance)

    usable = np.flatnonzero(np.isfinite(corr))
    if len(usable) < k_min:
        raise ValueError(
            f"Seulement {len(usable)} donneurs ont au moins {min_common} points communs")
    combined = (1 - euclid_weight) * corr[usable] + euclid_weight * euclid_score[usable]
    order = usable[np.argsort(-combined)]
    above = order[corr[order] >= corr_threshold]
    chosen = above[:k] if len(above) >= k_min else order[:min(k_min, len(order))]

    pool_sel = pool_vals[:, chosen]
    selected_obs = pool_sel[obs_idx, :]
    center = np.nanmedian(selected_obs, axis=0)
    floor = 0.02 * (np.abs(center) + 1.0)
    spread = np.maximum(
        1.4826 * np.nanmedian(np.abs(selected_obs - center), axis=0), floor)
    target_center = np.nanmedian(target_obs)
    target_spread = max(
        1.4826 * np.nanmedian(np.abs(target_obs - target_center)),
        0.02 * (abs(target_center) + 1.0))
    ratio = np.clip(target_spread / spread, ratio_clip[0], ratio_clip[1])
    scaled = (pool_sel - center) * ratio + target_center
    return chosen, scaled


def same_curve_seasonality_native_nan(values, index):
    """Médiane saisonnière observée, sans fallback pour les buckets absents."""
    dt_index = pd.to_datetime(index)
    bucket = dt_index.hour.values * 60 + dt_index.minute.values
    observed = np.isfinite(values)
    medians = pd.DataFrame({"bucket": bucket[observed], "value": np.asarray(values)[observed]}).groupby(
        "bucket")["value"].median()
    return pd.Series(bucket).map(medians).to_numpy(dtype=float).reshape(-1, 1)


def neighbor_rolling_native_nan(scaled, window=5):
    """Agrégats glissants qui conservent NaN lorsqu'aucune donnée n'existe."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        consensus = np.nanmean(scaled, axis=1)
    series = pd.Series(consensus)
    mean = series.rolling(window, center=True, min_periods=1).mean().to_numpy()
    std = series.rolling(window, center=True, min_periods=2).std().to_numpy()
    return np.column_stack([mean, std])


DEFAULT_OPTS = dict(
    k=K_DEFAULT, n_clusters=15, corr_threshold=0.15, euclid_weight=0.3, k_min=5,
    extra_lags=(), neighbor_rolling=False, neighbor_consensus=False,
    pmm=False, pmm_blend=0.3, periodic_alpha_strength=0.5,
    volatility_alpha=False, vol_weight=0.5, vol_cap=0.6,
    regressor_cls=MedianTabPFN, regressor_kwargs=None,
    ratio_clip=(0.1, 10.0), alpha_scale=1.0,
    weekday_seasonality=False, daytype_periodic=False, daytype_flag_feat=False,
    local_trend_feat=False, local_trend_window=72,
    profile_clustering=False,
    neighbor_mad_feat=False, neighbor_interval_feat=False,
    peak_offpeak_feat=False, collective_index_feat=False,
    distribution50=False, v15_groups=(), iterative_passes=1,
    learned_blend_model=None,
    n_svd=0,  # nb de facteurs SVD du pool (compute_svd_factors) concaténés
              # à `extra` ; 0 = désactivé (comportement inchangé par défaut)
    clip_nonneg=False,  # clippe la prediction finale a >=0 (coherent avec le
                         # clip(lower=0) deja applique par TOUS les scripts
                         # generate_submission_*.py en aval, mais absent du
                         # harnais d'evaluation -- cf. benchmark.md, des
                         # predictions negatives (jusqu'a -9W) apparaissent
                         # sur ~3% des colonnes reelles). False par defaut
                         # pour ne pas changer le score deja documente
                         # (63.24 +/- 30.46 W) ; ne peut mathematiquement que
                         # reduire ou laisser inchangee la MAE (la vraie
                         # conso est toujours >=0), jamais la degrader.
    interp_features=(),  # tuple de methodes d'interpolation intra-courbe
                         # (parmi "linear"/"pchip"/"quadratic") ajoutees comme
                         # features via interpolation_method_features ; ()
                         # = desactive (comportement inchange par defaut)
    n_spearman=0,  # k pour le second bloc de voisins Spearman+PowerTransform
                   # (spearman_powertransform_neighbor_features), COMPLEMENTAIRE
                   # au bloc principal Pearson+euclidien (`scaled`) ; 0 = desactive
    periodic_availability_feat=True,  # 10 flags 0/1 de disponibilite des copies
                   # periodiques dans `periodic_copy_features` (par lag, back/forward).
                   # True = comportement inchange (defaut historique). Screening local
                   # (LightGBM proxy, 150 colonnes reelles) : correlation quasi nulle
                   # avec la cible (tous |r|<0.08) et importance par permutation
                   # negative ou nulle sur la quasi-totalite des colonnes testees --
                   # candidat a suppression, mais NON confirme sur TabPFN reel
                   # (cf. tabpfn_v14_noavail_confirm.py).
    lagged_eof_lags=(),  # tuple de decalages (en pas de 30min) pour
                   # lagged_eof_features (ex. (0,48,96,144)) ; () = desactive.
                   # ST-PCA/EOF via matrice de decalage temporel, calcule UNE
                   # FOIS par split (comme n_svd), pas par colonne.
    native_nan=False,  # transmet les NaN au régresseur : aucun fill du pool,
                       # aucun clustering/interpolation/blend/fallback manuel.
)


def distribution50_features(pool_vals, cluster_vals, scaled, proxy):
    """Exactly 50 finite, leakage-free distribution descriptors per timestamp.

    ``pool_vals`` contains only donors/development curves authorized by the
    outer split; ``cluster_vals`` is the shape-cluster subset and ``scaled``
    the selected neighbours.  ``proxy`` is interpolation from observed target
    values, never the hidden truth.
    """
    def stats16(a):
        q = np.nanpercentile(a, [1, 5, 10, 25, 50, 75, 90, 95, 99], axis=1).T
        mean = np.nanmean(a, axis=1); std = np.nanstd(a, axis=1)
        mn = np.nanmin(a, axis=1); mx = np.nanmax(a, axis=1)
        med = q[:, 4]
        mad = np.nanmedian(np.abs(a - med[:, None]), axis=1)
        centered = a - mean[:, None]
        skew = np.nanmean(centered ** 3, axis=1) / (std ** 3 + 1e-9)
        return np.column_stack([mean, std, mn, mx, q, q[:, 5]-q[:, 3], mad, skew])

    global16 = stats16(pool_vals)
    # Twelve cluster descriptors.
    cq = np.nanpercentile(cluster_vals, [10, 25, 50, 75, 90], axis=1).T
    cmed = cq[:, 2]
    cmad = np.nanmedian(np.abs(cluster_vals-cmed[:, None]), axis=1)
    sorted_c = np.sort(cluster_vals, axis=1)
    trim = max(1, int(round(sorted_c.shape[1] * .1))) if sorted_c.shape[1] > 2 else 0
    trimmed = np.nanmean(sorted_c[:, trim:-trim], axis=1) if trim else np.nanmean(sorted_c, axis=1)
    cluster12 = np.column_stack([np.nanmean(cluster_vals, axis=1), np.nanstd(cluster_vals, axis=1),
        np.nanmin(cluster_vals, axis=1), np.nanmax(cluster_vals, axis=1), cq,
        cq[:, 3]-cq[:, 1], cmad, trimmed])
    # Ten selected-neighbour descriptors.
    nq = np.nanpercentile(scaled, [10, 25, 50, 75, 90], axis=1).T
    nmed = nq[:, 2]; nmad = np.nanmedian(np.abs(scaled-nmed[:, None]), axis=1)
    sign_agree = np.abs(np.nanmean(np.sign(np.diff(scaled, axis=0, prepend=scaled[:1])), axis=1))
    neigh10 = np.column_stack([np.nanmean(scaled, axis=1), np.nanstd(scaled, axis=1), nq,
                               nq[:, 3]-nq[:, 1], nmad, sign_agree])
    # Six global distribution changes: 30 minutes and 24 hours.
    gbase = global16[:, [0, 5, 1]]
    changes6 = np.column_stack([gbase-np.roll(gbase, 1, axis=0),
                                gbase-np.roll(gbase, 48, axis=0)])
    changes6[:48, 3:] = 0.; changes6[0, :3] = 0.
    # Six target-relative descriptors: percentile rank and robust z-score in
    # the global, cluster and selected-neighbour populations.
    rel = []
    for a in (pool_vals, cluster_vals, scaled):
        med = np.nanmedian(a, axis=1); mad = np.nanmedian(np.abs(a-med[:, None]), axis=1)
        rel.extend([np.nanmean(a <= proxy[:, None], axis=1), (proxy-med)/(1.4826*mad+1e-9)])
    out = np.hstack([global16, cluster12, neigh10, changes6, np.column_stack(rel)])
    assert out.shape[1] == 50
    return np.nan_to_num(out, nan=0., posinf=0., neginf=0.)


def build_v14_features(col, s, pool_vals, km, cluster_col_idx, tfeat, index, opts,
                       collective_idx=None, feature_values=None, svd_feat=None,
                       lagged_eof_feat=None):
    """Portion "features" du pipeline V14, extraite de predict_column_v14 pour
    être réutilisable telle quelle par un régresseur autre que TabPFN (ex.
    recherche d'hyperparamètres GBM, cf. gbm_optuna_vs_tabpfn.py) : construit
    exactement les mêmes (obs_idx, miss_idx, target_obs, scaled, extra) que
    ceux passés à feb.local_regression_pred dans predict_column_v14, sans
    fitter/predire quoi que ce soit. Comportement de predict_column_v14
    inchangé (simple découpage, pas de logique modifiée)."""
    o = {**DEFAULT_OPTS, **opts}
    observed = s.notna().to_numpy(); missing = ~observed
    if not missing.any():
        return None
    obs_idx, miss_idx = np.flatnonzero(observed), np.flatnonzero(missing)
    target_obs = s.to_numpy()[obs_idx]
    feat_values = (s.to_numpy() if feature_values is None
                   else np.asarray(feature_values, dtype=np.float64))
    interp_full = None if o["native_nan"] else fill_nan_with_interpolation(s)
    if o["native_nan"]:
        sub_vals = pool_vals
        _, scaled = weighted_neighbors_native_nan(
            sub_vals, obs_idx, target_obs, o["k"], k_min=o["k_min"],
            corr_threshold=o["corr_threshold"], euclid_weight=o["euclid_weight"],
            ratio_clip=o["ratio_clip"])
        knn_pred = None
    elif o["profile_clustering"]:
        cluster = cluster_of_profile(interp_full.to_numpy(), index, km)
    else:
        cluster = feb.cluster_of(interp_full, km)
    if not o["native_nan"]:
        same_idx = cluster_col_idx.get(cluster)
        sub_vals = (pool_vals[:, same_idx]
                    if same_idx is not None and len(same_idx) >= max(2 * o["k"], 30)
                    else pool_vals)
        knn_pred, _, scaled = weighted_knn_pred_tunable(
            sub_vals, obs_idx, miss_idx, target_obs, o["k"],
            k_min=o["k_min"], corr_threshold=o["corr_threshold"],
            euclid_weight=o["euclid_weight"], ratio_clip=o["ratio_clip"])

    lags = (48, 96, 144, 336) + tuple(o["extra_lags"])
    if o["daytype_periodic"]:
        periodic = periodic_copy_features_daytype(
            feat_values, index, lags=lags, include_summary=True, normalize=True,
            include_availability=o["periodic_availability_feat"])
    else:
        periodic = feb.periodic_copy_features(
            feat_values, lags=lags, include_summary=True, normalize=True,
            include_availability=o["periodic_availability_feat"])
    if o["weekday_seasonality"]:
        seasonal = same_curve_seasonality_weekday_feature(feat_values, index)
    else:
        seasonal = (same_curve_seasonality_native_nan(feat_values, index)
                    if o["native_nan"] else feb.same_curve_seasonality_feature(feat_values, index))
    extra = np.hstack([tfeat, seasonal, periodic])
    if o["daytype_flag_feat"]:
        extra = np.hstack([extra, daytype_match_flag_features(index, lags=lags)])
    if o["local_trend_feat"]:
        extra = np.hstack([extra, local_trend_feature(feat_values, window=o["local_trend_window"])])
    if o["neighbor_rolling"]:
        extra = np.hstack([extra, (neighbor_rolling_native_nan(scaled)
                                  if o["native_nan"] else feb.neighbor_rolling_features(scaled))])
    if o["neighbor_consensus"]:
        extra = np.hstack([extra, feb.neighbor_consensus_std_feature(scaled)])
    if o["neighbor_mad_feat"]:
        extra = np.hstack([extra, neighbor_mad_feature(scaled)])
    if o["neighbor_interval_feat"]:
        extra = np.hstack([extra, neighbor_interval_width_feature(scaled)])
    if o["peak_offpeak_feat"]:
        extra = np.hstack([extra, peak_offpeak_flag(index)])
    if o["collective_index_feat"] and collective_idx is not None:
        extra = np.hstack([extra, collective_idx.reshape(-1, 1)])
    if o["n_svd"] and svd_feat is not None:
        extra = np.hstack([extra, svd_feat])
    if o["lagged_eof_lags"] and lagged_eof_feat is not None:
        extra = np.hstack([extra, lagged_eof_feat])
    if o["n_spearman"]:
        spearman_feat, _, _ = spearman_powertransform_neighbor_features(
            sub_vals, obs_idx, target_obs, k=o["n_spearman"])
        extra = np.hstack([extra, spearman_feat])
        # Mêmes features dérivées que celles déjà appliquées au bloc de
        # voisins principal (Pearson+euclidien, `scaled`) ci-dessus,
        # gouvernées par les MÊMES options -- pas dupliquées en dur, pour
        # que toute config qui active/désactive l'une de ces options
        # l'applique de façon cohérente aux deux blocs de voisins.
        if o["neighbor_rolling"]:
            extra = np.hstack([extra, feb.neighbor_rolling_features(spearman_feat)])
        if o["neighbor_consensus"]:
            extra = np.hstack([extra, feb.neighbor_consensus_std_feature(spearman_feat)])
        if o["neighbor_mad_feat"]:
            extra = np.hstack([extra, neighbor_mad_feature(spearman_feat)])
        if o["neighbor_interval_feat"]:
            extra = np.hstack([extra, neighbor_interval_width_feature(spearman_feat)])
    if o["interp_features"]:
        extra = np.hstack([extra, interpolation_method_features(s, methods=o["interp_features"])])
    if o["v15_groups"]:
        import tabpfn_v15_features as v15
        extra = np.hstack([extra, v15.build_v15_features(
            feat_values, index, scaled, missing, groups=o["v15_groups"], pool=sub_vals)])
    if o["distribution50"]:
        extra = np.hstack([extra, distribution50_features(
            pool_vals, sub_vals, scaled, np.where(np.isfinite(feat_values),
                                                   feat_values, interp_full.to_numpy()))])

    return obs_idx, miss_idx, target_obs, interp_full, knn_pred, scaled, extra, missing, lags


def predict_column_v14(col, s, pool_vals, km, cluster_col_idx, tfeat, index, opts,
                        collective_idx=None, svd_feat=None, lagged_eof_feat=None):
    """Coeur du pipeline V14 : renvoie (col, série complétée, erreur/repli
    éventuel). Utilisé à la fois pour l'évaluation (complete_column_v14, qui
    calcule la MAE contre y_true) et pour la génération de soumission
    (generate_submission_tabpfn_v14.py, qui n'a pas de vérité terrain)."""
    o = {**DEFAULT_OPTS, **opts}
    error = None
    feature_values = s.to_numpy().copy()
    pred = None
    for pass_idx in range(max(1, int(o["iterative_passes"]))):
        built = build_v14_features(col, s, pool_vals, km, cluster_col_idx, tfeat,
                                   index, opts, collective_idx, feature_values, svd_feat,
                                   lagged_eof_feat)
        if built is None:
            return col, s.copy(), None
        obs_idx, miss_idx, target_obs, interp_full, knn_pred, scaled, extra, missing, lags = built
        pred_neighbors = None
        if len(obs_idx) >= 15:
            for attempt in range(MAX_RETRIES):
                try:
                    make_reg = (lambda: o["regressor_cls"](**(o["regressor_kwargs"] or {})))
                    pred_neighbors = feb.local_regression_pred(
                        make_reg, scaled, obs_idx, miss_idx, target_obs, extra)
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    time.sleep(2 * (attempt + 1))
        if o["native_nan"]:
            if pred_neighbors is None:
                raise RuntimeError(
                    f"AutoGluon native_nan a échoué pour {col}: {error or 'prédiction absente'}")
            pred = pred_neighbors
            if pass_idx + 1 < int(o["iterative_passes"]):
                feature_values = feature_values.copy()
                feature_values[miss_idx] = pred
            continue
        if pred_neighbors is None:
            pred_neighbors = knn_pred
        if o["pmm"]:
            pred_neighbors = feb.pmm_snap(pred_neighbors, scaled[miss_idx, :], blend=o["pmm_blend"])
        runs = feb.run_lengths_for(missing); edges = feb.edge_distance_for(missing)
        alpha = np.array([feb.continuous_alpha(runs[i], edges[i], True) for i in miss_idx])
        if o["daytype_periodic"]:
            p_agree = periodic_agreement_daytype(feature_values, index, lags=lags)[miss_idx]
        else:
            p_agree = feb.periodic_agreement(feature_values)[miss_idx]
        alpha *= 1.0 - o["periodic_alpha_strength"] * p_agree
        if o["volatility_alpha"]:
            vol = feb.curve_volatility(s.to_numpy())
            alpha *= feb.volatility_alpha_mult(vol, vol_weight=o["vol_weight"], cap=o["vol_cap"])
        alpha = np.clip(alpha * o["alpha_scale"], 0.0, 0.95)
        if o["learned_blend_model"] is not None:
            # Round 2026-08-17 : remplace le blend alpha "à la main" par un
            # meta-modele appris (cf. tabpfn_oof_alpha_fullscale.py) sur
            # (interp_pred, model_pred, meta-features) -> y_true. `alpha`
            # reste calcule ci-dessus (inchange) mais n'est plus utilise pour
            # la prediction finale ; uniquement le meta-modele fourni l'est.
            vol_lb = feb.curve_volatility(s.to_numpy())
            meta_lb = np.column_stack([
                np.minimum(runs[miss_idx], 50) / 50.0,
                np.minimum(edges[miss_idx], 20) / 20.0,
                p_agree,
                np.full(len(miss_idx), min(vol_lb, 1.0)),
            ])
            X_meta = np.column_stack([interp_full.to_numpy()[miss_idx], pred_neighbors, meta_lb])
            pred = o["learned_blend_model"].predict(X_meta)
        else:
            pred = alpha * interp_full.to_numpy()[miss_idx] + (1 - alpha) * pred_neighbors
        if pass_idx + 1 < int(o["iterative_passes"]):
            feature_values = feature_values.copy(); feature_values[miss_idx] = pred

    if o["clip_nonneg"]:
        pred = np.clip(pred, 0.0, None)

    filled = s.copy()
    filled.iloc[miss_idx] = pred
    return col, filled, error


def complete_column_v14(col, s, y_true_col, pool_vals, km, cluster_col_idx, tfeat, index, opts,
                         collective_idx=None, svd_feat=None, lagged_eof_feat=None):
    """Variante évaluation : calcule la MAE de predict_column_v14 contre
    y_true_col (utilisée par run_config / les criblages 3x10 et 10x10)."""
    observed = s.notna().to_numpy(); missing = ~observed
    if not missing.any():
        return col, None, None
    miss_idx = np.flatnonzero(missing)
    _, filled, error = predict_column_v14(col, s, pool_vals, km, cluster_col_idx, tfeat, index, opts,
                                           collective_idx, svd_feat, lagged_eof_feat)
    pred = filled.to_numpy()[miss_idx]
    y_true = y_true_col.to_numpy()[miss_idx]
    valid = ~np.isnan(y_true)
    abs_err = np.abs(pred[valid] - y_true[valid])
    return col, abs_err, error


def eval_split(X_train, y_train, donor_cols, dev_cols, val_cols, opts, n_clusters=15):
    donor_df = X_train[donor_cols]
    dev_true_df = y_train[dev_cols]
    val_holed_df = X_train[val_cols]

    pool = pd.concat([donor_df, dev_true_df], axis=1)
    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    if opts.get("profile_clustering"):
        km, donor_labels = fit_shape_clusters_profile(
            donor_df, X_train.index, n_clusters=n_clusters, random_state=RANDOM_STATE)
        dev_filled = dev_true_df.ffill().bfill().to_numpy()
        dev_profiles = (daily_profiles_batch(dev_filled, X_train.index)
                         if dev_filled.shape[1] else np.empty((0, 48)))
        dev_labels = pd.Series(km.predict(dev_profiles) if len(dev_true_df.columns) else [],
                                index=dev_true_df.columns)
    else:
        km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=n_clusters, random_state=RANDOM_STATE)
        dev_filled = dev_true_df.ffill().bfill()
        Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
        dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
    pool_labels = pd.concat([donor_labels, dev_labels])
    pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
    cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(val_holed_df.index, ("hour", "dow", "dom", "year"))
    collective_idx = (collective_consumption_index(pool_vals, random_state=RANDOM_STATE)
                       if opts.get("collective_index_feat") else None)
    svd_feat = (compute_svd_factors(pool_vals, opts["n_svd"], RANDOM_STATE)
                if opts.get("n_svd") else None)
    lagged_eof_feat = (lagged_eof_features(pool_vals, lags=opts["lagged_eof_lags"], random_state=RANDOM_STATE)
                        if opts.get("lagged_eof_lags") else None)

    all_errs = []
    fallbacks = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(complete_column_v14, c, val_holed_df[c], y_train[c],
                                    pool_vals, km, cluster_col_idx, tfeat,
                                    val_holed_df.index, opts, collective_idx, svd_feat,
                                    lagged_eof_feat) for c in val_cols]
        for fut in as_completed(futures):
            col, abs_err, error = fut.result()
            if abs_err is not None:
                all_errs.append(abs_err)
            if error:
                fallbacks += 1
    return float(np.concatenate(all_errs).mean()), fallbacks


def run_config(name, opts, X, y, donor_cols, n_repeats, n_val_sub, n_clusters=15):
    holed_cols = list(y.columns)
    scores = []
    for i in range(n_repeats):
        seed_i = RANDOM_STATE + i
        dev_cols, val_cols_full = train_test_split(holed_cols, test_size=.1, random_state=seed_i)
        rng = np.random.RandomState(seed_i)
        val_cols = list(rng.choice(val_cols_full, size=min(n_val_sub, len(val_cols_full)), replace=False))
        t0 = time.time()
        mae, fallbacks = eval_split(X, y, donor_cols, dev_cols, val_cols, opts, n_clusters=opts.get("n_clusters", 15))
        dt = time.time() - t0
        scores.append(mae)
        print(f"  [{name}] repeat {i+1}/{n_repeats}: MAE={mae:8.2f} W "
              f"({dt:.1f}s, fallbacks={fallbacks})", flush=True)
    scores = np.array(scores)
    print(f"  -> [{name}] Mean MAE over {n_repeats} repeats: {scores.mean():.2f} +/- {scores.std():.2f} W\n", flush=True)
    return scores
