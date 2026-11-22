"""Catalogue de ~55 features candidates, ajoutées UNE PAR UNE aux features
exactes de tabpfn_exact_tabfm_bench.py (= tabfm_lora_finetune.V14_FEATURE_OPTS,
identiques à generate_submission_tabpfn_v14.py), pour un criblage LightGBM
(lgbm_v14_feature_screen.py).

CONTRAINTE DE CONCEPTION IMPORTANTE (découverte en smoke-test avant le
criblage complet, pas une supposition) : le régresseur (feb.local_regression_
pred, appelé par predict_column_v14 / complete_column_augmented) entraîne UN
MODÈLE SÉPARÉ PAR COLONNE, uniquement sur les points OBSERVÉS de cette même
courbe. Une feature "scalaire par courbe entière" (ex. skewness globale,
autocorrélation globale, entropie spectrale globale...) est donc CONSTANTE
sur toutes les lignes d'entraînement d'une même colonne -> variance nulle
intra-colonne -> LightGBM ne peut jamais splitter dessus -> contribution
mathématiquement nulle, quelle que soit la pertinence de l'idée sous-jacente
(vérifié : un smoke test avec une telle feature donne un MAE identique au
bit près au baseline). TOUTES les features ci-dessous sont donc formulées en
version LOCALE/GLISSANTE (fenêtre de quelques dizaines à ~700 points) plutôt
qu'en statistique globale, pour rester exploitables par ce pipeline précis.

Contexte déjà exploré (pour éviter les doublons) :
  - tabpfn_v15_features.py (15 groupes) -- tous neutres/rejetés sur écran
    LightGBM 3x10 (tabpfn_v15_feature_results.json).
  - tabpfn_v14_ablation.py (§15quater/15ter de benchmark.md) : profile
    clustering, neighbor_mad, neighbor_interval_width, peak_offpeak,
    collective_index -- tous rejetés.

Chaque fonction reçoit un dict `ctx` :
  values   : s.to_numpy(float) brut (NaN aux positions manquantes de X)
  interp   : interpolation linéaire leak-free de `values`
  index    : DatetimeIndex des timestamps
  scaled   : matrice (n, k) des voisins mis à l'échelle robuste
  missing  : masque booléen des positions manquantes dans X
  n        : nombre de timestamps (= len(values))
et renvoie un tableau (n, k) qui varie par POSITION (jamais une constante
diffusée sur toutes les lignes).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.signal import find_peaks

CATALOG: dict[str, dict] = {}


def register(name, category, note=""):
    def deco(fn):
        CATALOG[name] = {"fn": fn, "category": category, "note": note}
        return fn
    return deco


# ---------------------------------------------------------------------------
# Helpers vectorisés : fenêtres glissantes via sliding_window_view (rapide,
# pas de boucle Python par position -- indispensable vu le volume du criblage
# : jusqu'à ~500 colonnes x 5 repeats x ~55 features).
# ---------------------------------------------------------------------------

def _robust_scale(v):
    obs = v[np.isfinite(v)]
    if len(obs) < 2:
        return 0.0, 1.0
    center = float(np.median(obs))
    scale = max(1.4826 * float(np.median(np.abs(obs - center))), 0.02 * (abs(center) + 1.0), 1.0)
    return center, scale


def _sliding(v, window):
    """(n, window) : fenêtre centrée à chaque position, bords étendus par
    répétition (edge padding) pour ne jamais réduire la couverture."""
    n = len(v)
    half = window // 2
    padded = np.pad(v, (half, window - half - 1), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, window)


def _blocks(n, block):
    """Bornes des blocs NON chevauchants couvrant [0, n)."""
    return [(s, min(s + block, n)) for s in range(0, n, block)]


# ---------------------------------------------------------------------------
# A. Statistiques distributionnelles LOCALES (au-delà de mean/max/min/std/
#    mad/median/quantile déjà présents via les voisins), fenêtre ~2 jours
# ---------------------------------------------------------------------------

W_A = 97  # ~2 jours


@register("local_skewness", "A-distribution", "asymétrie dans une fenêtre glissante (~2j)")
def local_skewness(ctx):
    w = _sliding(ctx["interp"], W_A)
    return sps.skew(w, axis=1).reshape(-1, 1)


@register("local_kurtosis", "A-distribution", "aplatissement dans une fenêtre glissante (~2j)")
def local_kurtosis(ctx):
    w = _sliding(ctx["interp"], W_A)
    return sps.kurtosis(w, axis=1).reshape(-1, 1)


@register("local_iqr", "A-distribution", "écart interquartile dans une fenêtre glissante, normalisé")
def local_iqr(ctx):
    v = ctx["interp"]
    center, scale = _robust_scale(v)
    w = _sliding(v, W_A)
    iqr = np.percentile(w, 75, axis=1) - np.percentile(w, 25, axis=1)
    return (iqr / scale).reshape(-1, 1)


@register("local_trimmed_mean_dev", "A-distribution",
          "écart moyenne tronquée 10%/médiane globale, calculé sur une fenêtre glissante")
def local_trimmed_mean_dev(ctx):
    v = ctx["interp"]
    center, scale = _robust_scale(v)
    w = _sliding(v, W_A)
    sw = np.sort(w, axis=1)
    trim = max(1, int(round(W_A * 0.1)))
    tm = sw[:, trim:-trim].mean(axis=1)
    return ((tm - center) / scale).reshape(-1, 1)


@register("local_cv", "A-distribution", "coefficient de variation (std/|mean|) dans une fenêtre glissante")
def local_cv(ctx):
    w = _sliding(ctx["interp"], W_A)
    return (w.std(axis=1) / (np.abs(w.mean(axis=1)) + 1e-6)).reshape(-1, 1)


@register("local_range", "A-distribution", "étendue (max-min) dans une fenêtre glissante, normalisée")
def local_range(ctx):
    v = ctx["interp"]
    center, scale = _robust_scale(v)
    w = _sliding(v, W_A)
    return ((w.max(axis=1) - w.min(axis=1)) / scale).reshape(-1, 1)


@register("local_mad_std_ratio", "A-distribution",
          "ratio MAD/std local (proxy de queues épaisses) dans une fenêtre glissante")
def local_mad_std_ratio(ctx):
    w = _sliding(ctx["interp"], W_A)
    med = np.median(w, axis=1, keepdims=True)
    mad = np.median(np.abs(w - med), axis=1)
    std = w.std(axis=1) + 1e-9
    return (mad / std).reshape(-1, 1)


@register("robust_zscore_vs_own_curve", "A-distribution",
          "z-score robuste du point vs la MEDIANE/MAD de sa PROPRE courbe entière (jamais vs les voisins)")
def robust_zscore_vs_own_curve(ctx):
    v = ctx["interp"]
    center, scale = _robust_scale(v)
    return ((v - center) / scale).reshape(-1, 1)


# ---------------------------------------------------------------------------
# B. Autocorrélation LOCALE
# ---------------------------------------------------------------------------

def _rolling_acf(v, lag, window):
    w = _sliding(v, window)
    wc = w - w.mean(axis=1, keepdims=True)
    num = np.sum(wc[:, :-lag] * wc[:, lag:], axis=1)
    den = np.sum(wc ** 2, axis=1) + 1e-9
    return num / den


@register("local_acf_lag1_short", "B-autocorr",
          "autocorrélation lag1 (30 min) sur une fenêtre courte (~1j, 49 pts)")
def local_acf_lag1_short(ctx):
    return _rolling_acf(ctx["interp"], 1, 49).reshape(-1, 1)


@register("local_acf_lag1_window", "B-autocorr",
          "autocorrélation lag1 sur une fenêtre plus large (~3j, 145 pts) -- détecte les "
          "changements de régime de lissage le long de la courbe")
def local_acf_lag1_window(ctx):
    return _rolling_acf(ctx["interp"], 1, 145).reshape(-1, 1)


@register("local_acf_lag48", "B-autocorr",
          "autocorrélation à lag 48 (1 jour) sur une fenêtre glissante (~4j, 193 pts)")
def local_acf_lag48(ctx):
    return _rolling_acf(ctx["interp"], 48, 193).reshape(-1, 1)


@register("local_acf_lag336", "B-autocorr",
          "autocorrélation à lag 336 (1 semaine) sur une fenêtre glissante (~10j, 481 pts)")
def local_acf_lag336(ctx):
    return _rolling_acf(ctx["interp"], 336, 481).reshape(-1, 1)


@register("local_pacf_approx_lag48", "B-autocorr",
          "corrélation partielle approx. à lag48 (résidu du lag1 régressé), fenêtre glissante")
def local_pacf_approx_lag48(ctx):
    v = ctx["interp"]
    a1 = _rolling_acf(v, 1, 193)
    lag1_shift = np.r_[v[0], v[:-1]]
    resid = v - np.r_[0.0, a1[1:]] * lag1_shift
    return _rolling_acf(resid, 47, 193).reshape(-1, 1)


@register("local_diff_autocorr_lag1", "B-autocorr",
          "autocorrélation lag1 de la série DIFFÉRENCIÉE, fenêtre glissante "
          "(oscillation vs tendance locale)")
def local_diff_autocorr_lag1(ctx):
    d = np.diff(ctx["interp"], prepend=ctx["interp"][0])
    return _rolling_acf(d, 1, 97).reshape(-1, 1)


# ---------------------------------------------------------------------------
# C. Spectre de Fourier LOCAL (blocs non chevauchants, valeur répétée sur le
#    bloc -- un vrai spectrogramme point par point serait hors budget)
# ---------------------------------------------------------------------------

def _block_apply(v, block, fn, fallback=0.0):
    n = len(v)
    out = np.full(n, fallback)
    prev = fallback
    for a, b in _blocks(n, block):
        seg = v[a:b]
        val = fn(seg) if len(seg) >= 8 else prev
        out[a:b] = val
        prev = val
    return out


def _fft_power(seg):
    seg = seg - seg.mean()
    return np.abs(np.fft.rfft(seg)) ** 2


@register("local_fft_daily_energy", "C-spectral",
          "puissance spectrale à la fréquence journalière (1/48), par blocs de 96 pts (~2j)")
def local_fft_daily_energy(ctx):
    def fn(seg):
        spec = _fft_power(seg)
        idx = max(1, round(len(seg) / 48))
        idx = min(idx, len(spec) - 1)
        return spec[idx] / (spec.sum() + 1e-9)
    return _block_apply(ctx["interp"], 96, fn).reshape(-1, 1)


@register("local_fft_weekly_energy", "C-spectral",
          "puissance spectrale à la fréquence hebdomadaire (1/336), par blocs de 480 pts (~10j)")
def local_fft_weekly_energy(ctx):
    def fn(seg):
        spec = _fft_power(seg)
        idx = max(1, round(len(seg) / 336))
        idx = min(idx, len(spec) - 1)
        return spec[idx] / (spec.sum() + 1e-9)
    return _block_apply(ctx["interp"], 480, fn).reshape(-1, 1)


@register("local_spectral_entropy_fine", "C-spectral",
          "entropie spectrale par blocs FINS de 48 pts (~1j)")
def local_spectral_entropy_fine(ctx):
    def fn(seg):
        spec = _fft_power(seg)
        p = spec / (spec.sum() + 1e-9)
        p = p[p > 0]
        return -np.sum(p * np.log(p)) / np.log(len(p) + 1e-9) if len(p) > 1 else 0.0
    return _block_apply(ctx["interp"], 48, fn).reshape(-1, 1)


@register("local_spectral_entropy_coarse", "C-spectral",
          "entropie spectrale par blocs plus LARGES de 96 pts (~2j)")
def local_spectral_entropy_coarse(ctx):
    def fn(seg):
        spec = _fft_power(seg)
        p = spec / (spec.sum() + 1e-9)
        p = p[p > 0]
        return -np.sum(p * np.log(p)) / np.log(len(p) + 1e-9) if len(p) > 1 else 0.0
    return _block_apply(ctx["interp"], 96, fn).reshape(-1, 1)


@register("local_dominant_freq_ratio", "C-spectral",
          "part de puissance du plus fort bin non-DC par blocs de 96 pts (périodicité locale dominante)")
def local_dominant_freq_ratio(ctx):
    def fn(seg):
        spec = _fft_power(seg)
        total = spec.sum() + 1e-9
        return spec[1:].max() / total if len(spec) > 1 else 0.0
    return _block_apply(ctx["interp"], 96, fn).reshape(-1, 1)


@register("local_high_freq_energy_ratio", "C-spectral",
          "part de l'énergie spectrale au-delà de la fréquence journalière, par blocs de 96 pts")
def local_high_freq_energy_ratio(ctx):
    def fn(seg):
        spec = _fft_power(seg)
        cutoff = max(1, round(len(seg) / 48))
        total = spec.sum() + 1e-9
        return spec[cutoff:].sum() / total
    return _block_apply(ctx["interp"], 96, fn).reshape(-1, 1)


# ---------------------------------------------------------------------------
# D. Complexité / entropie LOCALE
# ---------------------------------------------------------------------------

def _sample_entropy_block(v, m=2, r_frac=0.2):
    n = len(v)
    r = r_frac * np.std(v)
    if r < 1e-9 or n < m + 2:
        return 0.0

    def _phi(mm):
        templates = np.array([v[i:i + mm] for i in range(n - mm)])
        dist = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
        return np.sum(dist <= r) - len(templates)

    cm, cm1 = _phi(m), _phi(m + 1)
    if cm <= 0 or cm1 <= 0:
        return 0.0
    return float(-np.log(cm1 / cm))


@register("local_sample_entropy", "D-complexity",
          "SampEn(m=2, r=0.2*std) par blocs de 150 pts (~3j) -- complexité locale")
def local_sample_entropy(ctx):
    return _block_apply(ctx["interp"], 150, _sample_entropy_block).reshape(-1, 1)


def _permutation_entropy_block(v, order=3):
    n = len(v)
    if n <= order:
        return 0.0
    patterns = {}
    for i in range(n - order):
        perm = tuple(np.argsort(v[i:i + order]))
        patterns[perm] = patterns.get(perm, 0) + 1
    counts = np.array(list(patterns.values()), dtype=float)
    p = counts / counts.sum()
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(math.factorial(order)))


@register("local_permutation_entropy", "D-complexity",
          "entropie ordinale (motifs d'ordre 3) par blocs de 96 pts")
def local_permutation_entropy(ctx):
    return _block_apply(ctx["interp"], 96, _permutation_entropy_block).reshape(-1, 1)


@register("local_hjorth_mobility", "D-complexity",
          "paramètre de Hjorth (mobilité=sqrt(var(diff)/var(x))) en fenêtre glissante (~2j)")
def local_hjorth_mobility(ctx):
    v = ctx["interp"]
    w = _sliding(v, W_A)
    dw = np.diff(w, axis=1)
    mob = np.sqrt(dw.var(axis=1) / (w.var(axis=1) + 1e-9))
    return mob.reshape(-1, 1)


@register("local_hjorth_complexity", "D-complexity",
          "paramètre de Hjorth (complexité) en fenêtre glissante plus large (~3j, 145 pts)")
def local_hjorth_complexity(ctx):
    v = ctx["interp"]
    w = _sliding(v, 145)
    d1 = np.diff(w, axis=1)
    d2 = np.diff(d1, axis=1)
    mob0 = np.sqrt(d1.var(axis=1) / (w.var(axis=1) + 1e-9))
    mob1 = np.sqrt(d2.var(axis=1) / (d1.var(axis=1) + 1e-9))
    return (mob1 / (mob0 + 1e-9)).reshape(-1, 1)


def _lz_complexity(bits):
    s = "".join(str(b) for b in bits)
    n = len(s)
    if n < 4:
        return 0.0
    i, c, k, l = 0, 1, 1, 1
    while True:
        if s[i + k - 1] != s[l + k - 1]:
            if k > 1:
                i += 1
            else:
                c += 1
                l += k
                i = 0
                k = 1
                if l + 1 > n:
                    break
                continue
            k = 1
        else:
            k += 1
            if l + k > n:
                c += 1
                break
        if i + k > l:
            i = 0
    norm = n / max(np.log2(n), 1.0)
    return float(c / norm) if norm > 0 else 0.0


@register("local_lempel_ziv_complexity", "D-complexity",
          "complexité de Lempel-Ziv normalisée (série binarisée par la médiane du bloc), blocs de 150 pts")
def local_lempel_ziv_complexity(ctx):
    def fn(seg):
        bits = (seg > np.median(seg)).astype(int)
        return _lz_complexity(bits)
    return _block_apply(ctx["interp"], 150, fn).reshape(-1, 1)


@register("local_cid_ce", "D-complexity",
          "CID (complexity-invariant distance) self-stat en fenêtre glissante (~2j), catch22-like")
def local_cid_ce(ctx):
    v = ctx["interp"]
    center, scale = _robust_scale(v)
    w = _sliding(v, W_A) / scale
    d = np.diff(w, axis=1)
    ce = np.sqrt(np.sum(d ** 2, axis=1)) / np.sqrt(w.shape[1])
    return ce.reshape(-1, 1)


@register("local_frac_above_window_mean", "D-complexity",
          "fraction de la fenêtre glissante (~2j) au-dessus de la moyenne DE CETTE FENÊTRE "
          "(persistance locale, proxy de longest_strike_above_mean)")
def local_frac_above_window_mean(ctx):
    w = _sliding(ctx["interp"], W_A)
    above = w > w.mean(axis=1, keepdims=True)
    return above.mean(axis=1).reshape(-1, 1)


# ---------------------------------------------------------------------------
# E. Tendance / rupture LOCALE
# ---------------------------------------------------------------------------

@register("local_trend_r2", "E-trend",
          "R2 d'une régression linéaire locale (fenêtre ~2j) -- qualité/force de la tendance locale")
def local_trend_r2(ctx):
    v = ctx["interp"]
    w = _sliding(v, W_A)
    t = np.arange(W_A, dtype=float)
    tc = t - t.mean()
    wc = w - w.mean(axis=1, keepdims=True)
    cov = (wc * tc).sum(axis=1)
    var_t = (tc ** 2).sum()
    var_w = (wc ** 2).sum(axis=1) + 1e-9
    r2 = (cov ** 2) / (var_t * var_w)
    return r2.reshape(-1, 1)


@register("local_turning_point_rate", "E-trend",
          "taux de changements de signe de la différence 1ere en fenêtre glissante (~2j) -- rugosité locale")
def local_turning_point_rate(ctx):
    d = np.diff(ctx["interp"], prepend=ctx["interp"][0])
    sign = np.sign(d)
    sign[sign == 0] = 1
    turn = np.r_[0, (sign[1:] != sign[:-1]).astype(float)]
    w = _sliding(turn, W_A)
    return w.mean(axis=1).reshape(-1, 1)


@register("local_cusum_deviation", "E-trend",
          "déviation CUSUM (cumul des écarts à la moyenne globale) AU point, normalisée -- "
          "indique où se situe une éventuelle rupture de niveau, pas seulement son max global")
def local_cusum_deviation(ctx):
    v = ctx["interp"]
    center, scale = _robust_scale(v)
    cusum = np.cumsum(v - v.mean())
    return (cusum / (scale * len(v))).reshape(-1, 1)


@register("piecewise_level_shift_flag", "E-trend",
          "indicateur binaire: le point est-il après la plus grande rupture de niveau détectée (CUSUM argmax)")
def piecewise_level_shift_flag(ctx):
    v = ctx["interp"]
    cusum = np.cumsum(v - v.mean())
    idx_break = np.argmax(np.abs(cusum))
    flag = (np.arange(len(v)) > idx_break).astype(float)
    return flag.reshape(-1, 1)


@register("rolling_trend_multiscale_own", "E-trend",
          "pente de régression glissante (moindres carrés) à 3 échelles (24/96/288 pts) sur la courbe propre")
def rolling_trend_multiscale_own(ctx):
    v = ctx["interp"]
    center, scale = _robust_scale(v)
    t = pd.Series(np.arange(len(v), dtype=float))
    sv = pd.Series(v)
    cols = []
    for w in (24, 96, 288):
        cov = sv.rolling(w, center=True, min_periods=5).cov(t)
        var = t.rolling(w, center=True, min_periods=5).var()
        slope = (cov / var.replace(0, np.nan)).fillna(0.0).to_numpy() / scale
        cols.append(slope)
    return np.column_stack(cols)


@register("rolling_std_of_slope", "E-trend",
          "écart-type glissant de la pente locale (fenêtre 96) -- capture les zones où la tendance change vite")
def rolling_std_of_slope(ctx):
    v = ctx["interp"]
    d = np.diff(v, prepend=v[0])
    s = pd.Series(d)
    out = s.rolling(96, center=True, min_periods=10).std().fillna(0.0).to_numpy()
    center, scale = _robust_scale(v)
    return (out / scale).reshape(-1, 1)


# ---------------------------------------------------------------------------
# F. Forme / pics LOCALE
# ---------------------------------------------------------------------------

@register("local_peak_density", "F-peaks",
          "densité de pics locaux (> médiane+1 MAD du bloc) par blocs de 97 pts (~2j)")
def local_peak_density(ctx):
    def fn(seg):
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) * 1.4826 + 1e-9
        peaks, _ = find_peaks(seg, height=med + mad)
        return len(peaks) / len(seg)
    return _block_apply(ctx["interp"], 97, fn).reshape(-1, 1)


@register("time_since_last_peak", "F-peaks",
          "distance (en points) au pic local le plus proche (avant), normalisée")
def time_since_last_peak(ctx):
    v = ctx["interp"]
    peaks, _ = find_peaks(v)
    n = len(v)
    dist = np.full(n, n, dtype=float)
    peak_set = set(peaks.tolist())
    last = -1
    for i in range(n):
        if i in peak_set:
            last = i
        dist[i] = (i - last) if last >= 0 else n
    return (dist / n).reshape(-1, 1)


@register("beyond_r_sigma_flag", "F-peaks",
          "indicateur PAR POINT: est-il au-delà de 1.5 sigma (std global) de la moyenne globale "
          "(catch22 ratio_beyond_r_sigma rendu local en gardant l'indicateur brut par point)")
def beyond_r_sigma_flag(ctx):
    v = ctx["interp"]
    mu, sd = v.mean(), v.std() + 1e-9
    return (np.abs(v - mu) > 1.5 * sd).astype(float).reshape(-1, 1)


@register("nearest_peak_prominence", "F-peaks",
          "proéminence du pic le plus proche (avant OU après), normalisée -- variante par point "
          "de mean_peak_prominence")
def nearest_peak_prominence(ctx):
    v = ctx["interp"]
    center, scale = _robust_scale(v)
    peaks, props = find_peaks(v, prominence=0)
    n = len(v)
    if not len(peaks):
        return np.zeros((n, 1))
    prom = props["prominences"] / scale
    idx = np.arange(n)
    nearest = np.argmin(np.abs(idx[:, None] - peaks[None, :]), axis=1)
    return prom[nearest].reshape(-1, 1)


@register("local_zero_crossing_rate", "F-peaks",
          "taux de passages par zéro de la différence 1ere (fenêtre glissante ~2j) -- fréquence "
          "d'oscillation locale")
def local_zero_crossing_rate(ctx):
    d = np.diff(ctx["interp"], prepend=ctx["interp"][0])
    d = d - np.median(d)
    cross = np.r_[0, (np.diff(np.sign(d)) != 0).astype(float)]
    w = _sliding(cross, W_A)
    return w.mean(axis=1).reshape(-1, 1)


@register("bucket_volatility_percentile", "F-peaks",
          "rang percentile (parmi les 48 créneaux de 30 min) de l'écart-type OBSERVÉ propre à ce "
          "créneau -- indique si le créneau du point est historiquement calme ou volatil pour cette courbe")
def bucket_volatility_percentile(ctx):
    v = ctx["values"]
    dt = pd.DatetimeIndex(ctx["index"])
    bucket = dt.hour.values * 2 + dt.minute.values // 30
    observed = np.isfinite(v)
    df = pd.DataFrame({"bucket": bucket, "val": v})
    std_by_bucket = df.loc[observed].groupby("bucket")["val"].std().fillna(0.0)
    ranks = std_by_bucket.rank(pct=True)
    global_rank = 0.5
    feat = pd.Series(bucket).map(ranks).fillna(global_rank).to_numpy()
    return feat.reshape(-1, 1)


# ---------------------------------------------------------------------------
# G. Calendrier conditionnel (déjà par-point, fenêtre = position dans le
#    calendrier, pas de conversion nécessaire)
# ---------------------------------------------------------------------------

@register("hour_bucket_iqr", "G-calendar",
          "IQR (pas seulement la médiane) des valeurs OBSERVÉES au même créneau de 30 min")
def hour_bucket_iqr(ctx):
    v = ctx["values"]
    dt = pd.DatetimeIndex(ctx["index"])
    bucket = dt.hour.values * 60 + dt.minute.values
    observed = np.isfinite(v)
    df = pd.DataFrame({"bucket": bucket, "val": v})
    q75 = df.loc[observed].groupby("bucket")["val"].quantile(.75)
    q25 = df.loc[observed].groupby("bucket")["val"].quantile(.25)
    iqr_by_bucket = (q75 - q25)
    center, scale = _robust_scale(ctx["interp"])
    global_iqr = float(np.nanmedian(iqr_by_bucket)) if len(iqr_by_bucket) else 0.0
    feat = pd.Series(bucket).map(iqr_by_bucket).fillna(global_iqr).to_numpy()
    return (feat / scale).reshape(-1, 1)


@register("weekday_weekend_mean_std", "G-calendar",
          "moyenne/std par (créneau x semaine/week-end) au lieu d'un médian fusionné")
def weekday_weekend_mean_std(ctx):
    v = ctx["values"]
    dt = pd.DatetimeIndex(ctx["index"])
    bucket = dt.hour.values * 60 + dt.minute.values
    weekend = (dt.dayofweek.values >= 5).astype(int)
    observed = np.isfinite(v)
    df = pd.DataFrame({"bucket": bucket, "weekend": weekend, "val": v})
    obs = df.loc[observed]
    grp = obs.groupby(["weekend", "bucket"])["val"]
    mean_map = grp.mean()
    std_map = grp.std()
    center, scale = _robust_scale(ctx["interp"])
    global_mean = float(np.nanmean(v[observed])) if observed.any() else 0.0
    keys = list(zip(weekend, bucket))
    mean_feat = np.array([mean_map.get(k, global_mean) for k in keys])
    std_feat = np.array([std_map.get(k, 0.0) for k in keys])
    std_feat = np.nan_to_num(std_feat, nan=0.0)
    return np.column_stack([(mean_feat - center) / scale, std_feat / scale])


@register("month_position_level", "G-calendar",
          "niveau moyen selon la position grossière dans la période (début/milieu/fin janvier)")
def month_position_level(ctx):
    v = ctx["values"]
    dt = pd.DatetimeIndex(ctx["index"])
    third = pd.cut(dt.day, bins=3, labels=False)
    observed = np.isfinite(v)
    df = pd.DataFrame({"third": third, "val": v})
    mean_by_third = df.loc[observed].groupby("third")["val"].mean()
    center, scale = _robust_scale(ctx["interp"])
    global_mean = float(np.nanmean(v[observed])) if observed.any() else 0.0
    feat = pd.Series(third).map(mean_by_third).fillna(global_mean).to_numpy()
    return ((feat - center) / scale).reshape(-1, 1)


@register("hour_bucket_trend", "G-calendar",
          "pente (jour 1 -> jour N) de la valeur observée à CHAQUE créneau horaire (dérive fine par heure)")
def hour_bucket_trend(ctx):
    v = ctx["values"]
    dt = pd.DatetimeIndex(ctx["index"])
    bucket = dt.hour.values * 60 + dt.minute.values
    day_idx = (dt.normalize() - dt.normalize().min()).days
    observed = np.isfinite(v)
    center, scale = _robust_scale(ctx["interp"])
    slopes = {}
    for b in np.unique(bucket):
        m = observed & (bucket == b)
        if m.sum() >= 3 and np.std(day_idx[m]) > 0:
            slope = sps.linregress(day_idx[m], v[m]).slope
        else:
            slope = 0.0
        slopes[b] = slope
    feat = np.array([slopes.get(b, 0.0) for b in bucket])
    return (feat / scale).reshape(-1, 1)


@register("weekday_specific_median", "G-calendar",
          "médiane par (jour de semaine complet x créneau) -- plus fin que semaine/week-end binaire")
def weekday_specific_median(ctx):
    v = ctx["values"]
    dt = pd.DatetimeIndex(ctx["index"])
    bucket = dt.hour.values * 60 + dt.minute.values
    dow = dt.dayofweek.values
    observed = np.isfinite(v)
    df = pd.DataFrame({"dow": dow, "bucket": bucket, "val": v})
    med_map = df.loc[observed].groupby(["dow", "bucket"])["val"].median()
    med_by_bucket = df.loc[observed].groupby("bucket")["val"].median()
    center, scale = _robust_scale(ctx["interp"])
    keys = list(zip(dow, bucket))
    feat = np.array([med_map.get(k, med_by_bucket.get(b, center))
                      for k, b in zip(keys, bucket)])
    return ((feat - center) / scale).reshape(-1, 1)


@register("weekend_transition_distance", "G-calendar",
          "distance (en points) à la transition semaine<->week-end la plus proche")
def weekend_transition_distance(ctx):
    dt = pd.DatetimeIndex(ctx["index"])
    weekend = (dt.dayofweek.values >= 5)
    change = np.flatnonzero(np.diff(weekend.astype(int)) != 0) + 1
    n = len(weekend)
    if not len(change):
        return np.full((n, 1), float(n))
    pos = np.arange(n)
    dist = np.min(np.abs(pos[:, None] - change[None, :]), axis=1)
    return (dist / n).reshape(-1, 1)


@register("week_index_flag", "G-calendar",
          "index de la semaine calendaire (0/1/2) observée dans la fenêtre de 3 semaines -- dérive semaine 1 vs 3")
def week_index_flag(ctx):
    dt = pd.DatetimeIndex(ctx["index"])
    days = (dt.normalize() - dt.normalize().min()).days.to_numpy()
    week = np.clip(days // 7, 0, 2).astype(float)
    return week.reshape(-1, 1)


# ---------------------------------------------------------------------------
# H. Distribution du consensus voisins (déjà par-point : la matrice `scaled`
#    a des valeurs différentes à chaque timestamp, donc toute statistique
#    calculée par ligne varie naturellement -- pas de conversion nécessaire,
#    sauf l'autocorrélation qui portait sur la série CONSENSUS entière)
# ---------------------------------------------------------------------------

@register("neighbor_skewness", "H-neighbors", "asymétrie inter-voisins à chaque position")
def neighbor_skewness(ctx):
    sc = ctx["scaled"]
    return sps.skew(sc, axis=1, nan_policy="omit").reshape(-1, 1)


@register("neighbor_kurtosis", "H-neighbors", "aplatissement inter-voisins à chaque position")
def neighbor_kurtosis(ctx):
    sc = ctx["scaled"]
    return sps.kurtosis(sc, axis=1, nan_policy="omit").reshape(-1, 1)


@register("neighbor_bimodality_coeff", "H-neighbors",
          "coefficient de bimodalité (skew^2+1)/kurtosis -- désaccord bimodal vs bruit unimodal")
def neighbor_bimodality_coeff(ctx):
    sc = ctx["scaled"]
    sk = sps.skew(sc, axis=1)
    ku = sps.kurtosis(sc, axis=1, fisher=False) + 1e-6
    return ((sk ** 2 + 1) / ku).reshape(-1, 1)


@register("neighbor_rank_of_interp", "H-neighbors",
          "rang percentile de la valeur interpolée propre dans la distribution des voisins")
def neighbor_rank_of_interp(ctx):
    sc = ctx["scaled"]
    proxy = ctx["interp"]
    rank = (sc <= proxy[:, None]).mean(axis=1)
    return rank.reshape(-1, 1)


@register("neighbor_entropy", "H-neighbors",
          "entropie de Shannon de l'histogramme (8 bins) des voisins à chaque position")
def neighbor_entropy(ctx):
    sc = ctx["scaled"]
    n = sc.shape[0]
    out = np.zeros(n)
    lo, hi = np.percentile(sc, 1), np.percentile(sc, 99)
    bins = np.linspace(lo, hi, 9) if hi > lo else np.linspace(-1, 1, 9)
    for i in range(n):
        hist, _ = np.histogram(sc[i], bins=bins)
        p = hist / (hist.sum() + 1e-9)
        p = p[p > 0]
        out[i] = -np.sum(p * np.log(p))
    return out.reshape(-1, 1)


@register("local_neighbor_consensus_acf", "H-neighbors",
          "autocorrélation lag1 LOCALE (fenêtre ~2j) de la série CONSENSUS des voisins "
          "(moyenne des voisins scalés) -- lissage/bruit du signal typique suivi, pas de la cible")
def local_neighbor_consensus_acf(ctx):
    consensus = ctx["scaled"].mean(axis=1)
    return _rolling_acf(consensus, 1, W_A).reshape(-1, 1)


@register("neighbor_disagreement_trend", "H-neighbors",
          "tendance glissante (fenêtre 96) de l'écart-type inter-voisins -- incertitude croissante/decroissante")
def neighbor_disagreement_trend(ctx):
    disagreement = ctx["scaled"].std(axis=1)
    s = pd.Series(disagreement)
    t = pd.Series(np.arange(len(disagreement), dtype=float))
    w = 96
    cov = s.rolling(w, center=True, min_periods=10).cov(t)
    var = t.rolling(w, center=True, min_periods=10).var()
    slope = (cov / var.replace(0, np.nan)).fillna(0.0).to_numpy()
    return slope.reshape(-1, 1)


# ---------------------------------------------------------------------------
# I. Contexte de manquance (déjà par-point : densité dans une fenêtre autour
#    de CHAQUE position -- au-delà de run-length/edge-distance déjà utilisés)
# ---------------------------------------------------------------------------

@register("local_missing_density_48", "I-missingness",
          "proportion de points manquants dans une fenêtre +/-48 (1 jour) autour de chaque position")
def local_missing_density_48(ctx):
    m = ctx["missing"].astype(float)
    s = pd.Series(m)
    out = s.rolling(97, center=True, min_periods=1).mean().to_numpy()
    return out.reshape(-1, 1)


@register("local_missing_density_336", "I-missingness",
          "proportion de points manquants dans une fenêtre +/-336 (1 semaine)")
def local_missing_density_336(ctx):
    m = ctx["missing"].astype(float)
    s = pd.Series(m)
    out = s.rolling(673, center=True, min_periods=1).mean().to_numpy()
    return out.reshape(-1, 1)


@register("distance_weighted_missing_density", "I-missingness",
          "densité de manquance pondérée par proximité (noyau exponentiel, tau=48)")
def distance_weighted_missing_density(ctx):
    m = ctx["missing"].astype(float)
    n = len(m)
    tau = 48.0
    kernel_half = np.exp(-np.arange(0, 4 * int(tau)) / tau)
    kernel = np.r_[kernel_half[::-1][:-1], kernel_half]
    conv = np.convolve(m, kernel, mode="same") / kernel.sum()
    return conv.reshape(-1, 1)


# ---------------------------------------------------------------------------
# J. Extras (au-delà de 50, testés aussi si le temps le permet)
# ---------------------------------------------------------------------------

@register("own_vs_neighbor_daily_profile_gap", "J-extra",
          "écart PAR POINT entre le niveau typique propre à ce créneau (profil journalier propre) "
          "et le niveau typique du consensus voisin au même créneau -- la courbe est-elle "
          "structurellement au-dessus/en-dessous de ses voisins à CETTE heure précise")
def own_vs_neighbor_daily_profile_gap(ctx):
    v = ctx["interp"]
    dt = pd.DatetimeIndex(ctx["index"])
    bucket = dt.hour.values * 2 + dt.minute.values // 30
    consensus = ctx["scaled"].mean(axis=1)
    center, scale = _robust_scale(v)
    own_profile = np.array([np.median(v[bucket == b]) if (bucket == b).any() else center
                             for b in range(48)])
    cons_profile = np.array([np.median(consensus[bucket == b]) if (bucket == b).any() else 0.0
                              for b in range(48)])
    gap_by_bucket = (own_profile - center) / scale - cons_profile
    return gap_by_bucket[bucket].reshape(-1, 1)


@register("local_percentile_rank_own", "J-extra",
          "rang percentile du point dans une fenêtre glissante (~2j) de sa PROPRE courbe "
          "(complète robust_zscore_vs_own_curve, qui est global, par une version locale)")
def local_percentile_rank_own(ctx):
    v = ctx["interp"]
    w = _sliding(v, W_A)
    center_col = W_A // 2
    rank = (w <= w[:, center_col:center_col + 1]).mean(axis=1)
    return rank.reshape(-1, 1)


@register("local_hurst_exponent_proxy", "J-extra",
          "exposant de Hurst approché (analyse R/S) en fenêtre glissante (~3j, 145 pts) -- "
          "mesure de persistance/mémoire longue, jamais essayée dans ce dépôt")
def local_hurst_exponent_proxy(ctx):
    v = ctx["interp"]
    w = _sliding(v, 145)
    wc = w - w.mean(axis=1, keepdims=True)
    cum = np.cumsum(wc, axis=1)
    r = cum.max(axis=1) - cum.min(axis=1)
    s = w.std(axis=1) + 1e-9
    rs = r / s
    hurst = np.log(rs + 1e-9) / np.log(145)
    return hurst.reshape(-1, 1)


assert len(CATALOG) >= 50, f"Only {len(CATALOG)} features registered"
