"""V2 : SAITS "natif" (matrice brute + masque interne) vs TabPFN v14,
protocole officiel 10x10 -- copie de saits_v14_paired_eval.py, côté SAITS
entièrement redessiné.

Pourquoi cette V2
-----------------
La V1 (`saits_v14_paired_eval.py`) réutilisait telles quelles les features
artisanales de TabPFN v14 (`build_v14_features` : copies périodiques par
colonne, saisonnalité par créneau, stats de voisins scalés au niveau de la
cible...) comme entrée de SAITS. Ce n'est pas ce pour quoi SAITS est conçu,
et ça a un coût : ~1430 colonnes pour 10 cibles, l'attention sur l'axe
features devient hors de portée, et ça reproduit exactement le style
"résumé par colonne" que benchmark.md §15quater a déjà trouvé inutile...
pour TabPFN (10 idées de features testées, 0 gain -- "TabPFN, modèle
foundation par apprentissage en contexte, semble déjà extraire l'essentiel
du signal brut sans qu'un résumé statistique manuel soit nécessaire").

SAITS n'est PAS TabPFN : c'est un modèle entraîné from scratch sur LA
matrice (temps x courbes), dont l'intérêt est justement d'apprendre lui-même
la structure INTER-courbes par self-attention sur l'axe `n_features`, sans
qu'on la lui pré-mâche colonne par colonne. Lui donner des features
artisanales par colonne revient à contourner ce qui est censé être son
avantage sur TabPFN (qui, lui, traite les colonnes plus indépendamment).

Design V2 (repris des consignes) :
- **Entrée de base (obligatoire)** : la matrice brute `X` (temps x courbes),
  NaN compris -- SAITS gère le masque observé/manquant en interne (DMSA),
  rien à construire nous-mêmes ici. Donc plus de ffill/bfill des donneurs
  comme en V1 : on laisse SAITS voir leurs quelques vrais trous aussi.
- **Covariables temporelles globales**, ajoutées UNE SEULE FOIS (partagées
  par toutes les colonnes au même timestamp, pas dupliquées par courbe) :
  heure/jour de semaine/position dans l'année (sin/cos) + week-end, via
  `feb.time_features_full_seasonality(index, ("hour","dow","year"))`.
  Jour férié omis : la fenêtre de données (9-31 janvier 2023) ne contient
  aucun jour férié français (le seul de la période, le 1er janvier, est
  avant le début des données) -- un flag à 0 partout n'apporte rien.
- **Canaux auxiliaires optionnels, PAR courbe** (contrairement aux
  covariables temporelles, ceux-ci sont propres à chaque courbe, donc
  dupliqués par courbe) : volatilité locale (`feb.curve_volatility`, un
  scalaire par courbe diffusé sur tous les pas) et accord périodique
  (`feb.periodic_agreement`, un score par pas de temps) -- signal
  dynamique/de confiance, pas un résumé figé "moyenne/écart-type du
  client" (ça, c'est explicitement ce qu'il ne faut PAS coder en dur : cf.
  ci-dessous).
- **Explicitement exclu de l'entrée SAITS** : copies périodiques brutes par
  lag, saisonnalité par créneau, stats de voisins scalés au niveau d'UNE
  cible (ce dernier point présuppose justement la relation cible/voisin que
  l'attention est censée découvrir seule), contexte de trou (durée, distance
  au bord) -- ce dernier est gardé pour le POST-traitement ci-dessous, comme
  suggéré, pas comme feature d'entrée.
- **Post-traitement / ensemble** : le blend alpha-interpolation de V14
  (`continuous_alpha` x `periodic_agreement` x `volatility_alpha_mult`,
  mêmes poids que V14_BEST) est réappliqué APRÈS la prédiction brute de
  SAITS, exactement comme suggéré ("utiles surtout en post-traitement/
  ensemble") -- c'est le seul levier qui ait rapporté plus que n'importe
  quelle feature dans toute la recherche V14 (§13, -4,53%), donc le premier
  à essayer ici plutôt que de le laisser de côté.
- **Sous-échantillonnage des courbes** : comme recommandé ("teste d'abord
  sur 50-100 clients les plus corrélés avant de scaler à 1000"), on ne
  passe pas les 20 000 donneurs à SAITS -- seulement les
  `max_features` (défaut 90) les plus corrélés aux 10 cibles du repeat
  (union par colonne, `feb.adaptive_top_k_corr_euclid`, même sélection que
  le KNN de TabPFN v14).

Protocole (inchangé, strictement comparable à tabpfn_v14_confirm.py,
66,24 +/- 32,45 W) : voir docstring de saits_v14_paired_eval.py, identique
ici bit pour bit (mêmes seeds 42+i, mêmes val_cols par repeat, TabPFN et
SAITS évalués sur les mêmes colonnes -> comparaison appariée repeat par
repeat ET colonne par colonne).

Visualisation entraînement / validation (nouveau)
--------------------------------------------------
Une partie des FENÊTRES (pas des points, `val_frac` par repeat) est mise de
côté avant `saits.fit` pour servir de set de validation interne PyPOTS :
on y masque en plus, au hasard, une fraction (`val_hide_frac`) des points
DÉJÀ OBSERVÉS (jamais les vrais trous des colonnes cibles, jamais y_train)
pour avoir une vérité connue à comparer. PyPOTS logue alors "Epoch NNN -
training loss (...): x, validation ...: y" à CHAQUE epoch en direct dans la
console (`verbose=True`) ; un handler de logging (`_LossCapture`) récupère
ces valeurs sans dépendre d'attributs internes non documentés, et un PNG
train/val loss vs epoch est sauvegardé par repeat dans
`claude2/saits_training_plots/`. Ce set de validation interne ne sert QUE
pendant l'entraînement (courbe + early stopping, `patience`) -- la métrique
officielle reportée à la fin reste calculée séparément sur les vrais trous
vs y_train, comme avant.

Usage
-----
    .venv/bin/pip install pypots
    SAITS_PROFILE=smoke .venv/bin/python claude2/saits_v14_paired_eval_v2.py
    SAITS_PROFILE=full  .venv/bin/python claude2/saits_v14_paired_eval_v2.py

Resumable comme le reste du repo (progress JSON par repeat).
"""
import os
import re
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import tabpfn_client
import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif : on sauvegarde des PNG
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from evaluate import load_data, fill_nan_with_interpolation
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14

PROGRESS_PATH = os.path.join(HERE, "saits_v14_paired_v2_progress.json")
RESULTS_PATH = os.path.join(HERE, "saits_v14_paired_v2_results.json")
PLOTS_DIR = os.path.join(HERE, "saits_training_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Config TabPFN v14 = BEST confirmée (tabpfn_v14_confirm.py, 66,24 +/- 32,45 W)
V14_BEST = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                 extra_lags=(24,), neighbor_rolling=True)
# Mêmes poids, réutilisés pour le blend post-traitement côté SAITS.
VOL_WEIGHT = V14_BEST["vol_weight"]
PERIODIC_ALPHA_STRENGTH = V14_BEST["periodic_alpha_strength"]

PROFILE = os.environ.get("SAITS_PROFILE", "smoke")
N_REPEATS = int(os.environ.get("SAITS_N_REPEATS", 2 if PROFILE == "smoke" else 10))
N_VAL_SUB = 10  # colonnes de validation par repeat, comme run_config(n_val_sub=10)

# val_frac : fraction des FENÊTRES (pas des points) mises de côté pour un
#   set de validation interne PyPOTS -- entièrement séparé des fenêtres
#   d'entraînement (aucun chevauchement de fenêtre entre train et val).
# val_hide_frac : sur ces fenêtres de validation, fraction des points DÉJÀ
#   OBSERVÉS masqués en plus (artificiellement) pour avoir une vérité connue
#   à comparer -- jamais les vrais trous des colonnes cibles ni y_train,
#   uniquement pour piloter l'early stopping et visualiser la courbe train/
#   val (la métrique officielle reste calculée séparément, cf. plus bas).
# epochs/patience relevés par rapport à un simple smoke test (5 epochs) :
#   sans un minimum d'epochs il n'y a rien d'intéressant à visualiser.
_SMOKE = dict(
    max_features=20, aux_channels=True, window_len=96, stride=48,
    n_layers=1, d_model=64, n_heads=2, d_k=32, d_v=32, d_ffn=64, dropout=0.1,
    epochs=30, patience=8, batch_size=8, device=None,
    val_frac=0.2, val_hide_frac=0.1, plot_training=True,
)
_FULL = dict(
    # 50-100 courbes corrélées comme recommandé ; avec les canaux auxiliaires
    # (2 par courbe) + 7 covariables temporelles globales, F ~ (10+90)*3+7
    # ~= 307 -- très en dessous des ~1430 de la V1 malgré l'enrichissement.
    max_features=90, aux_channels=True, window_len=336, stride=48,
    n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=128, dropout=0.1,
    epochs=200, patience=20, batch_size=16, device=None,
    val_frac=0.15, val_hide_frac=0.1, plot_training=True,
)
SAITS_OPTS = _SMOKE if PROFILE == "smoke" else _FULL


# ---------------------------------------------------------------------------
# Côté TabPFN v14 : inchangé par rapport à la V1 (copie de
# tabpfn_v14_ablation.eval_split, erreurs absolues PAR COLONNE pour permettre
# l'appariement colonne-par-colonne avec SAITS).
# ---------------------------------------------------------------------------

def eval_tabpfn_repeat(X_train, y_train, donor_cols, dev_cols, val_cols, opts):
    donor_df = X_train[donor_cols]
    dev_true_df = y_train[dev_cols]
    val_holed_df = X_train[val_cols]

    pool = pd.concat([donor_df, dev_true_df], axis=1)
    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=15, random_state=v14.RANDOM_STATE)
    dev_filled = dev_true_df.ffill().bfill()
    Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
    pool_labels = pd.concat([donor_labels, dev_labels])
    pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
    cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(val_holed_df.index, ("hour", "dow", "dom", "year"))

    abs_err_by_col = {}
    with ThreadPoolExecutor(max_workers=v14.MAX_WORKERS) as executor:
        futures = [executor.submit(v14.complete_column_v14, c, val_holed_df[c], y_train[c],
                                    pool_vals, km, cluster_col_idx, tfeat,
                                    val_holed_df.index, opts) for c in val_cols]
        for fut in as_completed(futures):
            col, abs_err, error = fut.result()
            if abs_err is not None:
                abs_err_by_col[col] = abs_err
            if error:
                print(f"  [tabpfn] repli sur {col}: {error}", flush=True)
    return abs_err_by_col


# ---------------------------------------------------------------------------
# Côté SAITS "natif" : matrice brute (temps x courbes, NaN compris) +
# covariables temporelles globales + canaux auxiliaires optionnels par
# courbe. Pas de feature artisanale par colonne : SAITS doit retrouver la
# structure inter-courbes lui-même via l'attention sur l'axe features.
# ---------------------------------------------------------------------------

def select_saits_features(X_train, val_cols, donor_cols, max_features,
                            k_per_col=25, corr_threshold=0.15, euclid_weight=0.3):
    """Union, sur les val_cols du repeat, des donneurs les plus corrélés
    (feb.adaptive_top_k_corr_euclid -- même sélection que le KNN de TabPFN
    v14), plafonnée à max_features. Sélection = practicité mémoire (cf.
    consigne "teste d'abord sur 50-100 clients"), PAS une feature -- SAITS
    voit ensuite les courbes brutes, sans relation pré-calculée entre elles."""
    donor_vals = X_train[donor_cols].to_numpy(dtype=np.float64)
    chosen_idx, seen = [], set()
    for col in val_cols:
        s = X_train[col]
        obs_idx = np.flatnonzero(s.notna().to_numpy())
        if len(obs_idx) < 5:
            continue
        target_obs = s.to_numpy()[obs_idx]
        pool_obs = donor_vals[obs_idx, :]
        # quelques donneurs peuvent avoir un NaN résiduel pile sur les points
        # observés de la cible ; on les neutralise juste pour le calcul de
        # corrélation (sélection uniquement, la matrice finale garde les NaN bruts)
        pool_obs = np.where(np.isnan(pool_obs), np.nanmedian(pool_obs, axis=0, keepdims=True), pool_obs)
        idx, _ = feb.adaptive_top_k_corr_euclid(
            target_obs, pool_obs, k_min=5, k_max=k_per_col,
            corr_threshold=corr_threshold, euclid_weight=euclid_weight)
        for j in idx:
            if j not in seen:
                seen.add(j)
                chosen_idx.append(j)
        if len(chosen_idx) >= max_features:
            break
    return [donor_cols[j] for j in chosen_idx[:max_features]]


def global_time_covariates(index):
    """3 paires sin/cos (heure, jour de semaine, position dans l'année) +
    indicateur week-end -- feb.time_features_full_seasonality avec groups=
    ("hour","dow","year") calcule exactement ces 7 colonnes (le flag
    week-end est toujours ajouté par la fonction). PARTAGÉES par toutes les
    colonnes au même timestamp : ajoutées UNE SEULE FOIS à la matrice, pas
    par courbe."""
    return feb.time_features_full_seasonality(index, ("hour", "dow", "year"))


def per_curve_aux_channels(raw_curves):
    """Canaux auxiliaires PAR courbe (donc dupliqués, contrairement aux
    covariables temporelles) : volatilité locale (feb.curve_volatility, un
    scalaire par courbe diffusé sur tous les pas de temps) et accord
    périodique (feb.periodic_agreement, un score par pas de temps). Les deux
    restent des descripteurs DYNAMIQUES/de confiance dérivés de la courbe
    elle-même, pas un résumé figé du type "moyenne/écart-type du client"
    (explicitement à éviter : SAITS doit découvrir la structure
    inter-courbes lui-même via l'attention, pas la recevoir pré-résumée)."""
    T, F = raw_curves.shape
    vol_cols = np.zeros((T, F))
    agree_cols = np.zeros((T, F))
    for j in range(F):
        col = raw_curves[:, j]
        vol_cols[:, j] = feb.curve_volatility(col)
        agree_cols[:, j] = feb.periodic_agreement(col)
    return vol_cols, agree_cols


def robust_scale_matrix(mat):
    """Z-score robuste PAR COLONNE, ajusté uniquement sur les valeurs
    observées (nanmedian ignore les NaN) -- normalisation générique
    nécessaire avant tout réseau de neurones (les courbes vivent à des
    échelles de puissance très différentes), PAS une feature ajoutée : ne
    fait que remettre à l'échelle un canal déjà présent. Aucune fuite : ne
    regarde jamais y_train, seulement les points déjà observés de X_train."""
    center = np.nanmedian(mat, axis=0)
    mad = np.nanmedian(np.abs(mat - center), axis=0)
    scale = np.maximum(1.4826 * mad, 0.02 * (np.abs(center) + 1.0))
    scale = np.where((scale < 1e-6) | ~np.isfinite(scale), 1.0, scale)
    center = np.where(np.isfinite(center), center, 0.0)
    return (mat - center) / scale, center, scale


def build_windows(mat, window_len, stride):
    """Découpe l'axe temps en fenêtres chevauchantes (les "samples" attendus
    par PyPOTS) ; garde une dernière fenêtre alignée sur la fin de série même
    si le stride ne tombe pas juste, pour ne perdre aucun point."""
    T = mat.shape[0]
    window_len = min(window_len, T)
    starts = list(range(0, T - window_len + 1, stride))
    if not starts or starts[-1] + window_len < T:
        starts.append(T - window_len)
    starts = sorted(set(starts))
    windows = np.stack([mat[s:s + window_len] for s in starts], axis=0)
    return windows, starts


def reassemble(imputed_windows, starts, window_len, T, F):
    """Moyenne les prédictions des fenêtres qui se chevauchent pour
    reconstituer une seule matrice (T, F)."""
    out_sum = np.zeros((T, F))
    out_cnt = np.zeros((T, F))
    for w, s in zip(imputed_windows, starts):
        out_sum[s:s + window_len] += w
        out_cnt[s:s + window_len] += 1.0
    return out_sum / np.maximum(out_cnt, 1e-9)


# ---------------------------------------------------------------------------
# Visualisation entraînement / validation pendant l'entraînement SAITS.
#
# PyPOTS logue déjà "Epoch NNN - training loss (...): x, validation ...: y"
# à chaque epoch via son propre logger (verbose=True) -- mais ça suppose un
# val_set, qu'on ne construisait pas jusqu'ici. On (1) construit un vrai
# set de validation interne en cachant artificiellement une partie des
# points déjà OBSERVÉS de quelques fenêtres mises de côté (jamais les vrais
# trous des colonnes cibles, jamais y_train -- ça ne sert qu'à piloter
# l'entraînement, pas la métrique officielle), (2) on branche un handler de
# logging pour récupérer les pertes epoch par epoch, affichées EN DIRECT
# dans la console (verbose=True) ET tracées dans un PNG à la fin.
# ---------------------------------------------------------------------------

class _LossCapture(logging.Handler):
    """Intercepte les lignes "Epoch NNN - training loss (...): x[, validation
    ...: y]" que PyPOTS écrit sur son logger interne et récupère les valeurs
    numériques, sans dépendre d'attributs internes non documentés."""

    _PATTERN = re.compile(
        r"Epoch\s+(\d+)\s*-\s*training loss[^:]*:\s*([0-9.eE+-]+)"
        r"(?:,\s*validation[^:]*:\s*([0-9.eE+-]+))?"
    )

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.train_losses, self.val_losses = [], []

    def emit(self, record):
        m = self._PATTERN.search(record.getMessage())
        if m:
            self.train_losses.append(float(m.group(2)))
            self.val_losses.append(float(m.group(3)) if m.group(3) else None)


def make_internal_val_set(val_windows, hide_frac, seed):
    """Construit un val_set PyPOTS standard (X, X_ori, missing_mask) à
    partir de fenêtres mises de côté de l'entraînement : X_ori = fenêtres
    telles quelles (vrais trous compris), X = X_ori avec en plus
    `hide_frac` des points DÉJÀ OBSERVÉS masqués au hasard (MCAR). PyPOTS
    compare ensuite sa reconstruction à X_ori uniquement sur ces points
    additionnels -- jamais sur les vrais trous, dont on ne connaît pas la
    vérité ici (ce n'est pas la métrique officielle, juste un signal
    d'entraînement/early stopping)."""
    rng = np.random.RandomState(seed)
    X_ori = val_windows.copy()
    observed = ~np.isnan(X_ori)
    extra_hide = observed & (rng.random_sample(X_ori.shape) < hide_frac)
    X = X_ori.copy()
    X[extra_hide] = np.nan
    missing_mask = (~np.isnan(X)).astype(np.float32)
    return {"X": X, "X_ori": X_ori, "missing_mask": missing_mask}


def plot_training_curve(train_losses, val_losses, out_path, title):
    """Sauvegarde un PNG train/val loss vs epoch -- pas d'affichage
    interactif possible dans un script batch, donc on persiste la figure
    plutôt que de bloquer sur un plt.show()."""
    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, label="train loss (MAE)", color="#4C72B0", marker="o", ms=3)
    val_defined = [v for v in val_losses if v is not None]
    if val_defined:
        ax.plot(epochs[:len(val_defined)], val_defined, label="validation loss (MAE)",
                 color="#DD8452", marker="o", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def alpha_blend_post(s_series, miss_idx, saits_pred):
    """Post-traitement optionnel : blend alpha-interpolation de V14
    (continuous_alpha x periodic_agreement x volatility_alpha_mult, mêmes
    poids que V14_BEST) appliqué à la prédiction BRUTE de SAITS -- suggéré
    comme le bon usage du contexte de trou (durée, distance au bord) : en
    post-traitement/ensemble, pas comme feature d'entrée de SAITS."""
    missing = s_series.isna().to_numpy()
    runs = feb.run_lengths_for(missing)
    edges = feb.edge_distance_for(missing)
    alpha = np.array([feb.continuous_alpha(runs[i], edges[i], True) for i in miss_idx])
    p_agree = feb.periodic_agreement(s_series.to_numpy())[miss_idx]
    alpha *= 1.0 - PERIODIC_ALPHA_STRENGTH * p_agree
    vol = feb.curve_volatility(s_series.to_numpy())
    alpha *= feb.volatility_alpha_mult(vol, vol_weight=VOL_WEIGHT, cap=0.6)
    alpha = np.clip(alpha, 0.0, 0.95)
    interp = fill_nan_with_interpolation(s_series).to_numpy()[miss_idx]
    return alpha * interp + (1 - alpha) * saits_pred


def eval_saits_repeat(X_train, y_train, val_cols, donor_cols, opts, repeat_idx=0):
    from pypots.imputation import SAITS
    from pypots.utils.logging import logger as pypots_logger

    feat_cols = select_saits_features(X_train, val_cols, donor_cols, opts["max_features"])
    cols = list(val_cols) + feat_cols
    n_target = len(val_cols)

    # Entrée de base obligatoire : la matrice BRUTE, NaN compris (donneurs
    # non pré-remplis non plus -- SAITS gère tout le masque en interne).
    raw_curves = X_train[cols].to_numpy(dtype=np.float64)

    tcov = global_time_covariates(X_train.index)  # (T, 7), partagé une seule fois, jamais NaN

    if opts["aux_channels"]:
        vol_cols, agree_cols = per_curve_aux_channels(raw_curves)
        mat = np.hstack([raw_curves, tcov, vol_cols, agree_cols])
    else:
        mat = np.hstack([raw_curves, tcov])

    scaled, center, scale = robust_scale_matrix(mat)
    T, F = scaled.shape
    window_len = opts["window_len"]
    windows, starts = build_windows(scaled, window_len, opts["stride"])

    # --- Split train/val PAR FENÊTRE (aucun chevauchement), + masquage
    # artificiel supplémentaire côté val pour avoir un vrai signal de
    # validation à visualiser / utiliser pour l'early stopping. Purement
    # interne à l'entraînement -- n'entre jamais dans la métrique officielle.
    n_windows = len(windows)
    rng = np.random.RandomState(v14.RANDOM_STATE + repeat_idx)
    perm = rng.permutation(n_windows)
    n_val_windows = int(round(opts["val_frac"] * n_windows)) if n_windows >= 5 else 0
    val_win_idx, train_win_idx = perm[:n_val_windows], perm[n_val_windows:]
    train_set = {"X": windows[train_win_idx]}
    val_set = (make_internal_val_set(windows[val_win_idx], opts["val_hide_frac"], v14.RANDOM_STATE + repeat_idx)
               if n_val_windows > 0 else None)
    patience = opts["patience"] if val_set is not None else None
    if val_set is None:
        print(f"  [saits] repeat {repeat_idx + 1}: pas assez de fenêtres ({n_windows}) pour un val_set interne, "
              f"pas d'early stopping ni de courbe de validation", flush=True)

    saits = SAITS(
        n_steps=window_len, n_features=F, n_layers=opts["n_layers"],
        d_model=opts["d_model"], n_heads=opts["n_heads"], d_k=opts["d_k"], d_v=opts["d_v"],
        d_ffn=opts["d_ffn"], dropout=opts["dropout"], epochs=opts["epochs"],
        patience=patience, batch_size=min(opts["batch_size"], len(train_win_idx)),
        device=opts["device"], verbose=True,  # imprime "Epoch NNN - training loss: x[, validation: y]" en direct
    )

    capture = _LossCapture()
    pypots_logger.addHandler(capture)
    try:
        saits.fit(train_set, val_set)
    finally:
        pypots_logger.removeHandler(capture)

    if opts.get("plot_training") and capture.train_losses:
        tag = f"repeat{repeat_idx + 1}_{PROFILE}"
        out_path = os.path.join(PLOTS_DIR, f"saits_v2_{tag}.png")
        plot_training_curve(capture.train_losses, capture.val_losses, out_path,
                             title=f"SAITS training curve -- {tag} (F={F}, n_windows={n_windows})")
        n_ep = len(capture.train_losses)
        last_val = next((v for v in reversed(capture.val_losses) if v is not None), None)
        print(f"  [saits] repeat {repeat_idx + 1}: {n_ep} epochs, "
              f"train_loss final={capture.train_losses[-1]:.4f}"
              + (f", val_loss final={last_val:.4f}" if last_val is not None else "")
              + f" -> {out_path}", flush=True)

    imputed_windows = saits.predict({"X": windows})["imputation"]
    full_scaled = reassemble(imputed_windows, starts, window_len, T, F)
    full = full_scaled * scale + center  # retour à l'échelle d'origine (Watts) -- seules les n_target 1eres colonnes nous intéressent

    raw_err_by_col, blend_err_by_col = {}, {}
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
    return raw_err_by_col, blend_err_by_col


# ---------------------------------------------------------------------------
# Boucle 10x10 appariée + rapport (TabPFN v14 / SAITS brut / SAITS + blend)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant (export TABPFN_API_KEY=...)")
    tabpfn_client.set_access_token(token)

    try:
        import pypots  # noqa: F401
    except ImportError:
        raise SystemExit("pypots manquant : .venv/bin/pip install pypots")

    print(f"=== profil={PROFILE}  n_repeats={N_REPEATS}  saits_opts={SAITS_OPTS} ===", flush=True)
    print("Chargement des données...", flush=True)
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]
    holed_cols = list(y_train.columns)

    progress = json.load(open(PROGRESS_PATH)) if os.path.exists(PROGRESS_PATH) else {}

    for i in range(N_REPEATS):
        if str(i) in progress:
            print(f"repeat {i + 1}/{N_REPEATS} déjà fait, skip", flush=True)
            continue
        seed_i = v14.RANDOM_STATE + i
        dev_cols, val_cols_full = train_test_split(holed_cols, test_size=.1, random_state=seed_i)
        rng = np.random.RandomState(seed_i)
        val_cols = list(rng.choice(val_cols_full, size=min(N_VAL_SUB, len(val_cols_full)), replace=False))

        t0 = time.time()
        tabpfn_err = eval_tabpfn_repeat(X_train, y_train, donor_cols, dev_cols, val_cols, V14_BEST)
        print(f"  [repeat {i + 1}] tabpfn v14 : {time.time() - t0:.1f}s", flush=True)

        t0 = time.time()
        saits_raw_err, saits_blend_err = eval_saits_repeat(X_train, y_train, val_cols, donor_cols, SAITS_OPTS,
                                                             repeat_idx=i)
        print(f"  [repeat {i + 1}] saits      : {time.time() - t0:.1f}s", flush=True)

        progress[str(i)] = {
            "val_cols": val_cols,
            "tabpfn": {c: e.tolist() for c, e in tabpfn_err.items()},
            "saits_raw": {c: e.tolist() for c, e in saits_raw_err.items()},
            "saits_blend": {c: e.tolist() for c, e in saits_blend_err.items()},
        }
        json.dump(progress, open(PROGRESS_PATH, "w"))
        print(f"repeat {i + 1}/{N_REPEATS} sauvegardé.\n", flush=True)

    # --- Rapport apparié (repeat par repeat ET colonne par colonne) ---
    rows = []
    all_tabpfn, all_saits_raw, all_saits_blend = [], [], []
    wins = {"saits_blend": 0, "tabpfn": 0, "tie": 0}
    for i in range(N_REPEATS):
        rec = progress[str(i)]
        tb = {c: np.array(v) for c, v in rec["tabpfn"].items()}
        sr = {c: np.array(v) for c, v in rec["saits_raw"].items()}
        sb = {c: np.array(v) for c, v in rec["saits_blend"].items()}
        common = [c for c in rec["val_cols"] if c in tb and c in sr and c in sb]
        if not common:
            continue
        tb_cat = np.concatenate([tb[c] for c in common])
        sr_cat = np.concatenate([sr[c] for c in common])
        sb_cat = np.concatenate([sb[c] for c in common])
        all_tabpfn.append(tb_cat)
        all_saits_raw.append(sr_cat)
        all_saits_blend.append(sb_cat)
        for c in common:
            m_t, m_b = tb[c].mean(), sb[c].mean()
            if m_b < m_t - 1e-9:
                wins["saits_blend"] += 1
            elif m_t < m_b - 1e-9:
                wins["tabpfn"] += 1
            else:
                wins["tie"] += 1
        rows.append((i + 1, float(tb_cat.mean()), float(sr_cat.mean()), float(sb_cat.mean())))

    print("\n| Repeat | TabPFN v14 | SAITS brut | SAITS+blend | Δ (blend-tabpfn) |")
    print("|---:|---:|---:|---:|---:|")
    for r, t, sr, sb in rows:
        print(f"| {r} | {t:.2f} | {sr:.2f} | {sb:.2f} | {sb - t:+.2f} |")

    mae_tabpfn = float(np.concatenate(all_tabpfn).mean()) if all_tabpfn else float("nan")
    mae_saits_raw = float(np.concatenate(all_saits_raw).mean()) if all_saits_raw else float("nan")
    mae_saits_blend = float(np.concatenate(all_saits_blend).mean()) if all_saits_blend else float("nan")
    print(f"\nMAE TabPFN v14 (pool):    {mae_tabpfn:.2f} W")
    print(f"MAE SAITS brut (pool):    {mae_saits_raw:.2f} W")
    print(f"MAE SAITS + blend (pool): {mae_saits_blend:.2f} W")
    if mae_tabpfn == mae_tabpfn:  # not NaN
        print(f"gain relatif SAITS+blend vs TabPFN v14: {100 * (1 - mae_saits_blend / mae_tabpfn):.1f}%")
    print(f"colonnes gagnées par SAITS+blend: {wins['saits_blend']} / par TabPFN v14: {wins['tabpfn']} "
          f"/ égalité: {wins['tie']} / total: {sum(wins.values())}")

    out = {
        "profile": PROFILE, "saits_opts": SAITS_OPTS,
        "rows": [{"repeat": r, "tabpfn": t, "saits_raw": sr, "saits_blend": sb, "delta": sb - t}
                 for r, t, sr, sb in rows],
        "mae_tabpfn": mae_tabpfn, "mae_saits_raw": mae_saits_raw, "mae_saits_blend": mae_saits_blend,
        "wins_saits_blend": wins["saits_blend"], "wins_tabpfn": wins["tabpfn"], "ties": wins["tie"],
    }
    json.dump(out, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nrésultats écrits dans {RESULTS_PATH}", flush=True)
