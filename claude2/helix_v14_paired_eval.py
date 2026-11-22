"""HELIX (au lieu de M-SAITS, indisponible) vs TabPFN v14, protocole officiel
10x10 -- descend de `saits_v14_paired_eval_v2.py` (entrée "native" matrice
brute + covariables temporelles + canaux auxiliaires + blend post-traitement,
voir ce fichier pour le détail de ce design), backbone remplacé par HELIX.

Pourquoi HELIX et pas M-SAITS
-----------------------------
M-SAITS ("A Dual-Stage Time Series Imputation Network via Decoupled
Large-Kernel Convolution and Diagonally-Masked Attention", Su/Wang/Bai/Wan,
*ICCK Transactions on Machine Intelligence*, 8 mars 2026) est un vrai papier,
mais **sans code public** : ni dépôt GitHub, ni intégration PyPOTS (sa
"Data Availability Statement" dit juste "disponible sur demande"). Impossible
de faire un `from pypots.imputation import MSAITS` -- ça n'existe nulle part.

À la place (choix validé avec l'utilisateur) : **HELIX** ("Hybrid Encoding
with Learnable Identity and Cross-dimensional Synthesis for Time Series
Imputation", Zhang/Du/Zhang/Yu/Qu, **ICML 2026 Spotlight**) --
- explicitement conçu pour l'IMPUTATION de séries temporelles (pas une
  architecture générique de forecasting adaptée après coup) ;
- même auteur principal (Wenjie Du) que SAITS et que PyPOTS lui-même, donc
  même lignée et une vraie intégration officielle dans la librairie (pas un
  contributeur tiers) ;
- **1er sur 17 méthodes, 21 configurations expérimentales** -- SOTA vérifié,
  pas juste "récent" ;
- va plus loin que la simple attention DMSA de SAITS : chaque courbe reçoit
  une identité apprise (embedding persistant, `d_feature_embed`), combinée à
  une attention hybride temporelle/features -- c'est-à-dire qu'il apprend les
  dépendances inter-courbes *sans graphe/relation prédéfinie*, exactement le
  principe qu'on voulait déjà pour SAITS (cf. V2 : pas de feature artisanale
  par colonne, laisser le modèle découvrir la structure inter-courbes tout
  seul) -- HELIX le fait plus explicitement encore.

Tout le reste du design V2 est inchangé (voir `saits_v14_paired_eval_v2.py`
pour la justification détaillée de chaque point) :
- entrée = matrice brute (courbes + NaN réels), aucun ffill/bfill des
  donneurs, HELIX gère le masque en interne ;
- 7 covariables temporelles globales (heure/jour de semaine/position année
  en sin/cos + week-end), ajoutées UNE FOIS, partagées par toutes les
  courbes ;
- 2 canaux auxiliaires PAR courbe (volatilité locale, accord périodique) --
  signal dynamique, pas un résumé figé "moyenne/écart-type du client" ;
- contexte de trou (durée, bord) gardé pour le POST-traitement (blend alpha
  V14 sur la sortie brute du modèle), pas comme feature d'entrée ;
- 50-100 courbes les plus corrélées seulement (mémoire de l'attention sur
  l'axe features) ;
- set de validation interne (fenêtres mises de côté + masquage MCAR
  supplémentaire) + capture des logs PyPOTS + PNG train/val loss par repeat.

Protocole (inchangé, strictement comparable à tabpfn_v14_confirm.py,
66,24 +/- 32,45 W) : mêmes seeds 42+i, mêmes val_cols par repeat, TabPFN et
HELIX évalués sur les mêmes colonnes -> comparaison appariée repeat par
repeat ET colonne par colonne.

Installation (important)
-------------------------
HELIX vient d'être fusionné dans PyPOTS et sera dans la release officielle
PyPOTS v1.5 -- pas encore sur PyPI au moment d'écrire ce script. Installer
depuis la branche main :

    .venv/bin/pip install "git+https://github.com/WenjieDu/PyPOTS.git"

Usage
-----
    HELIX_PROFILE=smoke .venv/bin/python claude2/helix_v14_paired_eval.py
    HELIX_PROFILE=full  .venv/bin/python claude2/helix_v14_paired_eval.py

Resumable comme le reste du repo (progress JSON par repeat).
`saits_v14_paired_eval_v2.py` (backbone SAITS) reste intact à côté, au cas
où HELIX se révèle indisponible/instable en pratique.
"""
import os
import re
import sys
import gc
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

PROGRESS_PATH = os.path.join(HERE, "helix_v14_paired_progress.json")
RESULTS_PATH = os.path.join(HERE, "helix_v14_paired_results.json")
PLOTS_DIR = os.path.join(HERE, "helix_training_plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Config TabPFN v14 = BEST confirmée (tabpfn_v14_confirm.py, 66,24 +/- 32,45 W)
V14_BEST = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                 extra_lags=(24,), neighbor_rolling=True)
# Mêmes poids, réutilisés pour le blend post-traitement côté HELIX.
VOL_WEIGHT = V14_BEST["vol_weight"]
PERIODIC_ALPHA_STRENGTH = V14_BEST["periodic_alpha_strength"]

PROFILE = os.environ.get("HELIX_PROFILE", "smoke")
N_REPEATS = int(os.environ.get("HELIX_N_REPEATS", 2 if PROFILE == "smoke" else 10))
N_VAL_SUB = 10  # colonnes de validation par repeat, comme run_config(n_val_sub=10)

# val_frac : fraction des FENÊTRES (pas des points) mises de côté pour un
#   set de validation interne PyPOTS -- entièrement séparé des fenêtres
#   d'entraînement (aucun chevauchement de fenêtre entre train et val).
# val_hide_frac : sur ces fenêtres de validation, fraction des points DÉJÀ
#   OBSERVÉS masqués en plus (artificiellement) pour avoir une vérité connue
#   à comparer -- jamais les vrais trous des colonnes cibles ni y_train,
#   uniquement pour piloter l'early stopping et visualiser la courbe train/
#   val (la métrique officielle reste calculée séparément, cf. plus bas).
# d_pe / d_feature_embed : spécifiques à HELIX (positional encoding dim et
#   dimension de l'identité apprise par courbe) -- valeurs par défaut de la
#   librairie, à augmenter (surtout d_feature_embed, coeur de l'idée HELIX)
#   si les résultats déçoivent et que le budget le permet.
_SMOKE = dict(
    max_features=20, aux_channels=True, window_len=96, stride=48,
    n_layers=1, d_model=64, n_heads=2, d_pe=16, d_feature_embed=1, dropout=0.1,
    epochs=30, patience=8, batch_size=8, device=None,
    val_frac=0.2, val_hide_frac=0.1, plot_training=True,
)
_FULL = dict(
    # 50-100 courbes corrélées comme recommandé ; avec les canaux auxiliaires
    # (2 par courbe) + 7 covariables temporelles globales, F ~ (10+90)*3+7
    # ~= 307.
    max_features=90, aux_channels=True, window_len=336, stride=48,
    n_layers=2, d_model=256, n_heads=4, d_pe=16, d_feature_embed=1, dropout=0.1,
    epochs=200, patience=20, batch_size=16, device=None,
    val_frac=0.15, val_hide_frac=0.1, plot_training=True,
)
HELIX_OPTS = _SMOKE if PROFILE == "smoke" else _FULL


# ---------------------------------------------------------------------------
# Côté TabPFN v14 : inchangé (copie de tabpfn_v14_ablation.eval_split,
# erreurs absolues PAR COLONNE pour permettre l'appariement colonne-par-
# colonne avec HELIX).
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
# Côté HELIX "natif" : matrice brute (temps x courbes, NaN compris) +
# covariables temporelles globales + canaux auxiliaires optionnels par
# courbe. Pas de feature artisanale par colonne : HELIX retrouve la
# structure inter-courbes lui-même (identité apprise par courbe + attention
# hybride temps/features).
# ---------------------------------------------------------------------------

def select_donor_features(X_train, val_cols, donor_cols, max_features,
                            k_per_col=25, corr_threshold=0.15, euclid_weight=0.3):
    """Union, sur les val_cols du repeat, des donneurs les plus corrélés
    (feb.adaptive_top_k_corr_euclid -- même sélection que le KNN de TabPFN
    v14), plafonnée à max_features. Sélection = practicité mémoire (cf.
    consigne "teste d'abord sur 50-100 clients"), PAS une feature -- HELIX
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
    (explicitement à éviter : HELIX doit découvrir la structure
    inter-courbes lui-même via l'identité apprise + l'attention, pas la
    recevoir pré-résumée)."""
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
# Visualisation entraînement / validation pendant l'entraînement HELIX.
#
# PyPOTS logue déjà "Epoch NNN - training loss (...): x, validation ...: y"
# à chaque epoch via son propre logger (verbose=True) -- mais ça suppose un
# val_set. On (1) construit un vrai set de validation interne en cachant
# artificiellement une partie des points déjà OBSERVÉS de quelques fenêtres
# mises de côté (jamais les vrais trous des colonnes cibles, jamais y_train
# -- ça ne sert qu'à piloter l'entraînement, pas la métrique officielle),
# (2) on branche un handler de logging pour récupérer les pertes epoch par
# epoch, affichées EN DIRECT dans la console (verbose=True) ET tracées dans
# un PNG à la fin.
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


def alpha_blend_post(s_series, miss_idx, pred):
    """Post-traitement optionnel : blend alpha-interpolation de V14
    (continuous_alpha x periodic_agreement x volatility_alpha_mult, mêmes
    poids que V14_BEST) appliqué à la prédiction BRUTE de HELIX -- suggéré
    comme le bon usage du contexte de trou (durée, distance au bord) : en
    post-traitement/ensemble, pas comme feature d'entrée du modèle."""
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
    return alpha * interp + (1 - alpha) * pred


def eval_helix_repeat(X_train, y_train, val_cols, donor_cols, opts, repeat_idx=0):
    import torch
    from pypots.imputation import HELIX
    from pypots.nn.modules.loss import MAE as MAECriterion
    from pypots.utils.logging import logger as pypots_logger

    feat_cols = select_donor_features(X_train, val_cols, donor_cols, opts["max_features"])
    cols = list(val_cols) + feat_cols
    n_target = len(val_cols)

    # Entrée de base obligatoire : la matrice BRUTE, NaN compris (donneurs
    # non pré-remplis non plus -- HELIX gère tout le masque en interne).
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
        print(f"  [helix] repeat {repeat_idx + 1}: pas assez de fenêtres ({n_windows}) pour un val_set interne, "
              f"pas d'early stopping ni de courbe de validation", flush=True)

    # helix initialisé à None avant le try : garantit que le `finally` peut
    # toujours faire `del helix` (et donc libérer son état CUDA -- modèle,
    # optimiseur, buffers d'activation résiduels) que la construction/le fit
    # réussisse ou non. torch.cuda.empty_cache() SEUL ne suffit pas : il ne
    # rend que la mémoire déjà libérée par Python, pas celle encore
    # référencée par l'objet -- sans ce `del` + gc.collect() explicites, la
    # mémoire GPU d'un repeat s'accumule sur le suivant (observé en pratique :
    # un repeat qui passait seul se met à OOM au repeat suivant, à config
    # strictement identique).
    helix = None
    imputed_windows = None
    try:
        helix = HELIX(
            n_steps=window_len, n_features=F, d_pe=opts["d_pe"], d_feature_embed=opts["d_feature_embed"],
            d_model=opts["d_model"], n_heads=opts["n_heads"], n_layers=opts["n_layers"], dropout=opts["dropout"],
            epochs=opts["epochs"], patience=patience, batch_size=min(opts["batch_size"], len(train_win_idx)),
            device=opts["device"],
            # Défauts PyPOTS : training_loss=MAE (déjà aligné), validation_metric=MSE
            # (PAS aligné avec la métrique officielle du challenge) -- on force les
            # deux en MAE pour que l'early stopping (patience) et la courbe train/
            # val tracée plus bas comparent vraiment la même unité.
            training_loss=MAECriterion, validation_metric=MAECriterion,
            verbose=True,  # imprime "Epoch NNN - training loss: x[, validation: y]" en direct
        )

        capture = _LossCapture()
        pypots_logger.addHandler(capture)
        try:
            helix.fit(train_set, val_set)
        finally:
            pypots_logger.removeHandler(capture)

        if opts.get("plot_training") and capture.train_losses:
            tag = f"repeat{repeat_idx + 1}_{PROFILE}"
            out_path = os.path.join(PLOTS_DIR, f"helix_{tag}.png")
            plot_training_curve(capture.train_losses, capture.val_losses, out_path,
                                 title=f"HELIX training curve -- {tag} (F={F}, n_windows={n_windows})")
            n_ep = len(capture.train_losses)
            last_val = next((v for v in reversed(capture.val_losses) if v is not None), None)
            print(f"  [helix] repeat {repeat_idx + 1}: {n_ep} epochs, "
                  f"train_loss final={capture.train_losses[-1]:.4f}"
                  + (f", val_loss final={last_val:.4f}" if last_val is not None else "")
                  + f" -> {out_path}", flush=True)

        # .impute() = raccourci officiel PyPOTS (fit -> predict -> extrait la
        # clé "imputation"), utilisé tel quel dans l'exemple HELIX du dépôt.
        imputed_windows = helix.impute({"X": windows})
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"  [helix] repeat {repeat_idx + 1}: CUDA OOM avec F={F}, window_len={window_len}, "
                  f"batch_size={opts['batch_size']}. window_len est le facteur QUADRATIQUE dominant -- "
                  f"réduis-le en premier, puis max_features, puis batch_size, et relance -- ce repeat "
                  f"n'est PAS marqué comme fait, la progression précédente est conservée.", flush=True)
            return None, None
        raise
    finally:
        del helix
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if imputed_windows is None:
        return None, None
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
# Boucle 10x10 appariée + rapport (TabPFN v14 / HELIX brut / HELIX + blend)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant (export TABPFN_API_KEY=...)")
    tabpfn_client.set_access_token(token)

    try:
        import pypots  # noqa: F401
    except ImportError:
        raise SystemExit(
            "pypots manquant : .venv/bin/pip install \"git+https://github.com/WenjieDu/PyPOTS.git\"")
    try:
        from pypots.imputation import HELIX  # noqa: F401
    except ImportError:
        raise SystemExit(
            "HELIX indisponible dans cette version de pypots (fusionné sur la branche main, pas encore "
            "publié sur PyPI en tant que release v1.5) : .venv/bin/pip install --upgrade --force-reinstall "
            "\"git+https://github.com/WenjieDu/PyPOTS.git\"")

    print(f"=== profil={PROFILE}  n_repeats={N_REPEATS}  helix_opts={HELIX_OPTS} ===", flush=True)
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
        helix_raw_err, helix_blend_err = eval_helix_repeat(X_train, y_train, val_cols, donor_cols, HELIX_OPTS,
                                                             repeat_idx=i)
        print(f"  [repeat {i + 1}] helix      : {time.time() - t0:.1f}s", flush=True)

        if helix_raw_err is None:
            print(f"repeat {i + 1}/{N_REPEATS} NON sauvegardé (OOM) -- ajuste la config et relance.\n", flush=True)
            continue

        progress[str(i)] = {
            "val_cols": val_cols,
            "tabpfn": {c: e.tolist() for c, e in tabpfn_err.items()},
            "helix_raw": {c: e.tolist() for c, e in helix_raw_err.items()},
            "helix_blend": {c: e.tolist() for c, e in helix_blend_err.items()},
        }
        json.dump(progress, open(PROGRESS_PATH, "w"))
        print(f"repeat {i + 1}/{N_REPEATS} sauvegardé.\n", flush=True)

    # --- Rapport apparié (repeat par repeat ET colonne par colonne) ---
    rows = []
    all_tabpfn, all_helix_raw, all_helix_blend = [], [], []
    wins = {"helix_blend": 0, "tabpfn": 0, "tie": 0}
    for i in range(N_REPEATS):
        rec = progress[str(i)]
        tb = {c: np.array(v) for c, v in rec["tabpfn"].items()}
        hr = {c: np.array(v) for c, v in rec["helix_raw"].items()}
        hb = {c: np.array(v) for c, v in rec["helix_blend"].items()}
        common = [c for c in rec["val_cols"] if c in tb and c in hr and c in hb]
        if not common:
            continue
        tb_cat = np.concatenate([tb[c] for c in common])
        hr_cat = np.concatenate([hr[c] for c in common])
        hb_cat = np.concatenate([hb[c] for c in common])
        all_tabpfn.append(tb_cat)
        all_helix_raw.append(hr_cat)
        all_helix_blend.append(hb_cat)
        for c in common:
            m_t, m_b = tb[c].mean(), hb[c].mean()
            if m_b < m_t - 1e-9:
                wins["helix_blend"] += 1
            elif m_t < m_b - 1e-9:
                wins["tabpfn"] += 1
            else:
                wins["tie"] += 1
        rows.append((i + 1, float(tb_cat.mean()), float(hr_cat.mean()), float(hb_cat.mean())))

    print("\n| Repeat | TabPFN v14 | HELIX brut | HELIX+blend | Δ (blend-tabpfn) |")
    print("|---:|---:|---:|---:|---:|")
    for r, t, hr, hb in rows:
        print(f"| {r} | {t:.2f} | {hr:.2f} | {hb:.2f} | {hb - t:+.2f} |")

    mae_tabpfn = float(np.concatenate(all_tabpfn).mean()) if all_tabpfn else float("nan")
    mae_helix_raw = float(np.concatenate(all_helix_raw).mean()) if all_helix_raw else float("nan")
    mae_helix_blend = float(np.concatenate(all_helix_blend).mean()) if all_helix_blend else float("nan")
    print(f"\nMAE TabPFN v14 (pool):    {mae_tabpfn:.2f} W")
    print(f"MAE HELIX brut (pool):    {mae_helix_raw:.2f} W")
    print(f"MAE HELIX + blend (pool): {mae_helix_blend:.2f} W")
    if mae_tabpfn == mae_tabpfn:  # not NaN
        print(f"gain relatif HELIX+blend vs TabPFN v14: {100 * (1 - mae_helix_blend / mae_tabpfn):.1f}%")
    print(f"colonnes gagnées par HELIX+blend: {wins['helix_blend']} / par TabPFN v14: {wins['tabpfn']} "
          f"/ égalité: {wins['tie']} / total: {sum(wins.values())}")

    out = {
        "profile": PROFILE, "helix_opts": HELIX_OPTS,
        "rows": [{"repeat": r, "tabpfn": t, "helix_raw": hr, "helix_blend": hb, "delta": hb - t}
                 for r, t, hr, hb in rows],
        "mae_tabpfn": mae_tabpfn, "mae_helix_raw": mae_helix_raw, "mae_helix_blend": mae_helix_blend,
        "wins_helix_blend": wins["helix_blend"], "wins_tabpfn": wins["tabpfn"], "ties": wins["tie"],
    }
    json.dump(out, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nrésultats écrits dans {RESULTS_PATH}", flush=True)
