"""Idées issues de la recherche bibliographique (voir benchmark.md § "Round 7").

Modèle dérivé de `feature_engineering_bench.make_v10_model` (configuration V12
= BEST_FEATURES), avec 4 options nouvelles activables indépendamment pour
ablation honnête. Quand toutes les options sont désactivées, le chemin de
calcul est identique à `make_v10_model` (mêmes appels, même ordre) -> le
score doit reproduire la baseline R0_lightgbm_reference.

A1 `n_aggregate` : agrégats de N voisins (N >> k) au lieu de N colonnes brutes.
A2 `pca_rank`    : complétion bas-rang par projection sur la base ACP du pool.
A3 `gap_window`  : sélection des donneurs restreinte au voisinage des trous.
A4 `n_passes`    : imputation itérative type MICE (features recalculées).
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

from evaluate import fill_nan_with_interpolation
import feature_engineering_bench as feb


# ---------------------------------------------------------------------------
# A1 : agrégats de voisins
# ---------------------------------------------------------------------------

def neighbor_aggregate_features(scaled):
    """Résume N voisins déjà remis à l'échelle en 7 statistiques par instant.

    L'expérience `catboost_bigk_bench.py` a montré qu'empiler 10 000 colonnes
    de voisins brutes n'apporte rien (-0,65 W, porté par un seul repeat) : le
    signal utile des voisins lointains est noyé. La littérature "nearest
    neighbour matrix completion" (arXiv 2506.04166) agrège au contraire les
    voisins ; on garde donc les k meilleurs bruts + un résumé robuste d'un
    pool bien plus large.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q = np.nanquantile(scaled, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T
        mean = np.nanmean(scaled, axis=1)[:, None]
        std = np.nanstd(scaled, axis=1)[:, None]
    return np.nan_to_num(np.hstack([q, mean, std]))


# ---------------------------------------------------------------------------
# A2 : complétion bas-rang par ACP du pool de donneurs
# ---------------------------------------------------------------------------

def fit_pool_pca(pool_vals, rank, n_sample=3000, random_state=feb.RANDOM_STATE):
    """Base ACP des FORMES de courbes du pool (chaque courbe = un vecteur de
    1057 instants, normalisé médiane/MAD pour que la base capture la forme et
    non le niveau). Appris une seule fois par repeat sur un échantillon de
    courbes, ce qui garde le coût mémoire raisonnable."""
    rng = np.random.RandomState(random_state)
    n_curves = pool_vals.shape[1]
    idx = rng.choice(n_curves, size=min(n_sample, n_curves), replace=False)
    sub = pool_vals[:, idx].T.astype(np.float32)          # (n_sample, n_times)
    center = np.median(sub, axis=1, keepdims=True)
    scale = np.maximum(1.4826 * np.median(np.abs(sub - center), axis=1, keepdims=True),
                       0.02 * (np.abs(center) + 1.0))
    pca = PCA(n_components=rank, random_state=random_state)
    pca.fit((sub - center) / scale)
    return pca


def pca_completion(pca, values, obs_idx, ridge_alpha=1.0):
    """Projette la partie OBSERVEE de la courbe sur la base ACP (moindres
    carrés régularisés sur les seuls instants observés) puis reconstruit la
    courbe entière. C'est de la complétion bas-rang classique, mais la base
    est apprise sur 20 000 courbes complètes au lieu d'être estimée sur la
    matrice trouée elle-même (ce qui avait échoué : SoftImpute/NMF 84,04 W)."""
    target_obs = values[obs_idx]
    center = np.median(target_obs)
    scale = max(1.4826 * np.median(np.abs(target_obs - center)), 0.02 * (abs(center) + 1.0))
    z_obs = (target_obs - center) / scale

    basis = pca.components_[:, obs_idx].T                  # (n_obs, rank)
    mean_obs = pca.mean_[obs_idx]
    reg = Ridge(alpha=ridge_alpha, fit_intercept=False)
    reg.fit(basis, z_obs - mean_obs)
    z_full = pca.mean_ + pca.components_.T @ reg.coef_
    return z_full * scale + center


# ---------------------------------------------------------------------------
# A3 : sélection des donneurs restreinte au voisinage des trous
# ---------------------------------------------------------------------------

def gap_neighborhood_mask(missing, window):
    """Instants observés situés à moins de `window` pas d'un trou. Le
    classement des donneurs sur ces seuls instants privilégie ceux qui
    ressemblent à la cible LA OU on doit prédire, plutôt qu'en moyenne sur
    trois semaines (les corrélations sont faibles globalement mais fortes
    localement, cf. benchmark.md § 1)."""
    n = len(missing)
    near = np.zeros(n, dtype=bool)
    miss_pos = np.where(missing)[0]
    for p in miss_pos:
        near[max(0, p - window):min(n, p + window + 1)] = True
    return near & ~missing


def select_donors_local(pool_vals, obs_idx, rank_idx, target_obs, target_rank,
                        k, corr_threshold=0.15, euclid_weight=0.3, k_min=5):
    """Comme `adaptive_top_k_corr_euclid` mais le CLASSEMENT utilise
    `rank_idx` (voisinage des trous) tandis que la mise à l'échelle et la
    prédiction restent calculées sur tous les instants observés."""
    t_c = target_rank - target_rank.mean()
    d = pool_vals[rank_idx, :]
    d_c = d - d.mean(axis=0, keepdims=True)
    num = d_c.T @ t_c
    denom = np.sqrt((d_c ** 2).sum(axis=0) * (t_c ** 2).sum()) + 1e-9
    corr = num / denom
    t_scale = target_rank.std() + 1e-9
    euclid = np.sqrt(((d_c - t_c[:, None]) ** 2).mean(axis=0)) / t_scale
    combined = (1 - euclid_weight) * corr + euclid_weight * (1.0 / (1.0 + euclid))
    order = np.argsort(-combined)
    above = order[corr[order] >= corr_threshold]
    chosen = above[:k] if len(above) >= k_min else order[:k_min]
    return chosen, corr[chosen]


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

BEST_FEATURES = dict(
    use_seasonality_feat=True, use_full_calendar_feat=True,
    calendar_groups=("hour", "dow", "dom", "year"),
    use_periodic_copy_feat=True, use_periodic_consensus_feat=True,
    normalize_periodic_feat=True, include_periodic_availability=True,
    periodic_alpha_strength=.5,
)


def make_research_model(k=40, n_clusters=15, make_regressor=None,
                        random_state=feb.RANDOM_STATE,
                        n_aggregate=0, pca_rank=0, gap_window=0, n_passes=1,
                        **v12_kwargs):
    """V12 + les 4 idées, chacune neutre par défaut."""
    if make_regressor is None:
        make_regressor = lambda: Ridge(alpha=1.0)
    opts = dict(BEST_FEATURES)
    opts.update(v12_kwargs)

    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        pool_vals = pool.ffill().bfill().values.astype(np.float64)

        km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=n_clusters,
                                                  random_state=random_state)
        if dev_true_df.shape[1] > 0:
            dev_filled = dev_true_df.ffill().bfill()
            Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
            dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
            pool_labels = pd.concat([donor_labels, dev_labels])
        else:
            pool_labels = donor_labels
        pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
        cluster_col_idx = {cl: pos[idx].values
                           for cl, idx in pool_labels.groupby(pool_labels).groups.items()}

        tfeat = feb.time_features_full_seasonality(val_holed_df.index,
                                                   opts["calendar_groups"])
        pca = fit_pool_pca(pool_vals, pca_rank, random_state=random_state) if pca_rank else None

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
            cl = feb.cluster_of(interp_full, km)
            same_idx = cluster_col_idx.get(cl)
            if same_idx is not None and len(same_idx) >= max(2 * k, 30):
                sub_vals = pool_vals[:, same_idx]
            else:
                sub_vals = pool_vals

            # --- voisins (A3 : classement local aux trous) ---------------
            if gap_window:
                rank_mask = gap_neighborhood_mask(missing, gap_window)
                rank_idx = np.where(rank_mask)[0]
                if len(rank_idx) >= 10:
                    top_idx, corr = select_donors_local(
                        sub_vals, obs_idx, rank_idx, target_obs, s.values[rank_idx], k)
                    w = np.clip(corr, 1e-3, None)
                    w = w / w.sum()
                    scaled = feb.scale_neighbors(sub_vals[:, top_idx], obs_idx, target_obs, True)
                    knn_pred = (scaled[miss_idx, :] * w).sum(axis=1)
                else:
                    knn_pred, top_idx, scaled = feb.weighted_knn_pred(
                        sub_vals, obs_idx, miss_idx, target_obs, k, robust=True,
                        adaptive_k=True, use_euclid=True)
            else:
                knn_pred, top_idx, scaled = feb.weighted_knn_pred(
                    sub_vals, obs_idx, miss_idx, target_obs, k, robust=True,
                    adaptive_k=True, use_euclid=True)

            # --- A1 : agrégats d'un pool de voisins plus large ------------
            agg = None
            if n_aggregate:
                _, wide_idx, wide_scaled = feb.weighted_knn_pred(
                    sub_vals, obs_idx, miss_idx, target_obs, n_aggregate,
                    robust=True, adaptive_k=True, use_euclid=True)
                agg = neighbor_aggregate_features(wide_scaled)

            # --- A2 : reconstruction bas-rang ACP -------------------------
            pca_rec = pca_completion(pca, s.values, obs_idx).reshape(-1, 1) if pca else None

            values_for_feats = s.values
            pred = None
            for p in range(max(1, n_passes)):
                extra = np.hstack([
                    tfeat,
                    feb.same_curve_seasonality_feature(values_for_feats, val_holed_df.index),
                    feb.periodic_copy_features(
                        values_for_feats, include_summary=True, normalize=True,
                        include_availability=True),
                ])
                if agg is not None:
                    extra = np.hstack([extra, agg])
                if pca_rec is not None:
                    extra = np.hstack([extra, pca_rec])

                if len(obs_idx) >= 15:
                    pred_neighbors = feb.local_regression_pred(
                        make_regressor, scaled, obs_idx, miss_idx, target_obs, extra)
                else:
                    pred_neighbors = knn_pred

                interp = interp_full.values[miss_idx]
                run_len_full = feb.run_lengths_for(missing)
                edge_dist_full = feb.edge_distance_for(missing)
                alphas = np.array([feb.continuous_alpha(run_len_full[m], edge_dist_full[m],
                                                        adaptive=True) for m in miss_idx])
                alphas = alphas * (1.0 - opts["periodic_alpha_strength"]
                                   * feb.periodic_agreement(values_for_feats)[miss_idx])
                pred = alphas * interp + (1 - alphas) * pred_neighbors

                # --- A4 : passe suivante avec les trous provisoirement
                # comblés (les copies périodiques ±48/±336 redeviennent
                # disponibles là où elles manquaient) -----------------------
                if p + 1 < n_passes:
                    values_for_feats = values_for_feats.copy()
                    values_for_feats[miss_idx] = pred

            filled = s.copy()
            filled.iloc[miss_idx] = pred
            preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model
