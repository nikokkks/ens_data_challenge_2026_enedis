"""Meilleur modèle local (hors TabPFN) + 53 features supplémentaires
(`mega_features.py`) construites sur le pool de 20 000+ courbes donneuses et
sur la courbe cible. Base : LightGBM + imputation itérative 2 passes (A4,
68.89 W, cf. benchmark.md § 7bis) -- structure copiée de
`research_ideas.make_research_model(n_passes=2)`, avec le bloc de features
mega injecté à chaque passe."""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

from evaluate import fill_nan_with_interpolation
import feature_engineering_bench as feb
import mega_features as mf

BEST_FEATURES = dict(
    use_seasonality_feat=True, use_full_calendar_feat=True,
    calendar_groups=("hour", "dow", "dom", "year"),
    use_periodic_copy_feat=True, use_periodic_consensus_feat=True,
    normalize_periodic_feat=True, include_periodic_availability=True,
    periodic_alpha_strength=.5,
)


def make_mega_model(k=40, n_clusters=15, make_regressor=None,
                     random_state=feb.RANDOM_STATE, n_passes=2, use_mega=True):
    """add_mega=False reproduit EXACTEMENT A4_iterative_2pass (contrôle) ;
    add_mega=True ajoute les 53 features de mega_features.py à chaque passe."""
    if make_regressor is None:
        make_regressor = lambda: Ridge(alpha=1.0)

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

        tfeat = feb.time_features_full_seasonality(val_holed_df.index, BEST_FEATURES["calendar_groups"])

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
            sub_vals = pool_vals[:, same_idx] if (same_idx is not None
                        and len(same_idx) >= max(2 * k, 30)) else pool_vals

            knn_pred, top_idx, scaled = feb.weighted_knn_pred(
                sub_vals, obs_idx, miss_idx, target_obs, k, robust=True,
                adaptive_k=True, use_euclid=True)
            corr = None  # récupéré via top_idx/adaptive_top_k_corr en repassant par la même sélection
            # weighted_knn_pred ne renvoie pas corr directement -> recalcule via top_idx
            t_c = target_obs - target_obs.mean()
            d_c = sub_vals[obs_idx][:, top_idx] - sub_vals[obs_idx][:, top_idx].mean(axis=0, keepdims=True)
            num = d_c.T @ t_c
            denom = np.sqrt((d_c ** 2).sum(axis=0) * (t_c ** 2).sum()) + 1e-9
            corr = num / denom

            values_for_feats = s.values
            pred = None
            for p in range(max(1, n_passes)):
                periodic_raw = feb.periodic_copy_features(values_for_feats)
                periodic_median_full = np.nanmedian(periodic_raw, axis=1)
                periodic = feb.periodic_copy_features(
                    values_for_feats, include_summary=True, normalize=True,
                    include_availability=True)
                extra = np.hstack([
                    tfeat,
                    feb.same_curve_seasonality_feature(values_for_feats, val_holed_df.index),
                    periodic,
                ])
                if use_mega:
                    mega = mf.build_mega_extra(
                        s, val_holed_df.index, obs_idx, miss_idx, target_obs, scaled, corr,
                        sub_vals, missing, values_for_feats, periodic_median_full)
                    extra = np.hstack([extra, mega])

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
                alphas = alphas * (1.0 - BEST_FEATURES["periodic_alpha_strength"]
                                   * feb.periodic_agreement(values_for_feats)[miss_idx])
                pred = alphas * interp + (1 - alphas) * pred_neighbors

                if p + 1 < n_passes:
                    values_for_feats = values_for_feats.copy()
                    values_for_feats[miss_idx] = pred

            filled = s.copy()
            filled.iloc[miss_idx] = pred
            preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model
