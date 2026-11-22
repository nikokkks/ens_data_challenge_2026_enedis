"""
Script jetable : ablation des 3 pistes candidates pour V8 sur UN SEUL split
(repeat 1, random_state=42, 10 colonnes val), pour décider lesquelles garder
avant de lancer les runs complets (n=10) sur v8_ridge_bench.py / tabpfn_bench.py.
Résultats et interprétation dans benchmark.md § V8.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from evaluate import load_data, official_mae, fill_nan_with_interpolation
import feature_engineering_bench as feb

X_train, y_train, X_test = load_data()
donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]
holed_cols = list(y_train.columns)


def make_flex_model(use_adaptive_k=False, use_edge_alpha=False, use_lag_feats=False,
                     k=20, n_clusters=25):
    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        pool_vals = pool.ffill().bfill().values.astype(np.float64)
        km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=n_clusters)
        if dev_true_df.shape[1] > 0:
            dev_filled = dev_true_df.ffill().bfill()
            Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
            dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
            pool_labels = pd.concat([donor_labels, dev_labels])
        else:
            pool_labels = donor_labels
        pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
        cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
        tfeat = feb.time_features(val_holed_df.index)

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

            knn_pred, top_idx, scaled = feb.weighted_knn_pred(
                sub_vals, obs_idx, miss_idx, target_obs, k, robust=True, adaptive_k=use_adaptive_k
            )

            if use_lag_feats:
                lag_feats = feb.nearest_known_features(s.values)
                extra = np.hstack([tfeat, lag_feats])
            else:
                extra = tfeat

            if len(obs_idx) >= 15:
                pred_neighbors = feb.local_regression_pred(lambda: Ridge(alpha=1.0), scaled, obs_idx, miss_idx, target_obs, extra)
            else:
                pred_neighbors = knn_pred

            interp = interp_full.values[miss_idx]
            if use_edge_alpha:
                run_len_full = feb.run_lengths_for(missing)
                edge_dist_full = feb.edge_distance_for(missing)
                alphas = np.array([feb.continuous_alpha(run_len_full[m], edge_dist_full[m], adaptive=True) for m in miss_idx])
            else:
                run_len_full = feb.run_lengths_for(missing)
                alphas = np.array([feb.gap_alpha(run_len_full[m], adaptive=True) for m in miss_idx])
            pred = alphas * interp + (1 - alphas) * pred_neighbors

            filled = s.copy()
            filled.iloc[miss_idx] = pred
            preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model


if __name__ == "__main__":
    seed_i = 42
    dev_cols, val_cols_full = train_test_split(holed_cols, test_size=0.10, random_state=seed_i)
    sub_rng = np.random.RandomState(seed_i)
    val_cols = list(sub_rng.choice(val_cols_full, size=10, replace=False))
    mask = X_train[val_cols].isna()

    configs = {
        "V5 (baseline ref)": dict(use_adaptive_k=False, use_edge_alpha=False, use_lag_feats=False),
        "+adaptive_k only": dict(use_adaptive_k=True, use_edge_alpha=False, use_lag_feats=False),
        "+edge_alpha only": dict(use_adaptive_k=False, use_edge_alpha=True, use_lag_feats=False),
        "+lag_feats only": dict(use_adaptive_k=False, use_edge_alpha=False, use_lag_feats=True),
        "adaptive_k + edge_alpha (retenu)": dict(use_adaptive_k=True, use_edge_alpha=True, use_lag_feats=False),
        "all 3": dict(use_adaptive_k=True, use_edge_alpha=True, use_lag_feats=True),
    }

    for name, cfg in configs.items():
        model = make_flex_model(**cfg)
        y_pred = model(X_train[donor_cols], X_train[dev_cols], y_train[dev_cols], X_train[val_cols])
        mae = official_mae(y_train[val_cols], y_pred, mask)
        print(f"{name:34s} MAE = {mae:8.2f} W")
