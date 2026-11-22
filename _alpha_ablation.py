import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from evaluate import load_data, official_mae, fill_nan_with_interpolation
import feature_engineering_bench as feb
from catboost_gpu_finetune import make_catboost_factory, BEST_FEATURES

X, y, _ = load_data()
donors = [c for c in X if not str(c).startswith("holed_")]
holed_cols = list(y.columns)
seed = 42
dev_cols, val_cols_full = train_test_split(holed_cols, test_size=.1, random_state=seed)
rng = np.random.RandomState(seed)
val_cols = list(rng.choice(val_cols_full, size=20, replace=False))

donor_df, dev_true_df, val_holed_df = X[donors], y[dev_cols], X[val_cols]
pool = pd.concat([donor_df, dev_true_df], axis=1)
pool_vals = pool.ffill().bfill().values.astype(np.float64)
km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=15, random_state=seed)
dev_filled = dev_true_df.ffill().bfill()
Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
pool_labels = pd.concat([donor_labels, dev_labels])
pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
tfeat = feb.time_features_full_seasonality(val_holed_df.index, ("hour", "dow", "dom", "year"))

factory = make_catboost_factory(depth=4, learning_rate=.1, iterations=200,
    l2_leaf_reg=1.0, bagging_temperature=1.0, use_gpu=True)

modes = {"alpha_0_pure_model": 0.0, "alpha_1_pure_interp": 1.0, "alpha_blend": None}
results = {m: {} for m in modes}

for col in val_holed_df.columns:
    s = val_holed_df[col]
    observed = s.notna().values; missing = ~observed
    if not missing.any(): continue
    obs_idx, miss_idx = np.where(observed)[0], np.where(missing)[0]
    target_obs = s.values[obs_idx]
    interp_full = fill_nan_with_interpolation(s)
    cl = feb.cluster_of(interp_full, km)
    same_idx = cluster_col_idx.get(cl)
    sub_vals = pool_vals[:, same_idx] if same_idx is not None and len(same_idx) >= max(2*500, 30) else pool_vals
    knn_pred, top_idx, scaled = feb.weighted_knn_pred(sub_vals, obs_idx, miss_idx, target_obs, 500,
        robust=True, adaptive_k=True, use_euclid=True)
    extra = tfeat
    extra = np.hstack([extra, feb.same_curve_seasonality_feature(s.values, val_holed_df.index)])
    periodic = feb.periodic_copy_features(s.values, include_summary=True, normalize=True, include_availability=True)
    extra = np.hstack([extra, periodic])
    if len(obs_idx) >= 15:
        pred_neighbors = feb.local_regression_pred(factory, scaled, obs_idx, miss_idx, target_obs, extra)
    else:
        pred_neighbors = knn_pred
    interp = interp_full.values[miss_idx]
    run_len_full = feb.run_lengths_for(missing)
    edge_dist_full = feb.edge_distance_for(missing)
    alphas_blend = np.array([feb.continuous_alpha(run_len_full[m], edge_dist_full[m], adaptive=True) for m in miss_idx])
    alphas_blend = alphas_blend * (1.0 - 0.5*feb.periodic_agreement(s.values)[miss_idx])
    y_true = y[col].values[miss_idx]
    for mname, aval in modes.items():
        a = alphas_blend if aval is None else np.full(len(miss_idx), aval)
        pred = a * interp + (1 - a) * pred_neighbors
        mask_valid = ~np.isnan(y_true)
        results[mname][col] = np.abs(pred[mask_valid] - y_true[mask_valid])

print(f"n_val_cols={len(val_cols)}")
for mname, per_col in results.items():
    all_err = np.concatenate(list(per_col.values()))
    print(f"{mname:24s} MAE={all_err.mean():8.2f} W  (n_pts={len(all_err)})")
