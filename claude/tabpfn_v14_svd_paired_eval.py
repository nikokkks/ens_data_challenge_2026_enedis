import os
import sys
import json
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import ShuffleSplit
from sklearn.utils.extmath import randomized_svd
import tabpfn_client

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from evaluate import load_data, fill_nan_with_interpolation
import feature_engineering_bench as feb
from tabpfn_v14_ablation import build_v14_features, DEFAULT_OPTS, MedianTabPFN, MAX_RETRIES

HERE = os.path.dirname(os.path.abspath(__file__))
V14_OPTS = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                extra_lags=(24,), neighbor_rolling=True)

token = os.environ.get("TABPFN_API_KEY")
if token:
    tabpfn_client.set_access_token(token)

t0 = time.time()
X_train, y_train, _ = load_data()
donor_cols = [c for c in X_train.columns if not c.startswith("holed_")]
holed_cols = list(y_train.columns)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

ss = ShuffleSplit(n_splits=5, test_size=0.10, random_state=42)
dev_idx, val_idx = next(iter(ss.split(holed_cols)))
dev_cols = [holed_cols[i] for i in dev_idx]
val_cols = [holed_cols[i] for i in val_idx]
print("val_cols[:5]:", val_cols[:5], "n_val:", len(val_cols))

donor_df = X_train[donor_cols]
dev_true_df = y_train[dev_cols]
val_holed_df = X_train[val_cols]

t0 = time.time()
km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=15, random_state=42)
dev_filled = dev_true_df.ffill().bfill()
Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
pool_labels = pd.concat([donor_labels, dev_labels])
pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
tfeat = feb.time_features_full_seasonality(val_holed_df.index, ("hour", "dow", "dom", "year"))
print(f"clustering in {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
pool = pd.concat([donor_df, dev_true_df], axis=1)
pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
filled = pool.ffill().bfill().to_numpy(dtype=np.float64)
c_ = filled - filled.mean(axis=0, keepdims=True)
U, s, Vt = randomized_svd(c_, n_components=20, random_state=42)
svd_feat = U * s
print(f"pool+svd in {time.time()-t0:.1f}s", flush=True)


def alpha_blend(s_series, missing, miss_idx, interp_full, pred_neighbors, opts):
    runs = feb.run_lengths_for(missing)
    edges = feb.edge_distance_for(missing)
    alpha = np.array([feb.continuous_alpha(runs[i], edges[i], True) for i in miss_idx])
    p_agree = feb.periodic_agreement(s_series.to_numpy())[miss_idx]
    alpha *= 1.0 - opts["periodic_alpha_strength"] * p_agree
    vol = feb.curve_volatility(s_series.to_numpy())
    alpha *= feb.volatility_alpha_mult(vol, vol_weight=opts["vol_weight"], cap=opts["vol_cap"])
    alpha = np.clip(alpha, 0.0, 0.95)
    return alpha * interp_full.to_numpy()[miss_idx] + (1 - alpha) * pred_neighbors


def eval_column(col, use_svd):
    opts = {**DEFAULT_OPTS, **V14_OPTS}
    s_series = val_holed_df[col]
    built = build_v14_features(col, s_series, pool_vals, km, cluster_col_idx, tfeat, val_holed_df.index, opts)
    if built is None:
        return col, use_svd, None, None
    obs_idx, miss_idx, target_obs, interp_full, knn_pred, scaled, extra, missing, lags = built
    feats = np.hstack([extra, svd_feat]) if use_svd else extra
    pred_neighbors = None
    error = None
    if len(obs_idx) >= 15:
        for attempt in range(MAX_RETRIES):
            try:
                pred_neighbors = feb.local_regression_pred(
                    lambda: MedianTabPFN(), scaled, obs_idx, miss_idx, target_obs, feats)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(2 * (attempt + 1))
    if pred_neighbors is None:
        pred_neighbors = knn_pred
    pred = alpha_blend(s_series, missing, miss_idx, interp_full, pred_neighbors, opts)
    y_true = y_train[col].to_numpy()[miss_idx]
    valid = ~np.isnan(y_true)
    abs_err = np.abs(pred[valid] - y_true[valid])
    return col, use_svd, abs_err, error


results = {}
t0 = time.time()
with ThreadPoolExecutor(max_workers=20) as ex:
    futs = []
    for c in val_cols:
        futs.append(ex.submit(eval_column, c, False))
        futs.append(ex.submit(eval_column, c, True))
    for f in as_completed(futs):
        col, use_svd, abs_err, error = f.result()
        results.setdefault(col, {})[use_svd] = abs_err
        if error:
            print(f"error {col} svd={use_svd}: {error}", flush=True)
print(f"tabpfn paired eval done in {time.time()-t0:.1f}s", flush=True)

baseline_errs, svd_errs, deltas = [], [], []
wins_svd, wins_base, ties = 0, 0, 0
for c in val_cols:
    r = results.get(c, {})
    if r.get(False) is None or r.get(True) is None:
        continue
    mb, ms = r[False].mean(), r[True].mean()
    baseline_errs.append(r[False])
    svd_errs.append(r[True])
    deltas.append(ms - mb)
    if ms < mb - 1e-9:
        wins_svd += 1
    elif mb < ms - 1e-9:
        wins_base += 1
    else:
        ties += 1

mae_base = float(np.concatenate(baseline_errs).mean())
mae_svd = float(np.concatenate(svd_errs).mean())
deltas = np.array(deltas)
print("MAE baseline (sans SVD):", mae_base)
print("MAE avec SVD:", mae_svd)
print("gain relatif:", 100 * (1 - mae_svd / mae_base), "%")
print("colonnes gagnées par SVD:", wins_svd, "/ perdues:", wins_base, "/ egalite:", ties, "/ total:", len(deltas))
print("delta moyen par colonne (svd-base):", deltas.mean(), "std:", deltas.std())

out = {
    "val_cols": val_cols,
    "mae_baseline": mae_base,
    "mae_with_svd": mae_svd,
    "gain_pct": 100 * (1 - mae_svd / mae_base),
    "wins_svd": wins_svd, "wins_baseline": wins_base, "ties": ties,
    "delta_mean": float(deltas.mean()), "delta_std": float(deltas.std()),
}
with open(os.path.join(HERE, "tabpfn_v14_svd_paired_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print("written", flush=True)
