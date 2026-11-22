import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import json
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.utils.extmath import randomized_svd
from catboost import CatBoostRegressor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from evaluate import repeated_holdout_eval
import feature_engineering_bench as feb
from tabpfn_v14_ablation import build_v14_features, DEFAULT_OPTS

DATA_DIR = os.path.join(REPO_ROOT, "data")
HERE = os.path.dirname(os.path.abspath(__file__))

t0 = time.time()
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_78VdSWL.csv"), index_col=0)
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_u0UkKEh.csv"), index_col=0)
donor_cols = [c for c in X_train.columns if not c.startswith("holed_")]
print(f"loaded in {time.time()-t0:.1f}s, donors={len(donor_cols)}", flush=True)

V14_OPTS = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                extra_lags=(24,), neighbor_rolling=True)


def compute_svd_factors(pool_df, n_components=20):
    filled = pool_df.ffill().bfill().to_numpy(dtype=np.float64)
    c = filled - filled.mean(axis=0, keepdims=True)
    U, s, Vt = randomized_svd(c, n_components=n_components, random_state=42)
    return U * s


def make_catboost_svd_model(n_svd=20, n_clusters=15, opts=None, max_workers=4):
    opts = {**DEFAULT_OPTS, **(opts or {})}

    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
        svd_feat = compute_svd_factors(pool, n_components=n_svd)

        km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=n_clusters, random_state=42)
        if dev_true_df.shape[1]:
            dev_filled = dev_true_df.ffill().bfill()
            Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
            dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
        else:
            dev_labels = pd.Series([], dtype=int)
        pool_labels = pd.concat([donor_labels, dev_labels])
        pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
        cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
        tfeat = feb.time_features_full_seasonality(val_holed_df.index, ("hour", "dow", "dom", "year"))

        def complete(c_):
            s = val_holed_df[c_]
            built = build_v14_features(c_, s, pool_vals, km, cluster_col_idx, tfeat, val_holed_df.index, opts)
            if built is None:
                return c_, s.copy()
            obs_idx, miss_idx, target_obs, interp_full, knn_pred, scaled, extra, missing, lags = built
            extra2 = np.hstack([extra, svd_feat])
            if len(obs_idx) >= 15 and not np.all(target_obs == target_obs[0]):
                reg = CatBoostRegressor(iterations=150, depth=5, loss_function="MAE",
                                         verbose=False, random_state=42, thread_count=1,
                                         allow_writing_files=False)
                X_obs = np.hstack([scaled[obs_idx, :], extra2[obs_idx, :]])
                X_miss = np.hstack([scaled[miss_idx, :], extra2[miss_idx, :]])
                reg.fit(X_obs, target_obs)
                pred_neighbors = reg.predict(X_miss)
            elif np.all(target_obs == target_obs[0]):
                pred_neighbors = np.full(len(miss_idx), target_obs[0])
            else:
                pred_neighbors = knn_pred
            runs = feb.run_lengths_for(missing)
            edges = feb.edge_distance_for(missing)
            alpha = np.array([feb.continuous_alpha(runs[i], edges[i], True) for i in miss_idx])
            p_agree = feb.periodic_agreement(s.to_numpy())[miss_idx]
            alpha *= 1.0 - opts["periodic_alpha_strength"] * p_agree
            vol = feb.curve_volatility(s.to_numpy())
            alpha *= feb.volatility_alpha_mult(vol, vol_weight=opts["vol_weight"], cap=opts["vol_cap"])
            alpha = np.clip(alpha, 0.0, 0.95)
            pred = alpha * interp_full.to_numpy()[miss_idx] + (1 - alpha) * pred_neighbors
            filled = s.copy()
            filled.iloc[miss_idx] = pred
            return c_, filled

        preds = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(complete, c_) for c_ in val_holed_df.columns]
            for f in as_completed(futs):
                c_, filled = f.result()
                preds[c_] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)[val_holed_df.columns]

    return model


if __name__ == "__main__":
    t0 = time.time()
    model = make_catboost_svd_model(n_svd=20, opts=V14_OPTS)
    scores = repeated_holdout_eval(X_train, y_train, donor_cols, model, n_repeats=5, test_size=0.10, random_state=42)
    print(f"catboost+v14features+svd 5x100 done in {time.time()-t0:.1f}s", flush=True)
    print("MAE mean:", float(scores.mean()), "std:", float(scores.std()), "per_repeat:", scores.tolist())

    result = {
        "catboost_v14features_svd_5x100": {
            "mean": float(scores.mean()), "std": float(scores.std()), "per_repeat": scores.tolist()
        },
        "reference_svd_matrix_completion_5x100": {"mean": 104.90, "std": 11.58},
        "reference_lightgbm_5x100_k100": {"mean": 74.50, "std": 7.11},
        "reference_tabpfn_v14_10x10": {"mean": 63.24, "std": 30.46, "note": "protocole different, non comparable directement"},
    }
    out_path = os.path.join(HERE, "catboost_v14features_svd_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("written:", out_path, flush=True)
