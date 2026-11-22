"""Soumission V13 : V12 + pente/courbure du consensus des 40 voisins."""
import os, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import tabpfn_client
from tabpfn_client import TabPFNRegressor

from evaluate import load_data, fill_nan_with_interpolation
import feature_engineering_bench as feb
from tabpfn_v13_features import donor_consensus_features

warnings.filterwarnings("ignore")
OUT_PATH = "submission_tabpfn_v13.csv"
CHECKPOINT_PATH = "submission_tabpfn_v13_partial.csv"
K, N_CLUSTERS, RANDOM_STATE = 40, 15, 42
BATCH_SIZE, MAX_RETRIES = 20, 3


class MedianTabPFN:
    def __init__(self): self.reg = TabPFNRegressor(random_state=RANDOM_STATE)
    def fit(self, X, y): self.reg.fit(X, y); return self
    def predict(self, X): return self.reg.predict(X, output_type="median")


def complete_column(col, s, pool_vals, km, cluster_col_idx, tfeat, index):
    observed = s.notna().to_numpy(); missing = ~observed
    if not missing.any(): return col, s, None
    obs_idx = np.flatnonzero(observed); miss_idx = np.flatnonzero(missing)
    target_obs = s.to_numpy()[obs_idx]
    interp = fill_nan_with_interpolation(s)
    cluster = feb.cluster_of(interp, km); same = cluster_col_idx.get(cluster)
    sub = pool_vals[:, same] if same is not None and len(same) >= max(2*K, 30) else pool_vals
    knn, _, scaled = feb.weighted_knn_pred(sub, obs_idx, miss_idx, target_obs, K,
        robust=True, adaptive_k=True, use_euclid=True)
    _, corr = feb.adaptive_top_k_corr_euclid(target_obs, sub[obs_idx], k_max=K,
                                              corr_threshold=.15)
    base = np.hstack([
        tfeat,
        feb.same_curve_seasonality_feature(s.to_numpy(), index),
        feb.periodic_copy_features(s.to_numpy(), include_summary=True,
            normalize=True, include_availability=True),
    ])
    slopes = donor_consensus_features(scaled, corr, ("neighbor_slopes",))
    extra = np.hstack([base, slopes])
    error = None; pred_model = None
    if len(obs_idx) >= 15:
        for attempt in range(MAX_RETRIES):
            try:
                pred_model = feb.local_regression_pred(
                    MedianTabPFN, scaled, obs_idx, miss_idx, target_obs, extra)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"; time.sleep(2*(attempt+1))
    if pred_model is None: pred_model = knn
    runs = feb.run_lengths_for(missing); edges = feb.edge_distance_for(missing)
    alpha = np.array([feb.continuous_alpha(runs[i], edges[i], True) for i in miss_idx])
    alpha *= 1.0 - .5 * feb.periodic_agreement(s.to_numpy())[miss_idx]
    pred = np.maximum(alpha*interp.to_numpy()[miss_idx] + (1-alpha)*pred_model, 0)
    filled = s.copy(); filled.iloc[miss_idx] = pred
    return col, filled, error if pred_model is knn else None


if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        from tabpfn_bench_gemini import API_KEY as token
    tabpfn_client.set_access_token(token)
    X_train, y_train, X_test = load_data()
    test_donors = [c for c in X_test if not str(c).startswith("holed_")]
    train_donors = [c for c in X_train if not str(c).startswith("holed_")]
    holes = [c for c in X_test if str(c).startswith("holed_")]
    pool = pd.concat([X_test[test_donors], X_train[train_donors], y_train], axis=1)
    done = pd.read_csv(CHECKPOINT_PATH, index_col=0) if os.path.exists(CHECKPOINT_PATH) else pd.DataFrame(index=X_test.index)
    done.index = X_test.index
    remaining = [c for c in holes if c not in done]
    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    km, labels = feb.fit_shape_clusters(pool, N_CLUSTERS, RANDOM_STATE)
    positions = pd.Series(np.arange(len(labels)), index=labels.index)
    cluster_idx = {cl: positions[idx].to_numpy() for cl, idx in labels.groupby(labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(X_test.index, groups=("hour","dow","dom","year"))
    started = time.time(); fallbacks = 0
    for start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[start:start+BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = [executor.submit(complete_column, c, X_test[c], pool_vals,
                km, cluster_idx, tfeat, X_test.index) for c in batch]
            for future in as_completed(futures):
                col, filled, error = future.result(); done[col] = filled
                fallbacks += int(error is not None)
        done.to_csv(CHECKPOINT_PATH)
        completed = start + len(batch); elapsed = time.time()-started
        eta = elapsed/completed*(len(remaining)-completed)/60
        print(f"[{len(done.columns)}/1000] {elapsed/60:.1f} min, ETA={eta:.1f} min, replis={fallbacks}", flush=True)
    final = done[holes].clip(lower=0); final.index.name = "Horodate"
    sample = pd.read_csv("data/new_output_sample.csv", index_col=0)
    assert final.shape == sample.shape == (1057,1000)
    assert list(final.columns) == list(sample.columns)
    assert final.notna().all().all() and np.isfinite(final.to_numpy()).all()
    final.to_csv(OUT_PATH)
    print(f"Soumission écrite : {OUT_PATH}")
