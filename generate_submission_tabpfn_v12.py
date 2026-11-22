"""Génère la soumission avec le meilleur modèle validé : TabPFN V12 (66.24 W)."""
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tabpfn_client
from tabpfn_client import TabPFNRegressor

from evaluate import load_data, fill_nan_with_interpolation
import feature_engineering_bench as feb

warnings.filterwarnings("ignore")
OUT_PATH = "submission_tabpfn_v12.csv"
CHECKPOINT_PATH = "submission_tabpfn_v12_partial.csv"
K, N_CLUSTERS, RANDOM_STATE = 40, 15, 42
BATCH_SIZE, MAX_RETRIES = 20, 3


class MedianTabPFN:
    def __init__(self):
        self.reg = TabPFNRegressor(random_state=RANDOM_STATE)
    def fit(self, X, y):
        self.reg.fit(X, y)
        return self
    def predict(self, X):
        return self.reg.predict(X, output_type="median")


def complete_column(col, s, pool_vals, km, cluster_col_idx, tfeat, index):
    observed = s.notna().to_numpy(); missing = ~observed
    if not missing.any(): return col, s, None
    obs_idx, miss_idx = np.flatnonzero(observed), np.flatnonzero(missing)
    target_obs = s.to_numpy()[obs_idx]
    interp_full = fill_nan_with_interpolation(s)
    cluster = feb.cluster_of(interp_full, km)
    same_idx = cluster_col_idx.get(cluster)
    sub_vals = (pool_vals[:, same_idx]
                if same_idx is not None and len(same_idx) >= max(2*K, 30)
                else pool_vals)
    knn_pred, _, scaled = feb.weighted_knn_pred(
        sub_vals, obs_idx, miss_idx, target_obs, K, robust=True,
        adaptive_k=True, use_euclid=True)

    periodic = feb.periodic_copy_features(
        s.to_numpy(), include_summary=True, normalize=True,
        include_availability=True)
    seasonal = feb.same_curve_seasonality_feature(s.to_numpy(), index)
    extra = np.hstack([tfeat, seasonal, periodic])

    error = None
    pred_neighbors = None
    fallback_used = False
    if len(obs_idx) >= 15:
        for attempt in range(MAX_RETRIES):
            try:
                pred_neighbors = feb.local_regression_pred(
                    MedianTabPFN, scaled, obs_idx, miss_idx, target_obs, extra)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(2 * (attempt + 1))
    if pred_neighbors is None:
        # Repli déterministe : KNN, jamais une colonne/une soumission invalide.
        pred_neighbors = knn_pred
        fallback_used = True

    runs = feb.run_lengths_for(missing)
    edges = feb.edge_distance_for(missing)
    alpha = np.array([feb.continuous_alpha(runs[i], edges[i], True)
                      for i in miss_idx])
    alpha *= 1.0 - 0.5 * feb.periodic_agreement(s.to_numpy())[miss_idx]
    pred = alpha * interp_full.to_numpy()[miss_idx] + (1-alpha) * pred_neighbors
    filled = s.copy(); filled.iloc[miss_idx] = pred
    return col, filled, error if fallback_used else None


if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        from tabpfn_bench_gemini import API_KEY as token
    tabpfn_client.set_access_token(token)
    print("Chargement des données...", flush=True)
    X_train, y_train, X_test = load_data()
    test_donors = [c for c in X_test if not str(c).startswith("holed_")]
    train_donors = [c for c in X_train if not str(c).startswith("holed_")]
    holes = [c for c in X_test if str(c).startswith("holed_")]
    pool = pd.concat([X_test[test_donors], X_train[train_donors], y_train], axis=1)
    print(f"Pool maximal : {pool.shape[1]} donneurs", flush=True)

    if os.path.exists(CHECKPOINT_PATH):
        done = pd.read_csv(CHECKPOINT_PATH, index_col=0)
        done.index = X_test.index
    else:
        done = pd.DataFrame(index=X_test.index)
    remaining = [c for c in holes if c not in done]
    print(f"Reprise : {len(done.columns)} faites, {len(remaining)} restantes", flush=True)

    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    km, labels = feb.fit_shape_clusters(pool, N_CLUSTERS, RANDOM_STATE)
    positions = pd.Series(np.arange(len(labels)), index=labels.index)
    cluster_idx = {cl: positions[idx].to_numpy()
                   for cl, idx in labels.groupby(labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(
        X_test.index, groups=("hour", "dow", "dom", "year"))

    fallbacks = 0; started = time.time()
    for start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[start:start+BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = [executor.submit(complete_column, c, X_test[c], pool_vals,
                        km, cluster_idx, tfeat, X_test.index) for c in batch]
            for future in as_completed(futures):
                col, filled, error = future.result()
                done[col] = filled
                if error:
                    fallbacks += 1
                    print(f"Repli KNN pour {col}: {error}", flush=True)
        done.to_csv(CHECKPOINT_PATH)
        completed = start + len(batch)
        elapsed = time.time() - started
        eta = elapsed / completed * (len(remaining)-completed) / 60
        rate = completed / elapsed if elapsed else 0.0
        print(f"[{len(done.columns)}/1000] lot={len(batch)} parallèle={BATCH_SIZE} "
              f"temps={elapsed/60:.1f} min vitesse={rate*60:.1f} col/min "
              f"ETA={eta:.1f} min replis={fallbacks}",
              flush=True)

    final = done[holes].clip(lower=0)
    final.index.name = "Horodate"
    sample = pd.read_csv("data/new_output_sample.csv", index_col=0)
    assert final.shape == sample.shape == (1057, 1000)
    assert list(final.columns) == list(sample.columns)
    assert final.isna().sum().sum() == 0
    assert np.isfinite(final.to_numpy()).all()
    final.to_csv(OUT_PATH)
    print(f"Soumission écrite : {OUT_PATH}; replis KNN={fallbacks}", flush=True)
