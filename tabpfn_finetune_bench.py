"""Reprise du fine-tuning (cf. catboost_gpu_finetune.py) avec le VRAI meilleur
modèle, TabPFN, maintenant que l'accès réseau/API est disponible.

Teste, avec le pipeline de features V12 (BEST_FEATURES), le même axe que la
version CatBoost : k=40 (référence V12) vs k élargi (100, puis "toutes les
features pertinentes" via un cap large piloté par le seuil de corrélation).

Parallélisme par ThreadPoolExecutor sur les colonnes d'un même split (le
goulot est le réseau, pas le CPU) -- même pattern que
generate_submission_tabpfn_v12.py.
"""
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tabpfn_client
from tabpfn_client import TabPFNRegressor
from sklearn.model_selection import train_test_split

from evaluate import load_data, official_mae, fill_nan_with_interpolation
import feature_engineering_bench as feb

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_CLUSTERS = 15
RESULTS_PATH = "tabpfn_finetune_results.json"
MAX_WORKERS = 20
MAX_RETRIES = 3


class MedianTabPFN:
    def __init__(self):
        self.reg = TabPFNRegressor(random_state=RANDOM_STATE)

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X, output_type="median")


def complete_column(col, s, y_true_col, pool_vals, km, cluster_col_idx, tfeat, index, k):
    observed = s.notna().to_numpy()
    missing = ~observed
    if not missing.any():
        return col, None, None
    obs_idx, miss_idx = np.flatnonzero(observed), np.flatnonzero(missing)
    target_obs = s.to_numpy()[obs_idx]
    interp_full = fill_nan_with_interpolation(s)
    cluster = feb.cluster_of(interp_full, km)
    same_idx = cluster_col_idx.get(cluster)
    sub_vals = (pool_vals[:, same_idx]
                if same_idx is not None and len(same_idx) >= max(2 * k, 30)
                else pool_vals)
    knn_pred, _, scaled = feb.weighted_knn_pred(
        sub_vals, obs_idx, miss_idx, target_obs, k, robust=True,
        adaptive_k=True, use_euclid=True)

    periodic = feb.periodic_copy_features(
        s.to_numpy(), include_summary=True, normalize=True,
        include_availability=True)
    seasonal = feb.same_curve_seasonality_feature(s.to_numpy(), index)
    extra = np.hstack([tfeat, seasonal, periodic])

    error = None
    pred_neighbors = None
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
        pred_neighbors = knn_pred

    runs = feb.run_lengths_for(missing)
    edges = feb.edge_distance_for(missing)
    alpha = np.array([feb.continuous_alpha(runs[i], edges[i], True) for i in miss_idx])
    alpha *= 1.0 - 0.5 * feb.periodic_agreement(s.to_numpy())[miss_idx]
    pred = alpha * interp_full.to_numpy()[miss_idx] + (1 - alpha) * pred_neighbors

    y_true = y_true_col.to_numpy()[miss_idx]
    valid = ~np.isnan(y_true)
    abs_err = np.abs(pred[valid] - y_true[valid])
    return col, abs_err, error


def eval_split(X_train, y_train, donor_cols, dev_cols, val_cols, k, n_clusters=N_CLUSTERS):
    donor_df = X_train[donor_cols]
    dev_true_df = y_train[dev_cols]
    val_holed_df = X_train[val_cols]

    pool = pd.concat([donor_df, dev_true_df], axis=1)
    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=n_clusters, random_state=RANDOM_STATE)
    dev_filled = dev_true_df.ffill().bfill()
    Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
    pool_labels = pd.concat([donor_labels, dev_labels])
    pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
    cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(val_holed_df.index, ("hour", "dow", "dom", "year"))

    all_errs = []
    fallbacks = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(complete_column, c, val_holed_df[c], y_train[c],
                                    pool_vals, km, cluster_col_idx, tfeat,
                                    val_holed_df.index, k) for c in val_cols]
        for fut in as_completed(futures):
            col, abs_err, error = fut.result()
            if abs_err is not None:
                all_errs.append(abs_err)
            if error:
                fallbacks += 1
    return float(np.concatenate(all_errs).mean()), fallbacks


def run_config(name, k, X, y, donor_cols, n_repeats, n_val_sub):
    holed_cols = list(y.columns)
    scores = []
    for i in range(n_repeats):
        seed_i = RANDOM_STATE + i
        dev_cols, val_cols_full = train_test_split(holed_cols, test_size=.1, random_state=seed_i)
        rng = np.random.RandomState(seed_i)
        val_cols = list(rng.choice(val_cols_full, size=min(n_val_sub, len(val_cols_full)), replace=False))
        t0 = time.time()
        mae, fallbacks = eval_split(X, y, donor_cols, dev_cols, val_cols, k)
        dt = time.time() - t0
        scores.append(mae)
        print(f"  [{name}] repeat {i+1}/{n_repeats}: MAE={mae:8.2f} W "
              f"(n={len(val_cols)}, {dt:.1f}s, fallbacks={fallbacks})", flush=True)
    scores = np.array(scores)
    print(f"  -> [{name}] Mean MAE over {n_repeats} repeats: {scores.mean():.2f} +/- {scores.std():.2f} W\n", flush=True)
    return scores


CONFIGS = {"k40_v12_ref": 40, "k100": 100, "k300_allfeat": 300}


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "screen"
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant dans l'environnement")
    tabpfn_client.set_access_token(token)

    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]

    try:
        results = json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        results = {}

    if stage == "screen":
        n_repeats, n_val_sub = 3, 10
    elif stage == "confirm":
        n_repeats, n_val_sub = 10, 10
    else:
        raise SystemExit(f"stage inconnu: {stage}")

    names = sys.argv[2:] if len(sys.argv) > 2 else list(CONFIGS)
    for name in names:
        k = CONFIGS[name]
        print(f"=== {stage.upper()} {name} (k={k}) ===", flush=True)
        scores = run_config(name, k, X, y, donors, n_repeats, n_val_sub)
        results[f"{stage}_{name}"] = {"scores": scores.tolist(), "mean": float(scores.mean()),
                                       "std": float(scores.std()), "k": k}
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    print("Terminé. Résultats dans", RESULTS_PATH)
