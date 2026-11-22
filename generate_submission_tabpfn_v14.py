"""Génère la soumission avec le meilleur modèle validé : TabPFN V14
(63.24 +/- 30.46 W sur 10x10, -4.53% vs V12, 9 repeats/10 gagnés,
cf. benchmark.md §13). Réutilise le pipeline exact validé dans
tabpfn_v14_ablation.py (complete_column_v14)."""
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tabpfn_client

from evaluate import load_data
import feature_engineering_bench as feb
from tabpfn_v14_ablation import predict_column_v14

warnings.filterwarnings("ignore")
OUT_PATH = "submission_tabpfn_v14.csv"
CHECKPOINT_PATH = "submission_tabpfn_v14_partial.csv"
N_CLUSTERS, RANDOM_STATE = 15, 42
BATCH_SIZE = 20

V14_OPTS = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                extra_lags=(24,), neighbor_rolling=True)


if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant dans l'environnement")
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
        batch = remaining[start:start + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = [executor.submit(predict_column_v14, c, X_test[c], pool_vals, km,
                                        cluster_idx, tfeat, X_test.index, V14_OPTS)
                       for c in batch]
            for future in as_completed(futures):
                col, filled, error = future.result()
                done[col] = filled
                if error:
                    fallbacks += 1
                    print(f"Repli KNN pour {col}: {error}", flush=True)
        done.to_csv(CHECKPOINT_PATH)
        completed = start + len(batch)
        elapsed = time.time() - started
        eta = elapsed / completed * (len(remaining) - completed) / 60
        rate = completed / elapsed if elapsed else 0.0
        print(f"[{len(done.columns)}/1000] lot={len(batch)} parallèle={BATCH_SIZE} "
              f"temps={elapsed/60:.1f} min vitesse={rate*60:.1f} col/min "
              f"ETA={eta:.1f} min replis={fallbacks}", flush=True)

    final = done[holes].clip(lower=0)
    final.index.name = "Horodate"
    sample = pd.read_csv("data/new_output_sample.csv", index_col=0)
    assert final.shape == sample.shape == (1057, 1000)
    assert list(final.columns) == list(sample.columns)
    assert final.isna().sum().sum() == 0
    assert np.isfinite(final.to_numpy()).all()
    final.to_csv(OUT_PATH)
    print(f"Soumission écrite : {OUT_PATH}; replis KNN={fallbacks}", flush=True)
