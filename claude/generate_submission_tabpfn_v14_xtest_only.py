"""Génère une soumission avec TabPFN V14 en n'utilisant QUE X_test comme
source de données -- ni X_train ni y_train ne sont lus. Réutilise le
pipeline exact validé (predict_column_v14 de tabpfn_v14_ablation.py, même
config que generate_submission_tabpfn_v14.py), avec une seule différence :
le pool de donneurs pour le KNN/clustering/features est restreint aux
colonnes non "holed_*" de X_test (au lieu de X_test donors + X_train donors
+ y_train dans le script officiel).

Fit / predict
-------------
- Fit : sur l'ensemble des données de X_test SANS aucun NaN (les colonnes
  donneuses non "holed_*", ~37 140 colonnes) + les points déjà connus (non-NaN)
  de chaque colonne "holed_*" elle-même (comme pour tout le pipeline V14 :
  chaque colonne cible est régressée sur ses propres points observés).
- Predict : sur les positions réellement manquantes (NaN) des 1000 colonnes
  "holed_*" de X_test -- ce sont les valeurs à soumettre au challenge.

Attention : ceci restreint le pool de voisins par rapport au pipeline validé
(generate_submission_tabpfn_v14.py, qui ajoute aussi les donneurs X_train et
y_train), donc la MAE réelle sur cette variante n'est probablement pas
identique à la MAE validée à 63.24 +/- 30.46 W (cf. claude/tabpfn_v14_xtest_eval.py
pour estimer la MAE de cette config exacte via masquage synthétique avant de
soumettre).

Checkpointé comme generate_submission_tabpfn_v14.py : écrit
claude/submission_tabpfn_v14_xtest_only_partial.csv après chaque lot de
colonnes et saute les colonnes déjà faites en cas de relance.

Usage :
    export TABPFN_API_KEY=...   # optionnel si un token TabPFN est déjà en
                                 # cache localement (~/.config/.tabpfn)
    .venv/bin/python claude/generate_submission_tabpfn_v14_xtest_only.py
"""
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tabpfn_client

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import feature_engineering_bench as feb  # noqa: E402
from tabpfn_v14_ablation import predict_column_v14  # noqa: E402

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(REPO_ROOT, "data", "X_test_XKVc4no.csv")
SAMPLE_PATH = os.path.join(REPO_ROOT, "data", "new_output_sample.csv")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "submission_tabpfn_v14_xtest_only.csv")
CHECKPOINT_PATH = os.path.join(HERE, "submission_tabpfn_v14_xtest_only_partial.csv")

N_CLUSTERS, RANDOM_STATE = 15, 42
BATCH_SIZE = 20

# Config V14 validée (cf. README.md / generate_submission_tabpfn_v14.py),
# inchangée : seul le pool de donneurs diffère dans ce script.
V14_OPTS = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                extra_lags=(24,), neighbor_rolling=True)


if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if token:
        tabpfn_client.set_access_token(token)
    else:
        print("TABPFN_API_KEY absent : on tente l'authentification TabPFN "
              "mise en cache localement (~/.config/.tabpfn).", flush=True)

    print("Chargement de X_test uniquement (ni X_train ni y_train)...", flush=True)
    X_test = pd.read_csv(DATA_PATH, index_col=0)
    donor_cols = [c for c in X_test.columns if not str(c).startswith("holed_")]
    holes = [c for c in X_test.columns if str(c).startswith("holed_")]
    pool = X_test[donor_cols]
    print(f"Pool de donneurs (X_test uniquement) : {pool.shape[1]} colonnes, "
          f"{len(holes)} colonnes holed_* à compléter", flush=True)

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

    fallbacks = 0
    started = time.time()
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
        eta = elapsed / completed * (len(remaining) - completed) / 60 if completed else 0.0
        rate = completed / elapsed * 60 if elapsed else 0.0
        print(f"[{len(done.columns)}/1000] lot={len(batch)} parallèle={BATCH_SIZE} "
              f"temps={elapsed/60:.1f} min vitesse={rate:.1f} col/min "
              f"ETA={eta:.1f} min replis={fallbacks}", flush=True)

    final = done[holes].clip(lower=0)
    final.index.name = "Horodate"
    sample = pd.read_csv(SAMPLE_PATH, index_col=0)
    assert final.shape == sample.shape == (1057, 1000)
    assert list(final.columns) == list(sample.columns)
    assert final.isna().sum().sum() == 0
    assert np.isfinite(final.to_numpy()).all()
    final.to_csv(OUT_PATH)
    print(f"Soumission écrite : {OUT_PATH}; replis KNN={fallbacks}", flush=True)
