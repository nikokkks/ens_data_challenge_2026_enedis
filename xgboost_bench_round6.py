"""
Round 6 : le meilleur modèle (LightGBM, config E6 = seasonality + n_clusters=15,
cf. xgboost_bench_round4.py, 74.91 +/- 7.30 W) avec un k (nombre de voisins
= features de régression) BEAUCOUP plus grand -- jusqu'ici k=40 restreignait
chaque colonne à ses 40 voisins les plus corrélés avant de les passer en
features à LightGBM. Un modèle en arbres tolère mieux un grand nombre de
features (sélection implicite via les splits, colsample_bytree=0.8 déjà en
place) qu'un modèle linéaire -- vaut le coup de tester si laisser LightGBM
voir beaucoup plus de voisins (jusqu'à ~2000) améliore le score plutôt que
de pré-filtrer aussi agressivement.

Même harnais officiel EXACT (n=5 hold-out 90/10 indépendants, graines
différentes, MAE officielle NaN-safe, 100 col val/repeat, n_clusters=15,
seasonality activée) -- SEUL k varie.
"""

import time
import warnings
import numpy as np
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

from evaluate import load_data
import feature_engineering_bench as feb

RANDOM_STATE = 42
N_REPEATS = 5
TEST_SIZE = 0.10
N_CLUSTERS = 15

LGBM_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
                    colsample_bytree=0.8, objective="mae", n_jobs=1, verbosity=-1)


def make_lgbm():
    return LGBMRegressor(random_state=RANDOM_STATE, **LGBM_PARAMS)


def run(name, k):
    model = feb.make_v10_model(
        k=k, n_clusters=N_CLUSTERS, adaptive_k=True, use_euclid=True,
        use_seasonality_feat=True, make_regressor=make_lgbm, random_state=RANDOM_STATE)
    print(f"=== {name} (k={k}) ===")
    t0 = time.time()
    scores = feb.evaluate_repeated_splits(
        X_train, y_train, donor_cols, model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"  total: {time.time()-t0:.1f}s\n")
    return scores


CONFIGS = {
    "H1_k100":  dict(k=100),
    "H2_k300":  dict(k=300),
    "H3_k1000": dict(k=1000),
    "H4_k2000": dict(k=2000),
}

RESULTS_PATH = "xgboost_bench_round6_results.json"


def load_results():
    import json, os
    if os.path.exists(RESULTS_PATH):
        return json.load(open(RESULTS_PATH))
    return {}


def save_result(name, scores):
    import json
    results = load_results()
    results[name] = scores.tolist()
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    names = [only] if only else list(CONFIGS.keys())

    print("Chargement des données (X_train, y_train)...")
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    for name in names:
        cfg = CONFIGS[name]
        scores = run(name, cfg["k"])
        save_result(name, scores)
        print(f"[{name}] sauvegardé dans {RESULTS_PATH}")
