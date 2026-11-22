"""
Round 4 : 8 pistes de feature engineering (E1-E8, issues de recherche --
PMM/MICE, "time aggregation features", décomposition saisonnière légère,
combinaison des meilleurs réglages round 2), construites sur LightGBM =
meilleur modèle a ce stade (cf. R1_lightgbm, xgboost_bench_round2_results.json,
75.95 +/- 7.31). Même harnais officiel EXACT que round 1/2/3 (n=5 hold-out
90/10 indépendants, graines différentes, MAE officielle NaN-safe, aucun
sous-échantillonnage des colonnes val -- 100 col/repeat) pour rester
directement comparable.

7 autres pistes (E9-E15) documentées mais différées, cf. réponse dans la
conversation (tuning hyperparams LightGBM, DTW, multi-output, propagation
2nd ordre, STL complet, deep learning SAITS/BRITS, recherche Optuna globale).
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
K = 20
N_CLUSTERS = 25

LGBM_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
                    colsample_bytree=0.8, objective="mae", n_jobs=1, verbosity=-1)


def make_lgbm():
    return LGBMRegressor(random_state=RANDOM_STATE, **LGBM_PARAMS)


def run(name, v10_kwargs=None, k=K, n_clusters=N_CLUSTERS):
    v10_kwargs = v10_kwargs or {}
    model = feb.make_v10_model(
        k=k, n_clusters=n_clusters, adaptive_k=True, use_euclid=True,
        make_regressor=make_lgbm, random_state=RANDOM_STATE, **v10_kwargs)
    print(f"=== {name} ===")
    t0 = time.time()
    scores = feb.evaluate_repeated_splits(
        X_train, y_train, donor_cols, model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"  total: {time.time()-t0:.1f}s\n")
    return scores


CONFIGS = {
    "E1_seasonality":         dict(v10_kwargs=dict(use_seasonality_feat=True)),
    "E2_neighbor_rolling":    dict(v10_kwargs=dict(use_neighbor_rolling_feat=True)),
    "E3_neighbor_consensus":  dict(v10_kwargs=dict(use_neighbor_consensus_feat=True)),
    "E4_pmm":                 dict(v10_kwargs=dict(use_pmm=True, pmm_blend=0.3)),
    "E5_kitchen_sink_k40_c15": dict(v10_kwargs=dict(), k=40, n_clusters=15),
    "E7_weekday_cyclic":      dict(v10_kwargs=dict(use_weekday_cyclic=True)),
    "E8_pchip_feat":          dict(v10_kwargs=dict(use_pchip_feat=True)),
    "E0_lgbm_baseline_v9":    dict(v10_kwargs=dict()),  # = R1_lightgbm, ré-exécuté ici pour vérif de cohérence
    # E6 : combinaison des 2 seules pistes retenues à l'ablation individuelle
    # (E1 seasonality: 75.39, E5 kitchen-sink k40/c15: 75.61, vs baseline 75.95)
    # -- même logique que V8->V9 (ne combiner que ce qui a tenu seul).
    "E6_combo_seasonality_kitchen_sink": dict(v10_kwargs=dict(use_seasonality_feat=True), k=40, n_clusters=15),
}

RESULTS_PATH = "xgboost_bench_round4_results.json"


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
        scores = run(name, cfg.get("v10_kwargs"), cfg.get("k", K), cfg.get("n_clusters", N_CLUSTERS))
        save_result(name, scores)
        print(f"[{name}] sauvegardé dans {RESULTS_PATH}")
