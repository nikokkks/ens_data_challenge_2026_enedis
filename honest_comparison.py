"""
Comparaison honnête TabPFN vs XGBoost vs LightGBM : les 3 sur EXACTEMENT le
même protocole (celui utilisé pour TabPFN, cf. tabpfn_bench.py) --
n_repeats=10, test_size=0.10, random_state=42 (10 hold-out 90/10
indépendants, graines différentes), n_val_sub=10 colonnes val notées par
repeat (même sous-échantillon, mêmes graines -> mêmes colonnes tirées pour
les 3 modèles). Contrairement aux benchmarks XGBoost/LightGBM du reste du
projet (100 colonnes x 5 repeats), ici on aligne XGBoost/LightGBM sur
l'échelle de TabPFN plutôt que l'inverse (moins cher : pas d'appel API).
"""

import time
import warnings
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

from evaluate import load_data
import feature_engineering_bench as feb

RANDOM_STATE = 42
N_REPEATS = 10
TEST_SIZE = 0.10
N_VAL_SUB = 10
K = 20
N_CLUSTERS = 25

X1_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, objective="reg:absoluteerror")
LGBM_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
                    colsample_bytree=0.8, objective="mae", n_jobs=1, verbosity=-1)

if __name__ == "__main__":
    print("Chargement des données...")
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    configs = {
        "XGBoost_T1": lambda: XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, **X1_PARAMS),
        "LightGBM_R1": lambda: LGBMRegressor(random_state=RANDOM_STATE, **LGBM_PARAMS),
    }

    results = {}
    for name, make_regressor in configs.items():
        model = feb.make_v9_model(k=K, n_clusters=N_CLUSTERS, adaptive_k=True, use_euclid=True,
                                   use_volatility=False, make_regressor=make_regressor, random_state=RANDOM_STATE)
        print(f"=== {name} (protocole TabPFN : 10 repeats x 10 col) ===")
        t0 = time.time()
        scores = feb.evaluate_repeated_splits(
            X_train, y_train, donor_cols, model,
            n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE, n_val_sub=N_VAL_SUB)
        print(f"  total: {time.time()-t0:.1f}s\n")
        results[name] = scores

        import json
        with open("honest_comparison_results.json", "w") as f:
            json.dump({k: v.tolist() for k, v in results.items()}, f, indent=2)

    print("=== Résumé (même protocole que TabPFN) ===")
    import json
    tabpfn_results = json.load(open("tabpfn_benchmark_results.json"))
    results["TabPFN_V8"] = np.array(tabpfn_results["V8_tabpfn_local_subsample"])
    results["Ridge_V8"] = np.array(tabpfn_results["V8_ridge_local_subsample"])
    for name, scores in sorted(results.items(), key=lambda kv: np.mean(kv[1])):
        scores = np.array(scores)
        print(f"{name:20s} {scores.mean():8.2f} +/- {scores.std():6.2f}  (par repeat: {np.round(scores,1).tolist()})")
