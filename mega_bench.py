"""Comparaison avant/après : LightGBM + A4 (itératif 2 passes) SANS puis
AVEC les 53 features mega_features.py, protocole EXACT de
research_ideas_bench.py (10x10, mêmes graines) pour comparaison directe
avec A4_iterative_2pass (68.89 W) déjà stocké."""
import json
import sys
import time

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from evaluate import load_data, official_mae
import mega_model as mm

RANDOM_STATE = 42
N_REPEATS = 10
TEST_SIZE = 0.10
N_VAL_SUB = 10
K = 40
N_CLUSTERS = 15
RESULTS_PATH = "mega_bench_results.json"


def lgbm():
    return LGBMRegressor(objective="mae", n_estimators=200, max_depth=3,
        learning_rate=.1, subsample=.8, colsample_bytree=.8,
        random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)


def run(use_mega, label, X_train, y_train, donor_cols):
    model = mm.make_mega_model(k=K, n_clusters=N_CLUSTERS, make_regressor=lgbm,
                                random_state=RANDOM_STATE, n_passes=2, use_mega=use_mega)
    holed_cols = list(y_train.columns)
    scores = []
    for i in range(N_REPEATS):
        seed_i = RANDOM_STATE + i
        dev_cols, val_full = train_test_split(holed_cols, test_size=TEST_SIZE, random_state=seed_i)
        val_cols = list(np.random.RandomState(seed_i).choice(val_full, size=N_VAL_SUB, replace=False))
        t0 = time.time()
        y_pred = model(donor_df=X_train[donor_cols], dev_holed_df=X_train[dev_cols],
                       dev_true_df=y_train[dev_cols], val_holed_df=X_train[val_cols])
        mae = official_mae(y_train[val_cols], y_pred, X_train[val_cols].isna())
        scores.append(mae)
        print(f"  [{label}] repeat {i+1}/{N_REPEATS}: {mae:8.2f} W  ({time.time()-t0:.1f}s)", flush=True)
    scores = np.array(scores)
    print(f"  -> [{label}] {scores.mean():.2f} +/- {scores.std():.2f} W\n", flush=True)
    return scores


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    print("Chargement des données...", flush=True)
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    try:
        results = json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        results = {}

    if which in ("control", "both"):
        print("=== CONTROL (sans mega, doit reproduire A4_iterative_2pass) ===", flush=True)
        scores = run(False, "control", X_train, y_train, donor_cols)
        results["control_no_mega"] = scores.tolist()
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    if which in ("mega", "both"):
        print("=== AVEC 53 FEATURES MEGA ===", flush=True)
        scores = run(True, "mega", X_train, y_train, donor_cols)
        results["with_mega_53feat"] = scores.tolist()
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    if "control_no_mega" in results and "with_mega_53feat" in results:
        c = np.array(results["control_no_mega"])
        m = np.array(results["with_mega_53feat"])
        print(f"\n=== RESUME ===")
        print(f"control (sans mega) : {c.mean():.2f} +/- {c.std():.2f} W")
        print(f"avec mega (53 feat) : {m.mean():.2f} +/- {m.std():.2f} W")
        print(f"delta : {m.mean()-c.mean():+.2f} W ({100*(m.mean()-c.mean())/c.mean():+.2f}%)")
        print(f"mega meilleur sur {int((m<c).sum())}/{N_REPEATS} repeats")
