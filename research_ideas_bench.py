"""Screening des idées issues de la recherche (round 7).

Protocole : EXACTEMENT celui des baselines comparées -- `10 repeats x 10
courbes`, `test_size=0.10`, `random_state=42` (mêmes graines, donc mêmes
colonnes de validation à chaque repeat que R0/R1 et que TabPFN I4).

Baseline appariée : `R0_lightgbm_reference` (69,75 +/- 31,80 W), c'est-à-dire
le MEME régresseur (LightGBM MAE) et les MEMES features V12 -- seule l'idée
testée change. Le meilleur modèle absolu du dépôt (TabPFN I4, 66,24 W) est
rappelé à titre indicatif : il n'est pas la baseline du screening puisqu'il
change aussi de régresseur.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

from evaluate import load_data, official_mae
import research_ideas as ri

RANDOM_STATE = 42
N_REPEATS = 10
TEST_SIZE = 0.10
N_VAL_SUB = 10
K = 40
N_CLUSTERS = 15

RESULTS_PATH = "research_ideas_results.json"
BASELINE_PATH = "advanced_regressors_results.json"
BASELINE_KEY = "R0_lightgbm_reference"


def lgbm():
    return LGBMRegressor(objective="mae", n_estimators=200, max_depth=3,
        learning_rate=.1, subsample=.8, colsample_bytree=.8,
        random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)


def catboost():
    from xgboost_bench_round2 import SafeCatBoost
    return SafeCatBoost(iterations=250, depth=4, learning_rate=.06,
        loss_function="MAE", random_state=RANDOM_STATE, verbose=False,
        thread_count=1, allow_writing_files=False)


CONFIGS = {
    # contrôle : doit reproduire R0_lightgbm_reference au centième près
    "A0_control":            dict(),
    "A1_aggregates_500":     dict(n_aggregate=500),
    "A2_pca_rank20":         dict(pca_rank=20),
    # A3 avec gap_window=96 était un NO-OP : un voisinage de +/-96 pas autour
    # des trous couvre 100 % des points observés (les trous sont partout).
    # Seules des fenêtres courtes changent réellement le classement.
    "A3_gap_window_96":      dict(gap_window=96),
    "A3b_gap_window_12":     dict(gap_window=12),
    "A3c_gap_window_6":      dict(gap_window=6),
    "A4_iterative_2pass":    dict(n_passes=2),
    # approfondissement de A4 (seule idée retenue au screening)
    "A4b_iterative_3pass":   dict(n_passes=3),
    "A4c_iter2_catboost":    dict(n_passes=2, _regressor=catboost),
    "A4d_iter2_aggregates":  dict(n_passes=2, n_aggregate=500),
}


def progress_path(name):
    return f"research_{name}_progress.json"


def run(name, kwargs):
    from sklearn.model_selection import train_test_split

    baseline = json.load(open(BASELINE_PATH))[BASELINE_KEY]
    tabpfn = json.load(open("tabpfn_five_improvements_results.json"))["I4_confidence_alpha"]

    scores, start = [], 0
    if os.path.exists(progress_path(name)):
        prog = json.load(open(progress_path(name)))
        scores, start = prog["scores_so_far"], prog["n_repeats_done"]
        print(f"  reprise : {start} repeats déjà faits", flush=True)

    kwargs = dict(kwargs)
    regressor = kwargs.pop("_regressor", lgbm)
    model = ri.make_research_model(k=K, n_clusters=N_CLUSTERS,
        make_regressor=regressor, random_state=RANDOM_STATE, **kwargs)

    holed_cols = list(y_train.columns)
    for i in range(start, N_REPEATS):
        seed_i = RANDOM_STATE + i
        dev_cols, val_full = train_test_split(holed_cols, test_size=TEST_SIZE,
                                              random_state=seed_i)
        val_cols = list(np.random.RandomState(seed_i).choice(val_full, size=N_VAL_SUB,
                                                             replace=False))
        t0 = time.time()
        y_pred = model(donor_df=X_train[donor_cols], dev_holed_df=X_train[dev_cols],
                       dev_true_df=y_train[dev_cols], val_holed_df=X_train[val_cols])
        mae = official_mae(y_train[val_cols], y_pred, X_train[val_cols].isna())
        scores.append(mae)
        d = mae - baseline[i]
        print(f"  repeat {i+1}/{N_REPEATS}: {mae:8.2f} W | R0_lightgbm = {baseline[i]:8.2f} W"
              f" | delta = {d:+6.2f} W ({100*d/baseline[i]:+5.1f}%)"
              f" | TabPFN I4 = {tabpfn[i]:7.2f} W   ({time.time()-t0:.1f}s)", flush=True)
        with open(progress_path(name), "w") as f:
            json.dump({"scores_so_far": scores, "n_repeats_done": i + 1,
                       "n_repeats_target": N_REPEATS}, f)

    s, b = np.array(scores), np.array(baseline)
    print(f"\n  {name:22s} : {s.mean():.2f} +/- {s.std():.2f} W")
    print(f"  {BASELINE_KEY:22s} : {b.mean():.2f} +/- {b.std():.2f} W")
    print(f"  delta moyen : {s.mean()-b.mean():+.2f} W "
          f"({100*(s.mean()-b.mean())/b.mean():+.2f}%) | meilleur sur "
          f"{int((s < b).sum())}/{N_REPEATS} repeats appariés\n", flush=True)
    return s


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    print("Chargement des données...", flush=True)
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    try:
        results = json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        results = {}

    for name in ([only] if only else CONFIGS):
        print(f"=== {name} ===", flush=True)
        scores = run(name, CONFIGS[name])
        results[name] = scores.tolist()
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
