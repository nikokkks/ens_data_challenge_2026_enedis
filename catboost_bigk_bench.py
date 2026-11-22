"""Meilleur script CatBoost (advanced_regressors_bench.py, R1_catboost_mae,
features V12/I4 = BEST_FEATURES) mais avec k=10000 voisins (au lieu de 40) :
au lieu de ne garder que les 40 courbes donneuses les plus corrélées comme
features de la régression locale, on en garde les 10 000 les plus pertinentes
parmi les ~20 000 courbes donneuses disponibles (sélection faite par
`adaptive_top_k_corr_euclid`, identique à l'existant, seul k_max change).

Même régresseur EXACT que R1_catboost_mae (mêmes hyperparamètres CatBoost),
mêmes features V12 (BEST_FEATURES), même protocole d'évaluation que le
meilleur script TabPFN (tabpfn_five_improvements_bench.py) :
    n_repeats=10, test_size=0.10, random_state=42, n_val_sub=10.

Affiche la comparaison face au meilleur score CatBoost précédent
(R1_catboost_mae, k=40, cf. advanced_regressors_results.json) APRES CHAQUE
repeat (et pas seulement à la fin), pour pouvoir juger tôt si le gain est
réel ou marginal vu le coût (fit CatBoost sur 10 000+ features).
"""
import json
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from evaluate import load_data
import feature_engineering_bench as feb
from xgboost_bench_round2 import SafeCatBoost

RANDOM_STATE = 42
N_REPEATS = 10
TEST_SIZE = 0.10
N_VAL_SUB = 10
N_CLUSTERS = 15
K_BIG = 10000
K_BASELINE = 40

RESULTS_PATH = "catboost_bigk_results.json"
PROGRESS_PATH = "catboost_bigk_progress.json"
PREV_RESULTS_PATH = "advanced_regressors_results.json"
PREV_KEY = "R1_catboost_mae"


def catboost():
    # hyperparamètres identiques à advanced_regressors_bench.py::catboost()
    return SafeCatBoost(iterations=250, depth=4, learning_rate=.06,
        loss_function="MAE", random_state=RANDOM_STATE, verbose=False,
        thread_count=1, allow_writing_files=False)


BEST_FEATURES = dict(
    use_seasonality_feat=True, use_full_calendar_feat=True,
    calendar_groups=("hour", "dow", "dom", "year"),
    use_periodic_copy_feat=True, use_periodic_consensus_feat=True,
    normalize_periodic_feat=True, include_periodic_availability=True,
    periodic_alpha_strength=.5,
)


def run_bigk(X_train, y_train, donor_cols, start_repeat=0, scores_so_far=None):
    from sklearn.model_selection import train_test_split
    from evaluate import official_mae

    prev = json.load(open(PREV_RESULTS_PATH))[PREV_KEY]

    model = feb.make_v10_model(k=K_BIG, n_clusters=N_CLUSTERS, adaptive_k=True,
        use_euclid=True, make_regressor=catboost, random_state=RANDOM_STATE,
        **BEST_FEATURES)

    holed_cols = list(y_train.columns)
    scores = list(scores_so_far) if scores_so_far else []

    for i in range(start_repeat, N_REPEATS):
        seed_i = RANDOM_STATE + i
        dev_cols, val_cols_full = train_test_split(holed_cols, test_size=TEST_SIZE, random_state=seed_i)
        sub_rng = np.random.RandomState(seed_i)
        val_cols = list(sub_rng.choice(val_cols_full, size=N_VAL_SUB, replace=False))

        t0 = time.time()
        y_pred_val = model(
            donor_df=X_train[donor_cols],
            dev_holed_df=X_train[dev_cols],
            dev_true_df=y_train[dev_cols],
            val_holed_df=X_train[val_cols],
        )
        mask = X_train[val_cols].isna()
        mae = official_mae(y_train[val_cols], y_pred_val, mask)
        dt = time.time() - t0
        scores.append(mae)

        prev_i = prev[i]
        delta = mae - prev_i
        pct = 100 * delta / prev_i
        print(f"  repeat {i + 1}/{N_REPEATS}: k=10000 MAE = {mae:8.2f} W  "
              f"| k=40 (R1_catboost) = {prev_i:8.2f} W  | delta = {delta:+7.2f} W ({pct:+.1f}%)  "
              f"({dt:.1f}s)", flush=True)

        with open(PROGRESS_PATH, "w") as f:
            json.dump({"scores_so_far": scores, "n_repeats_done": i + 1,
                       "n_repeats_target": N_REPEATS}, f)

    scores = np.array(scores)
    prev_arr = np.array(prev[:len(scores)])
    print(f"\n  -> k=10000 : Mean MAE over {len(scores)} repeats: "
          f"{scores.mean():.2f} +/- {scores.std():.2f} W")
    print(f"  -> k=40 (R1_catboost, meme repeats) : Mean MAE: "
          f"{prev_arr.mean():.2f} +/- {prev_arr.std():.2f} W")
    diff = scores.mean() - prev_arr.mean()
    print(f"  -> delta k=10000 vs k=40 : {diff:+.2f} W ({100*diff/prev_arr.mean():+.1f}%)")
    n_better = int((scores < prev_arr).sum())
    print(f"  -> k=10000 meilleur sur {n_better}/{len(scores)} repeats appariés\n")
    return scores


if __name__ == "__main__":
    print("Chargement des données (X_train, y_train)...")
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]
    print(f"{len(donor_cols)} courbes donneuses disponibles, k={K_BIG} retenues par colonne à compléter.")

    resume = len(sys.argv) > 1 and sys.argv[1] == "resume"
    start_repeat = 0
    scores_so_far = []
    if resume:
        try:
            prog = json.load(open(PROGRESS_PATH))
            scores_so_far = prog["scores_so_far"]
            start_repeat = prog["n_repeats_done"]
            print(f"Reprise depuis {PROGRESS_PATH} : {start_repeat} repeats déjà faits.")
        except FileNotFoundError:
            pass

    scores = run_bigk(X_train, y_train, donor_cols, start_repeat, scores_so_far)

    try:
        results = json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        results = {}
    results["catboost_k10000"] = scores.tolist()
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Résultats sauvegardés dans {RESULTS_PATH}")
