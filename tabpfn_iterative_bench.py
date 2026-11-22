"""Idée A4 (imputation itérative type MICE) portée sur le MEILLEUR modèle du
dépôt : TabPFN I4 (66,24 +/- 32,45 W, `tabpfn_five_improvements_bench.py`).

A4 a été retenue au screening LightGBM du round 7 (69,75 -> 68,89 W, meilleure
sur 7/10 repeats appariés). Le principe : après une première passe, les trous
sont comblés provisoirement, si bien que les copies périodiques t±48/±96/±336
-- qui étaient indisponibles là où le jour précédent/suivant était lui-même
troué -- redeviennent renseignées pour la seconde passe.

Protocole identique à la baseline comparée : 10 repeats x 10 colonnes,
test_size=0.10, random_state=42 -> mêmes colonnes de validation que TabPFN I4.
"""

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import tabpfn_client
from tabpfn_client import TabPFNRegressor

from evaluate import load_data, official_mae
import research_ideas as ri

RANDOM_STATE = 42
N_REPEATS = 10
TEST_SIZE = 0.10
N_VAL_SUB = 10
K = 40
N_CLUSTERS = 15

RESULTS_PATH = "tabpfn_iterative_results.json"
BASELINE_KEY = "I4_confidence_alpha"
BASELINE_PATH = "tabpfn_five_improvements_results.json"


class MedianTabPFN:
    """Régresseur identique à celui de I4 (sortie médiane, cohérente MAE)."""

    def __init__(self):
        self.reg = TabPFNRegressor(random_state=RANDOM_STATE)

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X, output_type="median")


CONFIGS = {
    "T2_iterative_2pass": dict(n_passes=2),
    "T3_iterative_3pass": dict(n_passes=3),
}


def progress_path(name):
    return f"tabpfn_iter_{name}_progress.json"


def run(name, kwargs, X_train, y_train, donor_cols):
    from sklearn.model_selection import train_test_split

    baseline = json.load(open(BASELINE_PATH))[BASELINE_KEY]

    scores, start = [], 0
    if os.path.exists(progress_path(name)):
        prog = json.load(open(progress_path(name)))
        scores, start = prog["scores_so_far"], prog["n_repeats_done"]
        print(f"  reprise : {start} repeats déjà faits", flush=True)

    model = ri.make_research_model(k=K, n_clusters=N_CLUSTERS,
        make_regressor=MedianTabPFN, random_state=RANDOM_STATE, **kwargs)

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
        print(f"  repeat {i+1}/{N_REPEATS}: {mae:8.2f} W | TabPFN I4 = {baseline[i]:8.2f} W"
              f" | delta = {d:+6.2f} W ({100*d/baseline[i]:+5.1f}%)   "
              f"({time.time()-t0:.1f}s)", flush=True)
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
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        from tabpfn_bench_gemini import API_KEY as token
    tabpfn_client.set_access_token(token)

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
        scores = run(name, CONFIGS[name], X_train, y_train, donor_cols)
        results[name] = scores.tolist()
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
