"""Ablation TabPFN median des nouvelles features calendaires/périodiques.

Protocole strictement identique à tabpfn_bench_gemini.py : 10 répétitions,
split 90/10, 10 colonnes évaluées, graines 42..51, k=40, 15 clusters.
"""
import json
import os
import sys
import warnings

import numpy as np
import tabpfn_client
from tabpfn_client import TabPFNRegressor

from evaluate import load_data
import feature_engineering_bench as feb

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
RESULTS_PATH = "tabpfn_calendar_results.json"


class TabPFNMedianRegressor:
    def __init__(self):
        self.reg = TabPFNRegressor(random_state=RANDOM_STATE)

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X, output_type="median")


CONFIGS = {
    "C1_full_calendar": dict(
        use_seasonality_feat=True,
        use_full_calendar_feat=True,
    ),
    "C2_full_calendar_periodic_consensus": dict(
        use_seasonality_feat=True,
        use_full_calendar_feat=True,
        use_periodic_copy_feat=True,
        use_periodic_consensus_feat=True,
    ),
}


if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        # Compatibilité temporaire avec les anciens benchmarks du dépôt : ne
        # pas dupliquer le secret historique dans ce nouveau script.
        from tabpfn_bench_gemini import API_KEY as token
    tabpfn_client.set_access_token(token)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    names = [only] if only else list(CONFIGS)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]
    try:
        results = json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        results = {}
    for name in names:
        print(f"=== {name} ===", flush=True)
        model = feb.make_v10_model(
            k=40, n_clusters=15, adaptive_k=True, use_euclid=True,
            make_regressor=TabPFNMedianRegressor,
            random_state=RANDOM_STATE, **CONFIGS[name])
        scores = feb.evaluate_repeated_splits(
            X, y, donors, model, n_repeats=10, test_size=0.10,
            random_state=RANDOM_STATE, n_val_sub=10,
            save_path=f"tabpfn_{name}_progress.json")
        results[name] = scores.tolist()
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"{name}: {scores.mean():.3f} +/- {scores.std():.3f} W", flush=True)
