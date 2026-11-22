"""Test rapide V12 TabPFN : conserver 40 voisins et en ajouter 60 (k=100)."""
import json
import os
import numpy as np
import tabpfn_client
from tabpfn_client import TabPFNRegressor

from evaluate import load_data
import feature_engineering_bench as feb

RANDOM_STATE = 42


class MedianTabPFN:
    def __init__(self):
        self.reg = TabPFNRegressor(random_state=RANDOM_STATE)
    def fit(self, X, y):
        self.reg.fit(X, y)
        return self
    def predict(self, X):
        return self.reg.predict(X, output_type="median")


if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        from tabpfn_bench_gemini import API_KEY as token
    tabpfn_client.set_access_token(token)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]
    model = feb.make_v10_model(
        k=100, n_clusters=15, adaptive_k=True, use_euclid=True,
        use_seasonality_feat=True, use_full_calendar_feat=True,
        calendar_groups=("hour", "dow", "dom", "year"),
        use_periodic_copy_feat=True, use_periodic_consensus_feat=True,
        normalize_periodic_feat=True, include_periodic_availability=True,
        periodic_alpha_strength=.5, make_regressor=MedianTabPFN,
        random_state=RANDOM_STATE)
    scores = feb.evaluate_repeated_splits(
        X, y, donors, model, n_repeats=3, test_size=.1,
        random_state=RANDOM_STATE, n_val_sub=10,
        save_path="tabpfn_k100_progress.json")
    reference = np.array(json.load(open("tabpfn_five_improvements_results.json"))
                         ["I4_confidence_alpha"][:3])
    result = {
        "V12_k40_first3": reference.tolist(),
        "V12_k100_first3": scores.tolist(),
        "mean_k40": float(reference.mean()),
        "mean_k100": float(scores.mean()),
        "paired_delta_k100_minus_k40": float((scores-reference).mean()),
    }
    with open("tabpfn_k100_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
