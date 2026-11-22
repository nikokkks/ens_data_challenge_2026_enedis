import json, os
import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v17_donorfeat_results.json"
V12_REF_SCREEN = 79.241
BEST_V14 = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                extra_lags=(24,), neighbor_rolling=True)

CONFIGS = {
    "44_neighbor_mad":       {**BEST_V14, "neighbor_mad_feat": True},
    "45_neighbor_interval":  {**BEST_V14, "neighbor_interval_feat": True},
    "46_peak_offpeak":       {**BEST_V14, "peak_offpeak_feat": True},
    "47_collective_index":   {**BEST_V14, "collective_index_feat": True},
}

if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    tabpfn_client.set_access_token(token)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]
    try: results = json.load(open(RESULTS_PATH))
    except FileNotFoundError: results = {}
    for name, opts in CONFIGS.items():
        print(f"=== SCREEN {name}: {opts} ===", flush=True)
        scores = v14.run_config(name, opts, X, y, donors, n_repeats=3, n_val_sub=10)
        delta = float(scores.mean() - V12_REF_SCREEN)
        print(f"  Delta vs V12 ref: {delta:+.2f} W\n", flush=True)
        results[name] = {"scores": scores.tolist(), "mean": float(scores.mean()),
                          "std": float(scores.std()), "delta_vs_v12": delta}
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)
    print("Terminé.", flush=True)
