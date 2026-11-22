"""Round 4 : combiner les 2-3 meilleurs ajouts individuels du round 3."""
import json, os, sys
import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_round4_results.json"
V12_REF_SCREEN = 79.241
BASE = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0)

CONFIGS = {
    "31_base_lag24_rolling":        {**BASE, "extra_lags": (24,), "neighbor_rolling": True},
    "32_base_lag24_volcap04":       {**BASE, "extra_lags": (24,), "vol_cap": 0.4},
    "33_base_lag24_rolling_volcap04": {**BASE, "extra_lags": (24,), "neighbor_rolling": True, "vol_cap": 0.4},
}

if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant")
    tabpfn_client.set_access_token(token)
    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]
    try: results = json.load(open(RESULTS_PATH))
    except FileNotFoundError: results = {}
    names = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIGS)
    for name in names:
        opts = CONFIGS[name]
        print(f"=== SCREEN {name}: {opts} ===", flush=True)
        scores = v14.run_config(name, opts, X, y, donors, n_repeats=3, n_val_sub=10)
        delta = float(scores.mean() - V12_REF_SCREEN)
        print(f"  Delta vs V12 ref ({V12_REF_SCREEN}): {delta:+.2f} W\n", flush=True)
        results[name] = {"scores": scores.tolist(), "mean": float(scores.mean()),
                          "std": float(scores.std()), "delta_vs_v12": delta}
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)
    print("Terminé.", flush=True)
