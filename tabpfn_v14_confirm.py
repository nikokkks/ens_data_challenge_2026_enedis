"""Confirmation 10x10 du meilleur combo V14 face a V12 (66.24 +/- 32.45 W)."""
import json, os
import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_confirm_results.json"
BEST = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
            extra_lags=(24,), neighbor_rolling=True)

if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant")
    tabpfn_client.set_access_token(token)
    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]
    print(f"=== CONFIRM v14_best: {BEST} ===", flush=True)
    scores = v14.run_config("v14_best", BEST, X, y, donors, n_repeats=10, n_val_sub=10)
    result = {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}
    json.dump(result, open(RESULTS_PATH, "w"), indent=2)
    print("Terminé.", flush=True)
