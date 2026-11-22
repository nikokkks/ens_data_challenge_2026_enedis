"""Round 3 : pousser vol_weight/vol_cap encore plus loin, et ajouter les
petites idées UNE PAR UNE au meilleur combo (le combo complet du round 2 a
degrade -3.51 -> -2.10, donc empiler brutalement n'est pas la bonne
strategie -- on isole ce qui aide vraiment une fois ajoute au meilleur combo)."""
import json, os, sys
import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_round3_results.json"
V12_REF_SCREEN = 79.241
BEST_SO_FAR = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0)

CONFIGS = {
    "26_vol_weight_10":        {**BEST_SO_FAR, "vol_weight": 1.0},
    "27_vol_cap_04":           {**BEST_SO_FAR, "vol_cap": 0.4},
    "28_best_plus_lag24":      {**BEST_SO_FAR, "extra_lags": (24,)},
    "29_best_plus_rolling":    {**BEST_SO_FAR, "neighbor_rolling": True},
    "30_best_plus_k60":        {**BEST_SO_FAR, "k": 60},
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
