"""Round 2 : pousser plus loin les 2 meilleures idées du round 1
(volatility_alpha, periodic_alpha_strength) et tester leurs combinaisons,
plus retest du ratio_clip maintenant correctement câblé."""
import json, os, sys
import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_round2_results.json"
V12_REF_SCREEN = 79.241

CONFIGS = {
    "17b_ratio_clip_fixed": dict(ratio_clip=(0.3, 3.0)),
    "21_vol_weight_08":     dict(volatility_alpha=True, vol_weight=0.8),
    "22_periodic_alpha_10": dict(periodic_alpha_strength=1.0),
    "23_combo_top2":        dict(volatility_alpha=True, periodic_alpha_strength=0.8),
    "24_combo_top2_strong": dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0),
    "25_combo_full":        dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                                  extra_lags=(24, 672), neighbor_rolling=True, k=60, euclid_weight=0.5),
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
