"""Lance le criblage des 20 idées V14 (cf. tabpfn_v14_ablation.py), une par
une, sur l'écran apparié 3x10 (référence V12 : 79.24 +/- 31.09 W)."""
import json
import os
import sys

import tabpfn_client

from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = v14.RESULTS_PATH
V12_REF_SCREEN = 79.241  # cf. benchmark.md §9, memes graines 3x10

CONFIGS = {
    "01_corr_thr_010":      dict(corr_threshold=0.10),
    "02_corr_thr_025":      dict(corr_threshold=0.25),
    "03_nclusters_25":      dict(n_clusters=25),
    "04_nclusters_08":      dict(n_clusters=8),
    "05_euclid_w_015":      dict(euclid_weight=0.15),
    "06_euclid_w_050":      dict(euclid_weight=0.5),
    "07_neighbor_rolling":  dict(neighbor_rolling=True),
    "08_neighbor_consensus": dict(neighbor_consensus=True),
    "09_pmm_snap":          dict(pmm=True, pmm_blend=0.3),
    "10_periodic_alpha_08": dict(periodic_alpha_strength=0.8),
    "11_periodic_alpha_025": dict(periodic_alpha_strength=0.25),
    "12_extra_lag_24":      dict(extra_lags=(24,)),
    "13_extra_lag_672":     dict(extra_lags=(672,)),
    "14_seed_ensemble":     dict(regressor_cls=v14.SeedEnsembleTabPFN, regressor_kwargs={}),
    "15_mean_median_blend": dict(regressor_cls=v14.MeanMedianBlendTabPFN, regressor_kwargs={"mean_weight": 0.3}),
    "16_volatility_alpha":  dict(volatility_alpha=True),
    "17_ratio_clip_tight":  dict(ratio_clip=(0.3, 3.0)),
    "18_k_min_10":          dict(k_min=10),
    "19_k_60":              dict(k=60),
    "20_k_30":              dict(k=30),
}


if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant")
    tabpfn_client.set_access_token(token)

    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]

    try:
        results = json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        results = {}

    names = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIGS)
    for name in names:
        opts = CONFIGS[name]
        print(f"=== SCREEN {name}: {opts} ===", flush=True)
        scores = v14.run_config(name, opts, X, y, donors, n_repeats=3, n_val_sub=10)
        delta = float(scores.mean() - V12_REF_SCREEN)
        print(f"  Delta vs V12 ref ({V12_REF_SCREEN}): {delta:+.2f} W\n", flush=True)
        results[name] = {"scores": scores.tolist(), "mean": float(scores.mean()),
                          "std": float(scores.std()), "delta_vs_v12": delta, "opts": {
                              k: (v if not callable(v) else str(v)) for k, v in opts.items()}}
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    print("Terminé.", flush=True)
