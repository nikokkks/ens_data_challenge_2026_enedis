"""
V8 (Ridge) : V5 + 3 améliorations sélectionnées parmi les 15 pistes de
benchmark.md (cf. feature_engineering_bench.make_v8_model pour le détail) :
  - alpha continu selon la position dans le trou (bord vs milieu)
  - lags directs (valeur/distance au point connu le plus proche) en feature
  - K adaptatif par seuil de corrélation

Même harness que tabpfn_bench.py (evaluate_repeated_splits : n=10 hold-out
90/10 indépendants, graines différentes, PAS de k-fold) et mêmes colonnes
val échantillonnées à chaque repeat (n_val_sub=10, même graine) -> les deux
scripts sont directement comparables, à lancer en parallèle.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import Ridge

from evaluate import load_data
import feature_engineering_bench as feb

RANDOM_STATE = 42
N_REPEATS = 10
TEST_SIZE = 0.10
N_VAL_SUB = 10  # même échantillon que tabpfn_bench.py, pour une comparaison tête-à-tête

if __name__ == "__main__":
    print("Chargement des données...")
    X_train, y_train, X_test = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    print(f"=== V5_ridge_reference (pour rappel) — {N_REPEATS} x hold-out 90/10, "
          f"{N_VAL_SUB} colonnes val notées / repeat ===")
    v5_model = feb.make_stepped_model(k=20, use_cluster=True, adaptive_alpha=True,
                                       use_ridge=True, robust_scale=True, n_clusters=25)
    v5_scores = feb.evaluate_repeated_splits(
        X_train, y_train, donor_cols, v5_model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE, n_val_sub=N_VAL_SUB,
    )

    print(f"=== V8_ridge (+alpha continu +lags directs +K adaptatif) — {N_REPEATS} x hold-out 90/10, "
          f"{N_VAL_SUB} colonnes val notées / repeat ===")
    v8_model = feb.make_v8_model(k=20, n_clusters=25, adaptive_k=True,
                                  make_regressor=lambda: Ridge(alpha=1.0), random_state=RANDOM_STATE)
    v8_scores = feb.evaluate_repeated_splits(
        X_train, y_train, donor_cols, v8_model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE, n_val_sub=N_VAL_SUB,
        save_path="v8_ridge_partial_progress.json",
    )

    print("=== Résumé ===")
    print(f"V5 (Ridge, référence):        {v5_scores.mean():.2f} +/- {v5_scores.std():.2f} W "
          f"(par repeat: {np.round(v5_scores, 1).tolist()})")
    print(f"V8 (Ridge, +3 améliorations): {v8_scores.mean():.2f} +/- {v8_scores.std():.2f} W "
          f"(par repeat: {np.round(v8_scores, 1).tolist()})")
    gain = 100 * (1 - v8_scores.mean() / v5_scores.mean())
    print(f"Gain relatif V8 vs V5: {gain:.1f}%")

    import json
    with open("v8_ridge_results.json", "w") as f:
        json.dump({
            "V5_ridge_reference": v5_scores.tolist(),
            "V8_ridge": v8_scores.tolist(),
            "n_val_sub": N_VAL_SUB,
            "n_repeats": N_REPEATS,
        }, f, indent=2)
    print("\nRésultats sauvegardés dans v8_ridge_results.json")
