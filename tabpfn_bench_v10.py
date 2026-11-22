"""
V10 (TabPFN) : même pipeline que tabpfn_bench.py (V8), mais avec les 2
nouvelles features validées en round 4 sur LightGBM/XGBoost (cf.
xgboost_bench_round4.py, E6_combo_seasonality_kitchen_sink = meilleur score
toutes sessions confondues, 74.91 +/- 7.30 W sur le protocole officiel) :

  - feature de saisonnalité journalière propre à la courbe (médiane des
    points OBSERVES de la même courbe au même horaire -- make_v10_model,
    use_seasonality_feat=True)
  - k=40 voisins (au lieu de 20) + n_clusters=15 (au lieu de 25) -- réglage
    "kitchen sink" validé en round 4 (E5)
  - hérite aussi de use_euclid=True (distance complémentaire à la
    corrélation, validée V9) qui n'était PAS encore dans tabpfn_bench.py
    (celui-ci utilisait make_v8_model, pas make_v9_model) -- migration V8->V10
    donc légèrement plus large que juste "les 2 nouvelles features round 4"

Même harnais EXACT que tabpfn_bench.py (même random_state=42, n_repeats=10,
test_size=0.10, n_val_sub=10) pour comparer ancien (V8) vs nouveau (V10) sur
un pied strictement égal : mêmes colonnes dev/val tirées à chaque repeat,
même régresseur TabPFN, seule la feature engineering + les hyperparamètres
k/n_clusters changent.
"""

import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import tabpfn_client
from tabpfn_client import TabPFNRegressor

from evaluate import load_data
import feature_engineering_bench as feb

RANDOM_STATE = 42
N_REPEATS = 10
TEST_SIZE = 0.10
N_VAL_SUB = 10
K = 40
N_CLUSTERS = 15

API_KEY = os.environ.get(
    "TABPFN_API_KEY",
    "tabpfn_sk_7NPt3kHS4ejPFsJCmqH09hEBXVxB8WRm_GXRnNTu5Ac",
)
tabpfn_client.set_access_token(API_KEY)


if __name__ == "__main__":
    print("Chargement des données...")
    X_train, y_train, X_test = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    print(f"=== V10_tabpfn (V9 KNN+cluster+euclid + saisonnalité + k=40/c15, régression locale = TabPFN) — "
          f"{N_REPEATS} x hold-out 90/10, {N_VAL_SUB} colonnes val notées / repeat ===")
    tabpfn_model = feb.make_v10_model(
        k=K, n_clusters=N_CLUSTERS, adaptive_k=True, use_euclid=True,
        use_seasonality_feat=True,
        make_regressor=lambda: TabPFNRegressor(random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
    )
    tabpfn_scores = feb.evaluate_repeated_splits(
        X_train, y_train, donor_cols, tabpfn_model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE, n_val_sub=N_VAL_SUB,
        save_path="tabpfn_v10_partial_progress.json",
    )

    print(f"=== V10_ridge (référence, mêmes colonnes échantillonnées, régression locale = Ridge) ===")
    from sklearn.linear_model import Ridge
    ridge_model = feb.make_v10_model(
        k=K, n_clusters=N_CLUSTERS, adaptive_k=True, use_euclid=True,
        use_seasonality_feat=True,
        make_regressor=lambda: Ridge(alpha=1.0), random_state=RANDOM_STATE,
    )
    ridge_scores = feb.evaluate_repeated_splits(
        X_train, y_train, donor_cols, ridge_model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE, n_val_sub=N_VAL_SUB,
    )

    print("=== Résumé V10 (nouvelles features) ===")
    print(f"V10 (Ridge local):    {ridge_scores.mean():.2f} +/- {ridge_scores.std():.2f} W "
          f"(par repeat: {np.round(ridge_scores, 1).tolist()})")
    print(f"V10 (TabPFN local):   {tabpfn_scores.mean():.2f} +/- {tabpfn_scores.std():.2f} W "
          f"(par repeat: {np.round(tabpfn_scores, 1).tolist()})")

    import json
    with open("tabpfn_v10_results.json", "w") as f:
        json.dump({
            "V10_ridge_local_subsample": ridge_scores.tolist(),
            "V10_tabpfn_local_subsample": tabpfn_scores.tolist(),
            "n_val_sub": N_VAL_SUB,
            "n_repeats": N_REPEATS,
        }, f, indent=2)
    print("\nRésultats sauvegardés dans tabpfn_v10_results.json")

    print("\n=== Comparaison ancien (V8) vs nouveau (V10), même protocole exact ===")
    old = json.load(open("tabpfn_benchmark_results.json"))
    old_ridge = np.array(old["V8_ridge_local_subsample"])
    old_tabpfn = np.array(old["V8_tabpfn_local_subsample"])
    rows = [
        ("Ridge  V8  (ancien)", old_ridge),
        ("Ridge  V10 (nouveau)", ridge_scores),
        ("TabPFN V8  (ancien)", old_tabpfn),
        ("TabPFN V10 (nouveau)", tabpfn_scores),
    ]
    for name, scores in sorted(rows, key=lambda kv: kv[1].mean()):
        print(f"{name:24s} {scores.mean():8.2f} +/- {scores.std():6.2f} W")
