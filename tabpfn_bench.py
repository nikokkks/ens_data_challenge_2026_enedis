"""
V8 (TabPFN) : exactement le même pipeline de features que V8 (Ridge, cf.
feature_engineering_bench.make_v8_model) -- KNN corrélation + cluster de
formes + K adaptatif par seuil de corrélation + alpha continu selon la
position dans le trou + scaling robuste -- mais avec TabPFNRegressor comme
régresseur local à la place de Ridge. Le régresseur est injecté
(`make_regressor`), donc c'est LITTÉRALEMENT le même code de features pour
les deux scripts : seule la régression locale change.

Comparabilité avec v8_ridge_bench.py : même fonction de harness
(feature_engineering_bench.evaluate_repeated_splits), même random_state=42,
n_repeats=10, test_size=0.10, n_val_sub=10 -> les deux scripts tirent
exactement les mêmes dev_cols/val_cols à chaque repeat. Pas de k-fold
(train_test_split indépendant par repeat, une graine différente = random_state+i).

Contrainte pratique : TabPFNRegressor via l'API cloud fait un aller-retour
réseau par colonne (fit + predict, ~4-7s/colonne) -> n_val_sub réduit (10
colonnes/repeat) plutôt que les 100 colonnes val complètes, pour rester dans
un temps/coût d'appel API raisonnable pour un test exploratoire.
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
K = 20
N_CLUSTERS = 25

API_KEY = os.environ.get(
    "TABPFN_API_KEY",
    "tabpfn_sk_7NPt3kHS4ejPFsJCmqH09hEBXVxB8WRm_GXRnNTu5Ac",
)
tabpfn_client.set_access_token(API_KEY)


if __name__ == "__main__":
    print("Chargement des données...")
    X_train, y_train, X_test = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    print(f"=== V8_tabpfn (KNN+cluster+K adaptatif+alpha continu, régression locale = TabPFN) — "
          f"{N_REPEATS} x hold-out 90/10, {N_VAL_SUB} colonnes val notées / repeat ===")
    tabpfn_model = feb.make_v8_model(
        k=K, n_clusters=N_CLUSTERS, adaptive_k=True,
        make_regressor=lambda: TabPFNRegressor(random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
    )
    tabpfn_scores = feb.evaluate_repeated_splits(
        X_train, y_train, donor_cols, tabpfn_model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE, n_val_sub=N_VAL_SUB,
        save_path="tabpfn_partial_progress.json",
    )

    print(f"=== V8_ridge (référence, mêmes colonnes échantillonnées, régression locale = Ridge) ===")
    from sklearn.linear_model import Ridge
    ridge_model = feb.make_v8_model(
        k=K, n_clusters=N_CLUSTERS, adaptive_k=True,
        make_regressor=lambda: Ridge(alpha=1.0), random_state=RANDOM_STATE,
    )
    ridge_scores = feb.evaluate_repeated_splits(
        X_train, y_train, donor_cols, ridge_model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE, n_val_sub=N_VAL_SUB,
    )

    print("=== Résumé ===")
    print(f"V8 (Ridge local):    {ridge_scores.mean():.2f} +/- {ridge_scores.std():.2f} W "
          f"(par repeat: {np.round(ridge_scores, 1).tolist()})")
    print(f"V8 (TabPFN local):   {tabpfn_scores.mean():.2f} +/- {tabpfn_scores.std():.2f} W "
          f"(par repeat: {np.round(tabpfn_scores, 1).tolist()})")

    import json
    with open("tabpfn_benchmark_results.json", "w") as f:
        json.dump({
            "V8_ridge_local_subsample": ridge_scores.tolist(),
            "V8_tabpfn_local_subsample": tabpfn_scores.tolist(),
            "n_val_sub": N_VAL_SUB,
            "n_repeats": N_REPEATS,
        }, f, indent=2)
    print("\nRésultats sauvegardés dans tabpfn_benchmark_results.json")
