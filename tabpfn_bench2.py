"""
V9 (TabPFN) + criblage de 3 pistes spécifiques à TabPFN parmi les 5
sélectionnées (les 2 autres -- distance euclidienne, alpha volatilité -- sont
génériques et testées dans _ablation_v9.py / feature_engineering_bench.make_v9_model) :

  11. output_type="median" au lieu de "mean" (conso non-négative, distribution
      d'erreur probablement asymétrique -> la médiane pourrait être plus robuste)
  12. n_estimators augmenté (8 -> 16) pour un ensembling interne plus stable
  13. Ensemble Ridge + TabPFN (moyenne pondérée des deux prédictions)

Criblage volontairement PETIT (n_repeats=2, n_val_sub=5) : un run complet
TabPFN sur X_test tourne déjà en parallèle (generate_submission_tabpfn.py,
~1h30) et chaque variante ici ajoute des appels API cloud -- on limite le
volume pour ne pas cumuler indéfiniment les coûts/latences pendant que
l'autre job tourne. À relancer plus large une fois ce job terminé si les
résultats sont prometteurs.

Toutes les variantes utilisent V9 (make_v9_model, euclid=True, volatility=
False -- la config validée par ablation) avec un `make_regressor` différent
à chaque fois -- même pipeline de features, seule la régression locale change.
"""

import os
import warnings
import numpy as np
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

import tabpfn_client
from tabpfn_client import TabPFNRegressor

from evaluate import load_data
import feature_engineering_bench as feb

RANDOM_STATE = 42
N_REPEATS = 2
TEST_SIZE = 0.10
N_VAL_SUB = 5  # criblage volontairement petit, cf. docstring
K = 20
N_CLUSTERS = 25

API_KEY = os.environ.get(
    "TABPFN_API_KEY",
    "tabpfn_sk_7NPt3kHS4ejPFsJCmqH09hEBXVxB8WRm_GXRnNTu5Ac",
)
tabpfn_client.set_access_token(API_KEY)


class TabPFNMedianRegressor:
    """Wrapper fit/predict qui demande la médiane plutôt que la moyenne à TabPFN."""

    def __init__(self, random_state=RANDOM_STATE):
        self.reg = TabPFNRegressor(random_state=random_state)

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X, output_type="median")


class RidgeTabPFNEnsemble:
    """Moyenne pondérée des prédictions Ridge et TabPFN (fit les deux, blend)."""

    def __init__(self, random_state=RANDOM_STATE, ridge_alpha=1.0, weight_tabpfn=0.6):
        self.ridge = Ridge(alpha=ridge_alpha)
        self.tabpfn = TabPFNRegressor(random_state=random_state)
        self.weight_tabpfn = weight_tabpfn

    def fit(self, X, y):
        self.ridge.fit(X, y)
        self.tabpfn.fit(X, y)
        return self

    def predict(self, X):
        p_ridge = self.ridge.predict(X)
        p_tab = self.tabpfn.predict(X)
        return self.weight_tabpfn * p_tab + (1 - self.weight_tabpfn) * p_ridge


VARIANTS = {
    "TabPFN défaut (n_estimators=8, mean)": lambda: TabPFNRegressor(random_state=RANDOM_STATE),
    # n_estimators=16 : rejeté par l'API (HTTP 422, "n_estimators <= 8" sur ce
    # tier) -- 8 est déjà le max autorisé, piste #12 infaisable telle quelle.
    "TabPFN median": lambda: TabPFNMedianRegressor(random_state=RANDOM_STATE),
    "Ridge+TabPFN ensemble (60/40)": lambda: RidgeTabPFNEnsemble(random_state=RANDOM_STATE),
}


if __name__ == "__main__":
    print("Chargement des données...")
    X_train, y_train, X_test = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    results = {}
    for name, make_regressor in VARIANTS.items():
        print(f"=== {name} — {N_REPEATS} x hold-out 90/10, {N_VAL_SUB} colonnes val / repeat ===")
        model = feb.make_v9_model(
            k=K, n_clusters=N_CLUSTERS, adaptive_k=True, use_euclid=True, use_volatility=False,
            make_regressor=make_regressor, random_state=RANDOM_STATE,
        )
        scores = feb.evaluate_repeated_splits(
            X_train, y_train, donor_cols, model,
            n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE, n_val_sub=N_VAL_SUB,
        )
        results[name] = scores

    print("=== Résumé (criblage rapide, n petit) ===")
    for name, scores in results.items():
        print(f"{name:38s} {scores.mean():8.2f} +/- {scores.std():6.2f} W  {np.round(scores, 1).tolist()}")

    import json
    with open("tabpfn_bench2_results.json", "w") as f:
        json.dump({k: v.tolist() for k, v in results.items()}, f, indent=2)
    print("\nRésultats sauvegardés dans tabpfn_bench2_results.json")
