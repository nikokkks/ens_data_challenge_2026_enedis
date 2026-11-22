"""
Round 5 (G0-G6) : copie du meilleur script TabPFN (tabpfn_bench_v10.py,
70.80 +/- 31.79 W) avec les 6 pistes proposées (par un second modèle,
"Gemini") testées individuellement sur EXACTEMENT le même protocole
(n_repeats=10, test_size=0.10, random_state=42, n_val_sub=10, k=40,
n_clusters=15, use_seasonality_feat=True -- la config V10 gagnante sert de
base à chaque variante) :

  G1 interpolation PCHIP au lieu de linéaire (baseline ET composante du blend)
  G2 alpha forcé à 1.0 (100% interpolation) sur les trous de longueur <= 3
  G3 seuil de corrélation resserré (0.85 au lieu de 0.15) pour les voisins
  G4 ensemble TabPFN(median) + CatBoost(loss MAE), poids 60/40
  G5 plafonnement supérieur post-hoc (quantile 99.5% observé x 1.2)
  G6 output_type="median" explicite (vs predict() par défaut) -- argument
     théorique : la médiane minimise l'espérance de l'erreur absolue, alors
     que predict() sans argument est probablement une moyenne (déjà
     effleuré dans tabpfn_bench2.py à petite échelle n=2, re-testé ici sur
     le protocole robuste n=10)

G0 = V10 pur, ré-exécuté ici comme contrôle (doit redonner ~70.80 W).
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


class TabPFNMedianRegressor:
    """Wrapper fit/predict qui demande la médiane plutôt que la moyenne à
    TabPFN (piste Gemini #6 -- la médiane minimise l'espérance de l'erreur
    absolue, cf. théorie de la décision)."""

    def __init__(self, random_state=RANDOM_STATE):
        self.reg = TabPFNRegressor(random_state=random_state)

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X, output_type="median")


class TabPFNCatBoostEnsemble:
    """Moyenne pondérée TabPFN(median) + CatBoost(loss MAE) -- piste Gemini
    #4. CatBoost via SafeCatBoost (repli constant si le segment observé est
    parfaitement plat, cf. xgboost_bench_round2.py -- sinon CatBoostError)."""

    def __init__(self, weight_tabpfn=0.6, random_state=RANDOM_STATE):
        from xgboost_bench_round2 import SafeCatBoost
        self.tabpfn = TabPFNRegressor(random_state=random_state)
        self.catboost = SafeCatBoost(
            iterations=200, depth=3, learning_rate=0.1, loss_function="MAE",
            random_state=random_state, verbose=False, thread_count=1)
        self.weight_tabpfn = weight_tabpfn

    def fit(self, X, y):
        self.tabpfn.fit(X, y)
        self.catboost.fit(X, y)
        return self

    def predict(self, X):
        p_tab = self.tabpfn.predict(X, output_type="median")
        p_cat = self.catboost.predict(X)
        return self.weight_tabpfn * p_tab + (1 - self.weight_tabpfn) * p_cat


def default_tabpfn():
    return TabPFNRegressor(random_state=RANDOM_STATE)


CONFIGS = {
    "G0_v10_baseline": dict(v11_kwargs=dict(), regressor=default_tabpfn),
    "G1_pchip_interp": dict(v11_kwargs=dict(interp_method="pchip"), regressor=default_tabpfn),
    "G2_force_full_interp_short_gaps": dict(v11_kwargs=dict(force_full_interp_below=3), regressor=default_tabpfn),
    "G3_tight_corr_threshold": dict(v11_kwargs=dict(corr_threshold=0.85), regressor=default_tabpfn),
    "G4_ensemble_tabpfn_catboost": dict(v11_kwargs=dict(), regressor=lambda: TabPFNCatBoostEnsemble(weight_tabpfn=0.6)),
    "G5_upper_capping": dict(v11_kwargs=dict(upper_cap_margin=1.2), regressor=default_tabpfn),
    "G6_median_explicit": dict(v11_kwargs=dict(), regressor=lambda: TabPFNMedianRegressor()),
}

RESULTS_PATH = "tabpfn_bench_gemini_results.json"


def load_results():
    import json, os
    if os.path.exists(RESULTS_PATH):
        return json.load(open(RESULTS_PATH))
    return {}


def save_result(name, scores):
    import json
    results = load_results()
    results[name] = scores.tolist()
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    names = [only] if only else list(CONFIGS.keys())

    print("Chargement des données...")
    X_train, y_train, X_test = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    for name in names:
        cfg = CONFIGS[name]
        model = feb.make_v11_model(
            k=K, n_clusters=N_CLUSTERS, adaptive_k=True, use_euclid=True,
            use_seasonality_feat=True, make_regressor=cfg["regressor"],
            random_state=RANDOM_STATE, **cfg["v11_kwargs"])
        print(f"=== {name} — {N_REPEATS} x hold-out 90/10, {N_VAL_SUB} colonnes val / repeat ===")
        scores = feb.evaluate_repeated_splits(
            X_train, y_train, donor_cols, model,
            n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE, n_val_sub=N_VAL_SUB,
            save_path=f"tabpfn_gemini_{name}_progress.json")
        save_result(name, scores)
        print(f"[{name}] sauvegardé dans {RESULTS_PATH}\n")
