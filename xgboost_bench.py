"""
Itérations d'amélioration incrémentale avec XGBoost comme régresseur local,
sur le pipeline V9 (KNN corrélation + cluster de formes + K adaptatif +
distance euclidienne + alpha continu par position dans le trou).

Méthodologie IMPOSÉE (cf. énoncé du challenge et consignes) :
  - n_repeats = 5, test_size = 0.10, random_state = 42
    -> 5 hold-out 90/10 INDÉPENDANTS, chacun avec sa propre graine
       (random_state + i = 42, 43, 44, 45, 46) -- pas de k-fold.
  - Métrique = MAE officielle du challenge (moyenne des erreurs absolues,
    UNIQUEMENT sur les positions réellement manquantes de X_train, en
    excluant les rares positions où y_train est lui-même NaN -- cf.
    evaluate.official_mae). Reproduit exactement
    `X_test_holed.apply(fill_nan_with_interpolation, axis=0)` comme
    baseline de référence (V0).
  - 100 colonnes val par repeat (échelle complète, pas de sous-échantillon --
    XGBoost tourne en local, ~0.3s/colonne, donc pas de contrainte de coût
    d'API comme pour TabPFN).

X0 = V9 + XGBRegressor (hyperparamètres par défaut raisonnables pour la
     régression per-courbe, ~900-1050 lignes d'entraînement).
X1..Xn = X0 + améliorations testées une par une par ablation, gardées
     seulement si elles battent X0 sur les 5 repeats.
"""

import time
import warnings
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

from evaluate import load_data
import feature_engineering_bench as feb

RANDOM_STATE = 42
N_REPEATS = 5
TEST_SIZE = 0.10
K = 20
N_CLUSTERS = 25


class XGBEarlyStopRegressor:
    """Réserve les derniers val_frac% des points observés (chronologiquement)
    comme validation interne pour l'early stopping -- capacité (n_estimators)
    choisie automatiquement par courbe plutôt que fixée à l'avance."""

    def __init__(self, val_frac=0.15, random_state=RANDOM_STATE, **xgb_kwargs):
        self.val_frac = val_frac
        self.random_state = random_state
        self.xgb_kwargs = xgb_kwargs

    def fit(self, X, y):
        n = len(y)
        n_val = max(10, int(n * self.val_frac))
        if n_val >= n - 10:  # trop peu de données pour early-stopper proprement
            self.reg = XGBRegressor(random_state=self.random_state, n_jobs=1, **self.xgb_kwargs)
            self.reg.fit(X, y)
            return self
        X_tr, y_tr = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]
        try:
            self.reg = XGBRegressor(random_state=self.random_state, n_jobs=1,
                                     early_stopping_rounds=20, **self.xgb_kwargs)
            self.reg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        except TypeError:
            self.reg = XGBRegressor(random_state=self.random_state, n_jobs=1, **self.xgb_kwargs)
            self.reg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict(self, X):
        return self.reg.predict(X)


class XGBRidgeEnsemble:
    """Moyenne pondérée XGBoost + Ridge (idée déjà testée pour TabPFN, ici
    pour XGBoost)."""

    def __init__(self, random_state=RANDOM_STATE, weight_xgb=0.7, **xgb_kwargs):
        self.xgb = XGBRegressor(random_state=random_state, n_jobs=1, **xgb_kwargs)
        self.ridge = Ridge(alpha=1.0)
        self.weight_xgb = weight_xgb

    def fit(self, X, y):
        self.xgb.fit(X, y)
        self.ridge.fit(X, y)
        return self

    def predict(self, X):
        return self.weight_xgb * self.xgb.predict(X) + (1 - self.weight_xgb) * self.ridge.predict(X)


X0_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8)


def run(name, model_kwargs=None, regressor_factory=None):
    model_kwargs = model_kwargs or {}
    if regressor_factory is None:
        params = model_kwargs.pop("_xgb_params", X0_PARAMS)
        regressor_factory = lambda: XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
    model = feb.make_v9_model(
        k=model_kwargs.pop("k", K), n_clusters=N_CLUSTERS, adaptive_k=True, use_euclid=True,
        use_volatility=False, make_regressor=regressor_factory, random_state=RANDOM_STATE,
        **model_kwargs,
    )
    print(f"=== {name} ===")
    t0 = time.time()
    scores = feb.evaluate_repeated_splits(
        X_train, y_train, donor_cols, model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    print(f"  total: {time.time()-t0:.1f}s\n")
    return scores


CONFIGS = {
    "X0_baseline": dict(
        label="X0 baseline (n_estimators=200, max_depth=3, lr=0.1, subsample=0.8, colsample=0.8)"),
    "T1_mae_objective": dict(
        label="T1: +objective=reg:absoluteerror (aligné sur la metrique du challenge)",
        model_kwargs={"_xgb_params": dict(X0_PARAMS, objective="reg:absoluteerror")}),
    "T2_regularized": dict(
        label="T2: hyperparams plus regularises (depth=2, min_child_weight=5, reg_lambda=5, sub/col=0.7)",
        model_kwargs={"_xgb_params": dict(max_depth=2, min_child_weight=5, reg_lambda=5.0,
                                           subsample=0.7, colsample_bytree=0.7, n_estimators=200, learning_rate=0.1)}),
    "T3_capacity": dict(
        label="T3: plus de capacite (depth=5, n_estimators=400, lr=0.05)",
        model_kwargs={"_xgb_params": dict(max_depth=5, n_estimators=400, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8)}),
    "T4_cluster_mean_feat": dict(
        label="T4: +feature profil moyen du cluster",
        model_kwargs={"use_cluster_mean_feat": True}),
    "T5_lag_feats": dict(
        label="T5: +lags directs (rejetes pour Ridge en V8, re-testes pour arbres)",
        model_kwargs={"use_lag_feats": True}),
    "T6_early_stop": dict(
        label="T6: early stopping (validation interne chronologique, n_estimators=1000 max)",
        regressor_factory=lambda: XGBEarlyStopRegressor(
            val_frac=0.15, max_depth=4, learning_rate=0.05, n_estimators=1000,
            subsample=0.8, colsample_bytree=0.8)),
    "T7_k15": dict(label="T7: k=15 (moins de voisins)", model_kwargs={"k": 15}),
    "T8_k35": dict(label="T8: k=35 (plus de voisins)", model_kwargs={"k": 35}),
    "T9_ensemble_ridge": dict(
        label="T9: ensemble XGBoost(70%) + Ridge(30%)",
        regressor_factory=lambda: XGBRidgeEnsemble(weight_xgb=0.7, **X0_PARAMS)),
}

RESULTS_PATH = "xgboost_bench_results.json"


def load_results():
    import json, os
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {}


def save_result(name, scores):
    import json
    results = load_results()
    results[name] = scores.tolist()
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import sys
    # un seul config par process (mémoire système limitée sur cette machine --
    # chaque process reparti à zéro évite d'accumuler XGBoost/KMeans/pandas
    # d'un config a l'autre, et une config qui plante ne fait perdre que sa
    # propre config, pas tout le run -- cf. xgboost_bench_runner.sh)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    names = [only] if only else list(CONFIGS.keys())

    print("Chargement des données (X_train, y_train seulement -- X_test inutile ici)...")
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    for name in names:
        cfg = CONFIGS[name]
        scores = run(cfg["label"], cfg.get("model_kwargs"), cfg.get("regressor_factory"))
        save_result(name, scores)
        print(f"[{name}] sauvegardé dans {RESULTS_PATH}")
