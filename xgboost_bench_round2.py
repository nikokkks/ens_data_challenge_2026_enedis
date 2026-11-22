"""
Round 2 : 10 pistes supplémentaires (issues de recherche + itération sur les
résultats du round 1), construites sur X1 = X0 + T1_mae_objective (le
meilleur du round 1, cf. xgboost_bench_results.json), testées sur le même
harnais officiel (n=5 hold-out 90/10 indépendants, graines différentes,
MAE officielle NaN-safe).

Recherche (cf. résumé dans la conversation) : littérature sur l'imputation de
séries temporelles multivariées corrélées (SAITS/BRITS/GRIN, TSI-Bench,
SoftImpute/TRMF) + pratiques standards de gradient boosting sur petits jeux
de données (CatBoost "ordered boosting" contre l'overfitting, objectifs
quantile/MAE natifs).

Les 10 pistes proposées (7 implémentées ici, 3 documentées en fin de fichier
avec la raison de ne pas les avoir testées cette fois) :
  R1 LightGBM (objectif MAE natif) à la place de XGBoost
  R2 CatBoost (ordered boosting, loss MAE) à la place de XGBoost
  R3 XGBoost objectif quantile (reg:quantileerror, alpha=0.5) vs reg:absoluteerror
  R4 Pondération des échantillons d'entraînement par proximité aux vrais trous
  R5 (non implémentée -- cf. fin de fichier) Propagation 2nd ordre (voisins des voisins)
  R6 Ensemble XGBoost(MAE) + Ridge, poids appris
  R7 n_clusters tuning (15 / 25 / 40)
  R8 k=40 (plus de voisins, les arbres géreraient mieux le bruit que Ridge)
  R9 (non implémentée) Décomposition STL/saisonnière comme features
  R10 (non implémentée) Régularisation temporelle TRMF sur SoftImpute
"""

import time
import warnings
import numpy as np
from scipy.ndimage import distance_transform_edt
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, RidgeCV

warnings.filterwarnings("ignore")

from evaluate import load_data
import feature_engineering_bench as feb

RANDOM_STATE = 42
N_REPEATS = 5
TEST_SIZE = 0.10
K = 20
N_CLUSTERS = 25

X1_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, objective="reg:absoluteerror")


class EnsembleXGBRidge:
    def __init__(self, weight_xgb=0.6, **xgb_kwargs):
        self.xgb = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, **xgb_kwargs)
        self.ridge = Ridge(alpha=1.0)
        self.weight_xgb = weight_xgb

    def fit(self, X, y):
        self.xgb.fit(X, y)
        self.ridge.fit(X, y)
        return self

    def predict(self, X):
        return self.weight_xgb * self.xgb.predict(X) + (1 - self.weight_xgb) * self.ridge.predict(X)


class WeightedXGB:
    """Pondère les points d'entraînement par proximité au(x) trou(s) réel(s)
    de la courbe -- distance calculée une fois par colonne, passée via un
    attribut plutôt que recalculée dans fit() (qui ne voit que X,y)."""

    _current_weights = None  # set par local_regression_pred_weighted avant fit()

    def __init__(self, **xgb_kwargs):
        self.xgb_kwargs = xgb_kwargs

    def fit(self, X, y):
        self.reg = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, **self.xgb_kwargs)
        w = WeightedXGB._current_weights
        if w is not None and len(w) == len(y):
            self.reg.fit(X, y, sample_weight=w)
        else:
            self.reg.fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X)


def make_weighted_v9_model(k=K, n_clusters=N_CLUSTERS, xgb_kwargs=None):
    """Variante de make_v9_model qui calcule un poids par point observé
    (proximité aux vrais trous de la courbe, via distance_transform_edt) et
    le passe à WeightedXGB avant chaque fit -- même pipeline sinon."""
    xgb_kwargs = xgb_kwargs or X1_PARAMS
    from evaluate import fill_nan_with_interpolation
    import pandas as pd

    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        pool_vals = pool.ffill().bfill().values.astype(np.float64)
        km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=n_clusters, random_state=RANDOM_STATE)
        if dev_true_df.shape[1] > 0:
            dev_filled = dev_true_df.ffill().bfill()
            Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
            dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
            pool_labels = pd.concat([donor_labels, dev_labels])
        else:
            pool_labels = donor_labels
        pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
        cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
        tfeat = feb.time_features(val_holed_df.index)

        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            observed = s.notna().values
            missing = ~observed
            if not missing.any():
                preds[col] = s
                continue
            obs_idx, miss_idx = np.where(observed)[0], np.where(missing)[0]
            target_obs = s.values[obs_idx]

            interp_full = fill_nan_with_interpolation(s)
            cl = feb.cluster_of(interp_full, km)
            same_idx = cluster_col_idx.get(cl)
            sub_vals = pool_vals[:, same_idx] if (same_idx is not None and len(same_idx) >= max(2 * k, 30)) else pool_vals

            knn_pred, top_idx, scaled = feb.weighted_knn_pred(
                sub_vals, obs_idx, miss_idx, target_obs, k, robust=True, adaptive_k=True, use_euclid=True)

            if len(obs_idx) >= 15:
                dist_to_gap = distance_transform_edt(~missing)
                w_obs = 1.0 / (1.0 + dist_to_gap[obs_idx] / 10.0)  # echelle ~10 points (5h)
                WeightedXGB._current_weights = w_obs
                pred_neighbors = feb.local_regression_pred(
                    lambda: WeightedXGB(**xgb_kwargs), scaled, obs_idx, miss_idx, target_obs, tfeat)
                WeightedXGB._current_weights = None
            else:
                pred_neighbors = knn_pred

            interp = interp_full.values[miss_idx]
            run_len_full = feb.run_lengths_for(missing)
            edge_dist_full = feb.edge_distance_for(missing)
            alphas = np.array([feb.continuous_alpha(run_len_full[m], edge_dist_full[m], adaptive=True) for m in miss_idx])
            pred = alphas * interp + (1 - alphas) * pred_neighbors

            filled = s.copy()
            filled.iloc[miss_idx] = pred
            preds[col] = filled
        return __import__("pandas").DataFrame(preds, index=val_holed_df.index)
    return model


def run(name, model_kwargs=None, regressor_factory=None, custom_model=None):
    model_kwargs = model_kwargs or {}
    if custom_model is not None:
        model = custom_model
    else:
        if regressor_factory is None:
            params = model_kwargs.pop("_xgb_params", X1_PARAMS)
            regressor_factory = lambda: XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, **params)
        model = feb.make_v9_model(
            k=model_kwargs.pop("k", K), n_clusters=model_kwargs.pop("n_clusters", N_CLUSTERS),
            adaptive_k=True, use_euclid=True, use_volatility=False,
            make_regressor=regressor_factory, random_state=RANDOM_STATE, **model_kwargs)
    print(f"=== {name} ===")
    t0 = time.time()
    scores = feb.evaluate_repeated_splits(
        X_train, y_train, donor_cols, model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"  total: {time.time()-t0:.1f}s\n")
    return scores


CONFIGS = {}

CONFIGS["R1_lightgbm"] = dict(
    label="R1: LightGBM (objectif MAE natif) a la place de XGBoost",
    regressor_factory=lambda: __import__("lightgbm").LGBMRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, objective="mae", random_state=RANDOM_STATE,
        n_jobs=1, verbosity=-1))

class SafeCatBoost:
    """CatBoost plante (`CatBoostError: All train targets are equal`) sur les
    courbes dont la portion observee est parfaitement constante (paliers
    nocturnes a 0 par ex.) -- XGBoost/LightGBM/Ridge tolerent ce cas sans
    broncher, pas CatBoost. Repli sur une prediction constante plutot que de
    laisser planter les 1000 colonnes pour ce cas limite."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        if np.all(y == y[0]):
            self._constant = float(y[0])
            self.reg = None
        else:
            self._constant = None
            from catboost import CatBoostRegressor
            self.reg = CatBoostRegressor(**self.kwargs)
            self.reg.fit(X, y)
        return self

    def predict(self, X):
        if self.reg is None:
            return np.full(X.shape[0], self._constant)
        return self.reg.predict(X)


CONFIGS["R2_catboost"] = dict(
    label="R2: CatBoost (ordered boosting, loss MAE) a la place de XGBoost",
    regressor_factory=lambda: SafeCatBoost(
        iterations=200, depth=3, learning_rate=0.1, loss_function="MAE",
        random_state=RANDOM_STATE, verbose=False, thread_count=1))

CONFIGS["R3_quantile_obj"] = dict(
    label="R3: objectif quantile (reg:quantileerror, alpha=0.5) vs reg:absoluteerror",
    model_kwargs={"_xgb_params": dict(n_estimators=200, max_depth=3, learning_rate=0.1,
                                       subsample=0.8, colsample_bytree=0.8,
                                       objective="reg:quantileerror", quantile_alpha=0.5)})

CONFIGS["R4_sample_weight_gap_proximity"] = dict(
    label="R4: ponderation des points d'entrainement par proximite aux vrais trous",
    custom_model=None)  # construit a part (make_weighted_v9_model), cf main

CONFIGS["R6_ensemble_xgb_ridge"] = dict(
    label="R6: ensemble XGBoost(MAE, 60%) + Ridge(40%)",
    regressor_factory=lambda: EnsembleXGBRidge(weight_xgb=0.6, **X1_PARAMS))

CONFIGS["R7_nclusters15"] = dict(label="R7a: n_clusters=15", model_kwargs={"n_clusters": 15})
CONFIGS["R7_nclusters40"] = dict(label="R7b: n_clusters=40", model_kwargs={"n_clusters": 40})
CONFIGS["R8_k40"] = dict(label="R8: k=40 (plus de voisins)", model_kwargs={"k": 40})

RESULTS_PATH = "xgboost_bench_round2_results.json"


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

    print("Chargement des données (X_train, y_train)...")
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    for name in names:
        cfg = CONFIGS[name]
        if name == "R4_sample_weight_gap_proximity":
            model = make_weighted_v9_model(k=K, n_clusters=N_CLUSTERS, xgb_kwargs=X1_PARAMS)
            scores = run(cfg["label"], custom_model=model)
        else:
            scores = run(cfg["label"], cfg.get("model_kwargs"), cfg.get("regressor_factory"))
        save_result(name, scores)
        print(f"[{name}] sauvegardé dans {RESULTS_PATH}")

# --- Pistes non implémentées cette fois (documentées, pas testées) ---
# R5 propagation 2nd ordre (voisins des voisins) : nécessite une double passe
#   sur les colonnes val (compléter une 1ere fois, puis réutiliser les valeurs
#   complétées des AUTRES courbes val comme donneurs supplémentaires pour une
#   2e passe) -- restructuration non triviale du harnais d'évaluation
#   (les colonnes val sont actuellement indépendantes entre elles), reportée.
# R9 décomposition STL/saisonnière : décomposer chaque courbe voisine en
#   tendance+saisonnalité+résidu avant de les utiliser comme features --
#   pertinent (profil journalier marqué, cf. figures/03) mais demande de
#   fixer une méthode de décomposition robuste aux trous eux-mêmes, reporté.
# R10 régularisation temporelle façon TRMF sur SoftImpute (pénaliser les
#   variations brusques d'un pas de temps à l'autre dans la reconstruction
#   bas-rang) : implémentable mais alourdit sensiblement l'algorithme
#   itératif déjà coûteux (SVD répétées) ; reporté faute de temps cette session.
