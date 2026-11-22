"""Ablation des régresseurs locaux sur la meilleure configuration rapide."""
import json
import sys
import warnings

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from evaluate import load_data
import feature_engineering_bench as feb
from xgboost_bench_round2 import SafeCatBoost

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
RESULTS_PATH = "advanced_regressors_results.json"


def lgbm():
    return LGBMRegressor(objective="mae", n_estimators=200, max_depth=3,
        learning_rate=.1, subsample=.8, colsample_bytree=.8,
        random_state=RANDOM_STATE, n_jobs=1, verbosity=-1)


def catboost():
    return SafeCatBoost(iterations=250, depth=4, learning_rate=.06,
        loss_function="MAE", random_state=RANDOM_STATE, verbose=False,
        thread_count=1, allow_writing_files=False)


def extratrees():
    # ExtraTrees ne gère pas tous les NaN selon les versions/options : une
    # imputation médiane locale rend le test reproductible.
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
        ExtraTreesRegressor(n_estimators=300, min_samples_leaf=4,
            max_features=.8, criterion="squared_error",
            random_state=RANDOM_STATE, n_jobs=1))


def histgb():
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=200,
        learning_rate=.07, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, random_state=RANDOM_STATE)


class MedianTreeEnsemble:
    def __init__(self):
        self.models = [lgbm(), catboost(), extratrees()]
    def fit(self, X, y):
        for model in self.models: model.fit(X, y)
        return self
    def predict(self, X):
        return np.median(np.vstack([m.predict(X) for m in self.models]), axis=0)


FACTORIES = {
    "R0_lightgbm_reference": lgbm,
    "R1_catboost_mae": catboost,
    "R2_extratrees_fast": extratrees,
    "R3_histgradient_absolute": histgb,
    "R4_median_lgbm_cat_extra": MedianTreeEnsemble,
}

BEST_FEATURES = dict(
    use_seasonality_feat=True, use_full_calendar_feat=True,
    calendar_groups=("hour", "dow", "dom", "year"),
    use_periodic_copy_feat=True, use_periodic_consensus_feat=True,
    normalize_periodic_feat=True, include_periodic_availability=True,
    periodic_alpha_strength=.5,
)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]
    try: results = json.load(open(RESULTS_PATH))
    except FileNotFoundError: results = {}
    for name in ([only] if only else FACTORIES):
        print(f"=== {name} ===", flush=True)
        model = feb.make_v10_model(k=40, n_clusters=15, adaptive_k=True,
            use_euclid=True, make_regressor=FACTORIES[name],
            random_state=RANDOM_STATE, **BEST_FEATURES)
        scores = feb.evaluate_repeated_splits(X, y, donors, model,
            n_repeats=10, test_size=.1, random_state=RANDOM_STATE,
            n_val_sub=10, save_path=f"advanced_{name}_progress.json")
        results[name] = scores.tolist()
        with open(RESULTS_PATH, "w") as f: json.dump(results, f, indent=2)
