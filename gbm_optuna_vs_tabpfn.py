"""LightGBM enrichi (Optuna + CV temporelle + early stopping, MAE) branché sur
EXACTEMENT le même pipeline de features que le meilleur modèle validé du
dépôt, TabPFN V14 (cf. tabpfn_v14_ablation.py / tabpfn_v14_confirm.py), puis
évalué sur les 5 premiers repeats (mêmes graines 42..46, mêmes colonnes) du
protocole 10x10 utilisé pour confirmer V14 -- comparaison appariée directe
avec tabpfn_v14_confirm_results.json (63.24 +/- 30.46 W sur les 10 repeats).

Le pipeline V14 (`tabpfn_v14_ablation.predict_column_v14`) construit, PAR
COLONNE trouée, une matrice de features (voisins mis à l'échelle + features
calendaires/périodiques/saisonnalité) et un régresseur interchangeable
(`opts["regressor_cls"]`/`opts["regressor_kwargs"]`) fitté sur les points
observés de cette même colonne puis prédit sur ses points manquants -- il
suffit donc de brancher un régresseur GBM à la place de TabPFN pour une
comparaison tête-à-tête rigoureuse sur les mêmes features et le même blend
alpha (interpolation/modèle).

Etapes (cf. `python gbm_optuna_vs_tabpfn.py <stage>`) :
  prep    -- construit et met en cache (X_obs, y_obs) pour N_HPO_COLUMNS
             colonnes de recherche, STRICTEMENT DISJOINTES des colonnes
             pouvant apparaître en validation dans les 5 repeats de
             comparaison (union des val_cols_full des seeds 42..46).
  hpo     -- recherche d'hyperparamètres Optuna (TPE), MAE, CV 3-fold par
             colonne, early stopping interne à chaque fit (ESCatBoost).
             Etude persistée en SQLite -> résumable (kill/restart safe).
  confirm -- rejoue tabpfn_v14_ablation.run_config avec le CatBoost tuné sur
             n_repeats=5, n_val_sub=10 (seeds 42..46, IDENTIQUES à
             tabpfn_v14_confirm.py) et compare au tableau TabPFN V14.
  all     -- prep + hpo + confirm.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split

warnings.filterwarnings("ignore")

from evaluate import load_data
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14

RANDOM_STATE = 42
N_REPEATS_COMPARE = 5          # 5 random states demandés, mêmes graines que TabPFN V14
N_VAL_SUB = 10                 # même sous-échantillon de colonnes que le protocole 10x10
N_HPO_COLUMNS = 10
N_HPO_TRIALS = 50
N_CV_FOLDS = 3
HPO_COLUMN_SEED = 12345

STUDY_DB = "sqlite:///gbm_optuna_study.db"
STUDY_NAME = "lightgbm_all_features_distribution50_mae"
DATASET_CACHE = "gbm_hpo_datasets.npz"
BEST_PARAMS_PATH = "gbm_best_params.json"
RESULTS_PATH = "gbm_optuna_vs_tabpfn_results.json"
TABPFN_V14_CONFIRM_PATH = "tabpfn_v14_confirm_results.json"

# Meilleure configuration V14 (cf. tabpfn_v14_confirm.py) : mêmes features et
# même blend alpha pour les deux régresseurs -- seul le régresseur change.
import tabpfn_v15_features as v15

BEST_V14_OPTS = dict(
    volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
    extra_lags=(24,), neighbor_rolling=True, neighbor_consensus=True,
    weekday_seasonality=True, daytype_periodic=True, daytype_flag_feat=True,
    local_trend_feat=True, neighbor_mad_feat=True, neighbor_interval_feat=True,
    peak_offpeak_feat=True, distribution50=True, v15_groups=v15.FEATURE_GROUPS,
    iterative_passes=2,
)


# ---------------------------------------------------------------------------
# Régresseur CatBoost + arrêt anticipé sur MAE
# ---------------------------------------------------------------------------

class ESLightGBM:
    """LightGBM MAE avec validation chronologique et arrêt anticipé : une petite
    validation interne (`es_val_frac` des points passés à `fit`) pilote
    l'early stopping, sans jamais toucher aux points réellement manquants
    (miss_idx) ni aux colonnes de comparaison -- l'API reste fit(X, y) /
    predict(X), donc branchable telle quelle sur
    `opts["regressor_cls"]`/`opts["regressor_kwargs"]` du pipeline V14.
    Repli déterministe sur la constante si la cible est constante (mêmes
    paliers nocturnes que catboost_gpu_finetune.SafeCatBoostGPU)."""

    def __init__(self, num_leaves=31, max_depth=-1, learning_rate=0.05,
                 min_child_samples=20, min_split_gain=0.0,
                 feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                 lambda_l1=0.0, lambda_l2=0.0, n_estimators=1200, es_rounds=50,
                 es_val_frac=0.15, min_for_es=40, thread_count=-1,
                 random_state=RANDOM_STATE):
        self.params = dict(num_leaves=num_leaves, max_depth=max_depth,
            learning_rate=learning_rate, min_child_samples=min_child_samples,
            min_split_gain=min_split_gain, colsample_bytree=feature_fraction,
            subsample=bagging_fraction, subsample_freq=bagging_freq,
            reg_alpha=lambda_l1, reg_lambda=lambda_l2, n_estimators=n_estimators)
        self.es_rounds = es_rounds
        self.es_val_frac = es_val_frac
        self.min_for_es = min_for_es
        self.thread_count = thread_count
        self.random_state = random_state

    def fit(self, X, y):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0., posinf=0., neginf=0.)
        y = np.asarray(y, dtype=np.float64)
        if np.all(y == y[0]):
            self._constant = float(y[0])
            self.reg = None
            return self
        self._constant = None
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
        n = len(y)
        common = dict(objective="mae", random_state=self.random_state,
                      n_jobs=self.thread_count, verbosity=-1, **self.params)
        if n >= self.min_for_es:
            n_val = max(5, int(round(n * self.es_val_frac)))
            split = n - n_val
            self.reg = LGBMRegressor(**common)
            self.reg.fit(X[:split], y[:split], eval_set=[(X[split:], y[split:])],
                eval_metric="mae", callbacks=[early_stopping(self.es_rounds, verbose=False),
                                                log_evaluation(period=0)])
        else:
            # trop peu de points pour une validation interne fiable : pas
            # d'early stopping, budget d'arbres réduit pour limiter le
            # surapprentissage.
            common["n_estimators"] = min(common["n_estimators"], 150)
            self.reg = LGBMRegressor(**common)
            self.reg.fit(X, y)
        return self

    def predict(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0., posinf=0., neginf=0.)
        if self.reg is None:
            return np.full(X.shape[0], self._constant)
        return self.reg.predict(X)


# ---------------------------------------------------------------------------
# Colonnes de recherche HPO, disjointes des colonnes de comparaison
# ---------------------------------------------------------------------------

def reserved_comparison_columns(holed_cols, n_repeats=N_REPEATS_COMPARE):
    """Union des val_cols_full (10% complets, pas seulement le sous-échantillon
    n_val_sub) pour les seeds RANDOM_STATE..RANDOM_STATE+n_repeats-1 -- EXACTEMENT
    les mêmes appels que tabpfn_v14_ablation.run_config. Toute colonne ici est
    interdite de recherche d'hyperparamètres (fuite potentielle vers la
    comparaison finale)."""
    reserved = set()
    for i in range(n_repeats):
        seed_i = RANDOM_STATE + i
        _, val_cols_full = train_test_split(holed_cols, test_size=.1, random_state=seed_i)
        reserved.update(val_cols_full)
    return reserved


def pick_hpo_columns(y, n_repeats=N_REPEATS_COMPARE, n_columns=N_HPO_COLUMNS):
    holed_cols = list(y.columns)
    reserved = reserved_comparison_columns(holed_cols, n_repeats)
    candidates = [c for c in holed_cols if c not in reserved]
    rng = np.random.RandomState(HPO_COLUMN_SEED)
    chosen = rng.choice(candidates, size=min(n_columns, len(candidates)), replace=False)
    return list(chosen), reserved


# ---------------------------------------------------------------------------
# Construction des datasets HPO (mêmes features que le pipeline V14)
# ---------------------------------------------------------------------------

def build_pool(X, y, donor_cols, exclude_cols):
    holed_cols = list(y.columns)
    dev_cols_pool = [c for c in holed_cols if c not in exclude_cols]
    donor_df = X[donor_cols]
    dev_true_df = y[dev_cols_pool]
    pool = pd.concat([donor_df, dev_true_df], axis=1)
    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=15, random_state=RANDOM_STATE)
    dev_filled = dev_true_df.ffill().bfill()
    Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
    pool_labels = pd.concat([donor_labels, dev_labels])
    pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
    cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(X.index, ("hour", "dow", "dom", "year"))
    return pool_vals, km, cluster_col_idx, tfeat


def build_hpo_datasets(X, y, donor_cols, hpo_cols):
    pool_vals, km, cluster_col_idx, tfeat = build_pool(X, y, donor_cols, exclude_cols=set(hpo_cols))
    datasets = []
    for c in hpo_cols:
        s = X[c]
        built = v14.build_v14_features(c, s, pool_vals, km, cluster_col_idx, tfeat, X.index, BEST_V14_OPTS)
        if built is None:
            continue
        obs_idx, miss_idx, target_obs, interp_full, knn_pred, scaled, extra, missing, lags = built
        X_obs = np.nan_to_num(
            np.hstack([scaled[obs_idx, :], extra[obs_idx, :]]).astype(np.float64),
            nan=0., posinf=0., neginf=0.)
        y_obs = target_obs.astype(np.float64)
        datasets.append((c, X_obs, y_obs))
    return datasets


def save_datasets(datasets, path=DATASET_CACHE):
    payload = {}
    names = []
    for c, X_obs, y_obs in datasets:
        names.append(c)
        payload[f"X__{c}"] = X_obs
        payload[f"y__{c}"] = y_obs
    payload["__names__"] = np.array(names, dtype=object)
    np.savez_compressed(path, **payload)


def load_datasets(path=DATASET_CACHE):
    data = np.load(path, allow_pickle=True)
    names = data["__names__"]
    return [(str(c), data[f"X__{c}"], data[f"y__{c}"]) for c in names]


# ---------------------------------------------------------------------------
# Recherche Optuna : MAE, CV 3-fold par colonne, early stopping interne
# ---------------------------------------------------------------------------

def sample_lightgbm_params(trial):
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 8, 96),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 80),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
        feature_fraction=trial.suggest_float("feature_fraction", 0.55, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.55, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 1, 7),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-6, 20.0, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-6, 20.0, log=True),
        n_estimators=trial.suggest_int("n_estimators", 300, 2000),
    )


def cv_objective(trial, datasets, n_folds=N_CV_FOLDS, cv_seed=RANDOM_STATE):
    params = sample_lightgbm_params(trial)
    abs_errs = []
    for _, X_obs, y_obs in datasets:
        n = len(y_obs)
        if n < 30:
            continue
        kf = TimeSeriesSplit(n_splits=n_folds)
        for tr_idx, te_idx in kf.split(X_obs):
            reg = ESLightGBM(es_rounds=50, random_state=cv_seed, **params)
            reg.fit(X_obs[tr_idx], y_obs[tr_idx])
            pred = reg.predict(X_obs[te_idx])
            abs_errs.append(np.abs(pred - y_obs[te_idx]))
    if not abs_errs:
        return 1e6
    return float(np.concatenate(abs_errs).mean())


def run_hpo(datasets, n_trials=N_HPO_TRIALS):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STUDY_DB, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE), load_if_exists=True,
    )
    done = len(study.trials)
    remaining = max(0, n_trials - done)
    print(f"Optuna: {done} essais déjà enregistrés, {remaining} restants "
          f"(objectif {n_trials}).", flush=True)
    if remaining > 0:
        t0 = time.time()
        study.optimize(lambda t: cv_objective(t, datasets), n_trials=remaining,
                        show_progress_bar=False)
        print(f"  -> {remaining} essais en {time.time() - t0:.0f}s", flush=True)
    print(f"Meilleur essai : MAE(CV)={study.best_value:.2f} W, params={study.best_params}",
          flush=True)
    return study


# ---------------------------------------------------------------------------
# Confirmation : mêmes 5 repeats/graines que tabpfn_v14_confirm.py
# ---------------------------------------------------------------------------

def run_confirm(best_params, X, y, donor_cols):
    regressor_kwargs = dict(best_params)
    regressor_kwargs["es_rounds"] = 50
    regressor_kwargs["thread_count"] = 1  # cohérent avec le parallélisme par colonne (20 threads)
    opts = dict(BEST_V14_OPTS, regressor_cls=ESLightGBM, regressor_kwargs=regressor_kwargs)
    scores = v14.run_config("lightgbm_enriched_tuned", opts, X, y, donor_cols,
                             n_repeats=N_REPEATS_COMPARE, n_val_sub=N_VAL_SUB)
    return scores


def compare_to_tabpfn(gbm_scores):
    tabpfn_all = json.load(open(TABPFN_V14_CONFIRM_PATH))["scores"]
    tabpfn_5 = tabpfn_all[:N_REPEATS_COMPARE]
    print("\n=== Comparaison appariée (mêmes graines 42..46) ===")
    print(f"{'repeat':>6} | {'LightGBM (Optuna)':>18} | {'TabPFN V14':>11} | {'Δ (GBM-TabPFN)':>15}")
    deltas = []
    for i, (g, t) in enumerate(zip(gbm_scores, tabpfn_5), start=1):
        d = g - t
        deltas.append(d)
        print(f"{i:>6} | {g:18.2f} | {t:11.2f} | {d:+15.2f}")
    gbm_arr, tabpfn_arr, deltas = np.array(gbm_scores), np.array(tabpfn_5), np.array(deltas)
    print(f"{'mean':>6} | {gbm_arr.mean():18.2f} | {tabpfn_arr.mean():11.2f} | {deltas.mean():+15.2f}")
    print(f"{'std':>6} | {gbm_arr.std():18.2f} | {tabpfn_arr.std():11.2f} |")
    wins = int((deltas < 0).sum())
    print(f"\nLightGBM gagne {wins}/{N_REPEATS_COMPARE} repeats.")
    return {
        "gbm_scores": gbm_scores.tolist() if hasattr(gbm_scores, "tolist") else list(gbm_scores),
        "gbm_mean": float(gbm_arr.mean()), "gbm_std": float(gbm_arr.std()),
        "tabpfn_v14_scores_first5": tabpfn_5,
        "tabpfn_v14_mean_first5": float(tabpfn_arr.mean()),
        "tabpfn_v14_std_first5": float(tabpfn_arr.std()),
        "tabpfn_v14_mean_full10": float(np.mean(tabpfn_all)),
        "tabpfn_v14_std_full10": float(np.std(tabpfn_all)),
        "wins_gbm": wins, "n_repeats": N_REPEATS_COMPARE,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donor_cols = [c for c in X.columns if not str(c).startswith("holed_")]

    if stage in ("prep", "all"):
        print("=== prep : construction des datasets HPO ===", flush=True)
        hpo_cols, reserved = pick_hpo_columns(y)
        print(f"{len(reserved)} colonnes réservées à la comparaison (5 repeats), "
              f"{len(hpo_cols)} colonnes HPO choisies parmi les {len(y.columns) - len(reserved)} restantes.",
              flush=True)
        t0 = time.time()
        datasets = build_hpo_datasets(X, y, donor_cols, hpo_cols)
        print(f"  -> {len(datasets)} datasets construits en {time.time() - t0:.0f}s "
              f"(tailles: {[d[1].shape for d in datasets]})", flush=True)
        save_datasets(datasets)
        print(f"  -> cache écrit dans {DATASET_CACHE}", flush=True)

    if stage in ("hpo", "all"):
        print("=== hpo : recherche Optuna (MAE, CV 3-fold, early stopping) ===", flush=True)
        datasets = load_datasets()
        datasets = [(c, Xo, yo) for c, Xo, yo in datasets]
        study = run_hpo(datasets, n_trials=N_HPO_TRIALS)
        json.dump({"best_params": study.best_params, "best_cv_mae": study.best_value,
                   "n_trials": len(study.trials)},
                  open(BEST_PARAMS_PATH, "w"), indent=2)
        print(f"  -> meilleurs hyperparamètres écrits dans {BEST_PARAMS_PATH}", flush=True)

    if stage in ("confirm", "all"):
        print("=== confirm : 5 repeats 90/10 (seeds 42..46), LightGBM enrichi ===", flush=True)
        best = json.load(open(BEST_PARAMS_PATH))["best_params"]
        t0 = time.time()
        scores = run_confirm(best, X, y, donor_cols)
        dt = time.time() - t0
        result = compare_to_tabpfn(scores)
        result["best_params"] = best
        result["seconds"] = dt
        json.dump(result, open(RESULTS_PATH, "w"), indent=2)
        print(f"\nRésultats écrits dans {RESULTS_PATH} ({dt:.0f}s)", flush=True)

    print("\nTerminé.", flush=True)
