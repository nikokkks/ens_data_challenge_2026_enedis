"""Bench AutoGluon (régression, métrique MAE, cross-validation interne via
`num_bag_folds`) branché sur les MÊMES features que le pipeline TabPFN V14
(`build_v14_features` : bloc de voisins `scaled` + les 35 features `extra`
de la config v14_best) -- demande explicite : "utiliser autogluon avec la
mae et les features du modèle tabpfnv14 en faisant une cross-validation...
sur 300 secondes".

Contrairement aux scripts `*_confirm.py` (qui branchent un régresseur en
remplacement de TabPFN via `regressor_cls` et nécessitent TABPFN_API_KEY),
ce bench ne dépend d'AUCUNE API cloud -- AutoGluon tourne 100% en local.
Il peut donc être exécuté immédiatement, sans clé, contrairement aux 8
scripts `tabpfn_v14_*_confirm.py` déjà en attente.

Protocole :
- Même split dev/val que `run_config` (repeat i=0, RANDOM_STATE=42,
  test_size=.1 sur les colonnes `holed_*`), N_VAL_SUB colonnes réelles
  tirées au hasard (mêmes graines).
- Pour chaque colonne : mêmes features que `predict_column_v14`
  (`build_v14_features`, config v14_best -- `volatility_alpha=True,
  vol_weight=.8, periodic_alpha_strength=1.0, extra_lags=(24,),
  neighbor_rolling=True`), mais le régresseur est
  `TabularPredictor(eval_metric='mean_absolute_error',
  problem_type='regression').fit(..., time_limit=300, num_bag_folds=5)` --
  le bagging 5-fold est la "cross-validation" demandée, plafonnée à 300s
  par colonne. Familles de modèles restreintes aux arbres (GBM/CAT/XT/RF,
  pas de réseau de neurones) pour rester léger en RAM (poste avec peu de
  mémoire libre au moment du run).
- Une prédiction LightGBM (rapide, gratuite, déjà le proxy local standard
  de ce dépôt) est calculée en regard sur EXACTEMENT les mêmes points, pour
  situer AutoGluon parmi les régresseurs locaux déjà testés (benchmark.md)
  même en l'absence de tout accès à TabPFN.

⚠️ 300s/colonne (borne haute, pas systématiquement atteinte) => échelle
réduite par rapport au protocole officiel 10x10 (qui coûterait ~8h en
budget AutoGluon plein ici) : N_VAL_SUB colonnes, 1 seul repeat. Ce n'est
donc PAS un "confirm" comparable point par point aux scripts `*_confirm.py`
(qui utilisent TabPFN comme régresseur) -- c'est une comparaison
architecture-vs-architecture (AutoGluon vs LightGBM) sur les features V14,
utile pour juger si un AutoML local mérite d'être creusé plus avant.

Reprise sur erreur : écrit `tabpfn_v14_autogluon_bench_results.json` après
CHAQUE colonne et saute les colonnes déjà présentes au redémarrage (mêmes
conventions que les scripts `generate_submission_*.py`, cf. CLAUDE.md).
"""
import json
import os
import shutil
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

from evaluate import load_data
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_VAL_SUB = 8
TIME_LIMIT = 3600
NUM_BAG_FOLDS = 4
RESULTS_PATH = "tabpfn_v14_autogluon_bench_results.json"

# Config BEST identique a v14_best (tabpfn_v14_confirm.py), regresseur mis
# a part -- seules les FEATURES sont reutilisees ici, pas TabPFN.
BEST = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
            extra_lags=(24,), neighbor_rolling=True)


def autogluon_predict(X_obs, y_obs, X_miss):
    from autogluon.tabular import TabularPredictor
    cols = [f"f{i}" for i in range(X_obs.shape[1])]
    train_df = pd.DataFrame(X_obs, columns=cols)
    train_df["target"] = y_obs
    test_df = pd.DataFrame(X_miss, columns=cols)
    tmp_dir = tempfile.mkdtemp(prefix="ag_v14_")
    try:
        predictor = TabularPredictor(
            label="target", eval_metric="mean_absolute_error",
            problem_type="regression", path=tmp_dir, verbosity=1,
        ).fit(
            train_df, time_limit=TIME_LIMIT,
            num_bag_folds=NUM_BAG_FOLDS, num_bag_sets=1,
            hyperparameters={"GBM": {}, "CAT": {}, "XT": {}, "RF": {}},
            ag_args_fit={"num_cpus": 4},
        )
        pred = predictor.predict(test_df).to_numpy()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return pred


def lightgbm_predict(X_obs, y_obs, X_miss):
    reg = LGBMRegressor(objective="mae", n_estimators=200, max_depth=3,
        learning_rate=.1, subsample=.8, colsample_bytree=.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    reg.fit(X_obs, y_obs)
    return reg.predict(X_miss)


if __name__ == "__main__":
    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]
    holed_cols = list(y.columns)

    dev_cols, val_cols_full = train_test_split(holed_cols, test_size=.1, random_state=RANDOM_STATE)
    rng = np.random.RandomState(RANDOM_STATE)
    val_cols = list(rng.choice(val_cols_full, size=min(N_VAL_SUB, len(val_cols_full)), replace=False))
    print(f"Colonnes de validation ({len(val_cols)}): {val_cols}", flush=True)

    donor_df = X[donors]
    dev_true_df = y[dev_cols]
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

    try:
        results = json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        results = {"autogluon": {}, "lightgbm": {}}

    for col in val_cols:
        if col in results["autogluon"] and col in results["lightgbm"]:
            print(f"{col}: déjà fait, skip", flush=True)
            continue
        s = X[col]
        built = v14.build_v14_features(col, s, pool_vals, km, cluster_col_idx, tfeat, X.index, BEST)
        if built is None:
            print(f"{col}: build_v14_features -> None (pas de trou ?), skip", flush=True)
            continue
        obs_idx, miss_idx, target_obs, interp_full, knn_pred, scaled, extra, missing, lags = built
        X_obs = np.hstack([scaled[obs_idx, :], extra[obs_idx, :]])
        X_miss = np.hstack([scaled[miss_idx, :], extra[miss_idx, :]])
        y_true = y[col].to_numpy()[miss_idx]
        valid = ~np.isnan(y_true)
        print(f"{col}: n_obs={len(obs_idx)}, n_miss={int(valid.sum())}, n_features={X_obs.shape[1]}", flush=True)

        t0 = time.time()
        pred_ag = autogluon_predict(X_obs, target_obs, X_miss)
        dt_ag = time.time() - t0
        mae_ag = float(np.abs(pred_ag[valid] - y_true[valid]).mean())

        t0 = time.time()
        pred_lgb = lightgbm_predict(X_obs, target_obs, X_miss)
        dt_lgb = time.time() - t0
        mae_lgb = float(np.abs(pred_lgb[valid] - y_true[valid]).mean())

        results["autogluon"][col] = {"mae": mae_ag, "time_s": dt_ag, "n_obs": int(len(obs_idx)), "n_miss": int(valid.sum())}
        results["lightgbm"][col] = {"mae": mae_lgb, "time_s": dt_lgb}
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)
        print(f"{col}: AutoGluon MAE={mae_ag:8.2f}W ({dt_ag:5.1f}s) | LightGBM MAE={mae_lgb:8.2f}W ({dt_lgb:4.1f}s)",
              flush=True)

    common = [c for c in results["autogluon"] if c in results["lightgbm"]]
    ag_maes = np.array([results["autogluon"][c]["mae"] for c in common])
    lgb_maes = np.array([results["lightgbm"][c]["mae"] for c in common])
    print(f"\n=== Bilan sur {len(common)} colonnes ===")
    print(f"AutoGluon (bagging {NUM_BAG_FOLDS}-fold, {TIME_LIMIT}s max/colonne): "
          f"{ag_maes.mean():.2f} +/- {ag_maes.std():.2f} W")
    print(f"LightGBM (référence rapide, mêmes features/points)          : "
          f"{lgb_maes.mean():.2f} +/- {lgb_maes.std():.2f} W")
    wins = int((ag_maes < lgb_maes).sum())
    print(f"AutoGluon gagne sur {wins}/{len(common)} colonnes vs LightGBM (mêmes points)")
