"""Criblage LightGBM (early stopping sur MAE) des features EXACTES de
tabpfn_exact_tabfm_bench.py (= tabfm_lora_finetune.V14_FEATURE_OPTS,
identiques à generate_submission_tabpfn_v14.py), puis de chacune des
features candidates de feature_catalog_v16.py, ajoutée UNE PAR UNE à ce
même socle -- jamais combinées entre elles (demande explicite : une
évaluation baseline + une évaluation par nouveau type de feature).

Régresseur : ESLightGBM (gbm_optuna_vs_tabpfn.py) -- LightGBM objectif MAE,
early stopping (50 rounds, validation interne = 15% des points observés de
CHAQUE colonne, jamais les points réellement manquants), déjà le motif
établi dans ce dépôt pour comparer des jeux de features avec un régresseur
rapide.

Protocole : 'Robust 5x100' documenté dans CLAUDE.md/evaluate.py --
n_repeats=5, test_size=0.10 (900 colonnes dev / 100 colonnes val),
random_state=42 (graines 42..46 par repeat via train_test_split), IDENTIQUE
au protocole utilisé pour les comparaisons de régresseurs locaux rapides
(xgboost_bench*.py, feature_engineering_bench.py) -- comparable au tableau
§3 de benchmark.md, PAS au protocole 10x10 des criblages TabPFN (§13).

Note d'hypothèse : le pipeline TabFM-LoRA (tabfm_lora_finetune.make_episode)
tronque/pad les features à 96 colonnes et sous-échantillonne le contexte à
800 lignes -- des contraintes d'ingénierie propres au format "épisode" fixe
de TabFM/TabPFN. Ce script utilise les features BRUTES de
v14.build_v14_features (scaled + extra), sans ces deux contraintes
artificielles, puisque LightGBM n'a besoin ni de largeur fixe ni de cap sur
le nombre de lignes.

Checkpointé : chaque config terminée est écrite immédiatement dans
RESULTS_PATH ; relancer le script saute les configs déjà faites (safe to
kill/resume, cf. conventions *_bench.py du dépôt).
"""
from __future__ import annotations

import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from evaluate import load_data
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14
import tabfm_lora_finetune as ft
import feature_catalog_v16 as fc
from gbm_optuna_vs_tabpfn import ESLightGBM

RANDOM_STATE = 42
N_REPEATS = 5
N_VAL_SUB = 100  # "Robust 5x100" : split complet 10%, pas de sous-échantillonnage
MAX_WORKERS = 20
RESULTS_PATH = Path("lgbm_v14_feature_screen_results.json")
PROGRESS_LOG = Path("lgbm_v14_feature_screen.log")


def build_pool(X, y, donor_cols, dev_cols):
    donor_df = X[donor_cols]
    dev_true_df = y[dev_cols]
    pool = pd.concat([donor_df, dev_true_df], axis=1)
    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=15, random_state=RANDOM_STATE)
    dev_filled = dev_true_df.ffill().bfill()
    Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = (pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
                  if len(dev_cols) else pd.Series(dtype=int))
    pool_labels = pd.concat([donor_labels, dev_labels])
    pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
    cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(X.index, ("hour", "dow", "dom", "year"))
    return pool_vals, km, cluster_col_idx, tfeat


def complete_column_augmented(col, s, y_true_col, pool_vals, km, cluster_col_idx, tfeat, index, feature_fn):
    """Miroir de v14.predict_column_v14/complete_column_v14 (1 passe,
    opts=ft.V14_FEATURE_OPTS) : seule différence, `feature_fn` (si fourni)
    ajoute un bloc de features à `extra` AVANT le fit/predict, et le
    régresseur est ESLightGBM au lieu de TabPFN."""
    observed = s.notna().to_numpy()
    missing = ~observed
    if not missing.any():
        return col, None, None
    built = v14.build_v14_features(col, s, pool_vals, km, cluster_col_idx, tfeat, index, ft.V14_FEATURE_OPTS)
    if built is None:
        return col, None, None
    obs_idx, miss_idx, target_obs, interp_full, knn_pred, scaled, extra, missing2, lags = built
    if feature_fn is not None:
        ctx = dict(values=s.to_numpy(float), interp=interp_full.to_numpy(float),
                   index=index, scaled=scaled, missing=missing2, n=len(index))
        block = feature_fn(ctx)
        block = np.nan_to_num(np.asarray(block, dtype=np.float64), nan=0., posinf=0., neginf=0.)
        extra = np.hstack([extra, block])
    pred_neighbors = None
    error = None
    if len(obs_idx) >= 15:
        for attempt in range(3):
            try:
                make_reg = lambda: ESLightGBM(es_rounds=50, random_state=RANDOM_STATE, thread_count=1)
                pred_neighbors = feb.local_regression_pred(make_reg, scaled, obs_idx, miss_idx, target_obs, extra)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(1)
    if pred_neighbors is None:
        pred_neighbors = knn_pred
    runs = feb.run_lengths_for(missing2)
    edges = feb.edge_distance_for(missing2)
    alpha = np.array([feb.continuous_alpha(runs[i], edges[i], True) for i in miss_idx])
    # Fidèle à predict_column_v14 : feb.periodic_agreement est appelé SANS
    # les extra_lags (V14_FEATURE_OPTS["extra_lags"]=(24,) n'affecte que les
    # features, pas ce calcul d'agreement -- comportement V14 original
    # reproduit ici à l'identique, pas une nouvelle décision de conception).
    p_agree = feb.periodic_agreement(s.to_numpy(float))[miss_idx]
    alpha *= 1.0 - ft.V14_FEATURE_OPTS["periodic_alpha_strength"] * p_agree
    vol = feb.curve_volatility(s.to_numpy())
    alpha *= feb.volatility_alpha_mult(vol, vol_weight=ft.V14_FEATURE_OPTS["vol_weight"])
    alpha = np.clip(alpha, 0.0, 0.95)
    pred = alpha * interp_full.to_numpy()[miss_idx] + (1 - alpha) * pred_neighbors
    y_true = y_true_col.to_numpy()[miss_idx]
    valid = ~np.isnan(y_true)
    abs_err = np.abs(pred[valid] - y_true[valid])
    return col, abs_err, error


def eval_split(X, y, donor_cols, dev_cols, val_cols, feature_fn):
    pool_vals, km, cluster_col_idx, tfeat = build_pool(X, y, donor_cols, dev_cols)
    all_errs = []
    fallbacks = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(complete_column_augmented, c, X[c], y[c],
                                    pool_vals, km, cluster_col_idx, tfeat, X.index, feature_fn)
                   for c in val_cols]
        for fut in as_completed(futures):
            col, abs_err, error = fut.result()
            if abs_err is not None:
                all_errs.append(abs_err)
            if error:
                fallbacks += 1
    return float(np.concatenate(all_errs).mean()), fallbacks


def run_config(name, feature_fn, X, y, donor_cols, n_repeats=N_REPEATS, n_val_sub=N_VAL_SUB):
    holed_cols = list(y.columns)
    scores = []
    for i in range(n_repeats):
        seed_i = RANDOM_STATE + i
        dev_cols, val_cols_full = train_test_split(holed_cols, test_size=.1, random_state=seed_i)
        rng = np.random.RandomState(seed_i)
        val_cols = list(rng.choice(val_cols_full, size=min(n_val_sub, len(val_cols_full)), replace=False))
        t0 = time.time()
        mae, fallbacks = eval_split(X, y, donor_cols, dev_cols, val_cols, feature_fn)
        dt = time.time() - t0
        scores.append(mae)
        msg = f"  [{name}] repeat {i + 1}/{n_repeats}: MAE={mae:8.3f} W ({dt:.1f}s, fallbacks={fallbacks})"
        print(msg, flush=True)
        with open(PROGRESS_LOG, "a") as f:
            f.write(msg + "\n")
    scores = np.array(scores)
    summary = f"  -> [{name}] mean={scores.mean():.3f} +/- {scores.std():.3f} W\n"
    print(summary, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(summary + "\n")
    return scores


def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def save_results(results):
    tmp = str(RESULTS_PATH) + ".tmp"
    Path(tmp).write_text(json.dumps(results, indent=2))
    os.replace(tmp, RESULTS_PATH)


def main():
    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donor_cols = [c for c in X.columns if not str(c).startswith("holed_")]
    results = load_results()

    if "0_baseline_v14_features" not in results:
        print("=== baseline: features exactes de tabpfn_exact_tabfm_bench.py "
              "(V14_FEATURE_OPTS) + LightGBM (early stopping MAE) ===", flush=True)
        t0 = time.time()
        scores = run_config("0_baseline_v14_features", None, X, y, donor_cols)
        results["0_baseline_v14_features"] = {
            "scores": scores.tolist(), "mean": float(scores.mean()),
            "std": float(scores.std()), "seconds": time.time() - t0}
        save_results(results)

    baseline_mean = results["0_baseline_v14_features"]["mean"]

    for name, meta in fc.CATALOG.items():
        key = f"feat_{name}"
        if key in results:
            print(f"skip {key} (déjà fait)", flush=True)
            continue
        print(f"=== {key} [{meta['category']}] : {meta['note']} ===", flush=True)
        t0 = time.time()
        try:
            scores = run_config(key, meta["fn"], X, y, donor_cols)
            results[key] = {
                "scores": scores.tolist(), "mean": float(scores.mean()),
                "std": float(scores.std()),
                "delta_vs_baseline": float(scores.mean() - baseline_mean),
                "category": meta["category"], "note": meta["note"],
                "seconds": time.time() - t0}
        except Exception as exc:
            results[key] = {"error": f"{type(exc).__name__}: {exc}", "category": meta["category"]}
            print(f"  !! {key} a échoué: {exc}", flush=True)
        save_results(results)

    print("\n=== Terminé ===", flush=True)


if __name__ == "__main__":
    main()
