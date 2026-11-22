"""Tuning AutoGluon natif-NaN en CV 5-fold, puis confirmation en 10x10.

Le tuning global dispose de 300 secondes au total. Il fige la composition
et les hyperparamètres de l'ensemble avant l'évaluation. L'évaluation finale
réentraîne seulement les modèles et leurs poids pour chaque colonne. Aucun
ffill/bfill, aucune interpolation et aucun fallback manuel n'est appliqué :
AutoGluon reçoit les NaN des features et produit seul la prédiction finale.

Mode rapide : ``AG_V14_SMOKE=1`` (1 colonne de tuning, CV 2-fold, final 1x1).
Budgets configurables : ``AG_V14_TUNE_TIME_LIMIT`` et
``AG_V14_FIT_TIME_LIMIT``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluate import load_data
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14

RANDOM_STATE = 42
SMOKE = os.environ.get("AG_V14_SMOKE", "0") == "1"
N_REPEATS = 1 if SMOKE else 10
N_VAL_SUB = 1 if SMOKE else 10
N_TUNING_COLUMNS = 1 if SMOKE else 5
TUNING_TIME_LIMIT = int(os.environ.get("AG_V14_TUNE_TIME_LIMIT", "30" if SMOKE else "3600"))
FINAL_FIT_TIME_LIMIT = int(os.environ.get("AG_V14_FIT_TIME_LIMIT", "30" if SMOKE else "3600"))
NUM_BAG_FOLDS = 2 if SMOKE else 5
NUM_SELECTED_MODELS = 3
ARTIFACT_VERSION = 3

TUNING_PATH = Path("tabpfn_v14_autogluon_tuning_native_nan_v3.json")
PROGRESS_PATH = Path("tabpfn_v14_autogluon_confirm_progress_native_nan_v3.json")
RESULTS_PATH = Path("tabpfn_v14_autogluon_confirm_results_native_nan_v3.json")
BASELINE_PATH = Path("tabpfn_v14_confirm_results.json")

v14.MAX_WORKERS = 1
BEST = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
            extra_lags=(24,), neighbor_rolling=True, native_nan=True)

# Suffixes stables : ils permettent de retrouver la recette exacte depuis le
# leaderboard, sans sérialiser les paramètres internes des modèles entraînés.
CANDIDATES = [
    {"id": "gbm_default", "family": "GBM", "params": {}},
    {"id": "gbm_slow", "family": "GBM",
     "params": {"learning_rate": 0.05, "num_boost_round": 500}},
    {"id": "cat_default", "family": "CAT", "params": {}},
    {"id": "cat_slow", "family": "CAT",
     "params": {"learning_rate": 0.05, "iterations": 500}},
    {"id": "xt_squared", "family": "XT", "params": {"criterion": "squared_error"}},
    {"id": "xt_absolute", "family": "XT", "params": {"criterion": "absolute_error"}},
    {"id": "rf_squared", "family": "RF", "params": {"criterion": "squared_error"}},
    {"id": "rf_absolute", "family": "RF", "params": {"criterion": "absolute_error"}},
]


def atomic_json_dump(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def config_signature(selected):
    payload = {"version": ARTIFACT_VERSION, "best": BEST, "selected": selected,
               "folds": NUM_BAG_FOLDS, "fit_limit": FINAL_FIT_TIME_LIMIT}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def candidate_hyperparameters(candidates):
    result = {}
    for candidate in candidates:
        params = deepcopy(candidate["params"])
        params["ag_args"] = {"name_suffix": f"_{candidate['id']}"}
        result.setdefault(candidate["family"], []).append(params)
    return result


def build_context(X, y, donors, dev_cols, target_cols):
    donor_df, dev_true_df = X[donors], y[dev_cols]
    pool = pd.concat([donor_df, dev_true_df], axis=1)
    pool_vals = pool.to_numpy(dtype=np.float64)
    tfeat = feb.time_features_full_seasonality(
        X[target_cols].index, ("hour", "dow", "dom", "year"))
    nan_count = int(np.isnan(pool_vals).sum())
    print(f"[native_nan] pool brut: shape={pool_vals.shape}, NaN={nan_count} "
          f"({100 * nan_count / pool_vals.size:.3f}%), aucun fill/clustering", flush=True)
    return pool_vals, None, {}, tfeat


def tuning_frame(X, y, donors):
    """Corpus déterministe utilisant uniquement les points observés."""
    holed_cols = list(y.columns)
    dev_cols, tune_pool = train_test_split(
        holed_cols, test_size=.1, random_state=RANDOM_STATE)
    rng = np.random.RandomState(RANDOM_STATE)
    tune_cols = list(rng.choice(
        tune_pool, size=min(N_TUNING_COLUMNS, len(tune_pool)), replace=False))
    pool_vals, km, cluster_col_idx, tfeat = build_context(
        X, y, donors, dev_cols, tune_cols)
    frames = []
    print(f"[tuning 1/4] Colonnes: {tune_cols}", flush=True)
    for col in tune_cols:
        built = v14.build_v14_features(
            col, X[col], pool_vals, km, cluster_col_idx, tfeat, X.index, BEST)
        if built is None:
            continue
        obs_idx, _, target_obs, _, _, scaled, extra, _, _ = built
        features = np.hstack([scaled[obs_idx, :], extra[obs_idx, :]])
        median = float(np.median(target_obs))
        scale = float(np.quantile(target_obs, .75) - np.quantile(target_obs, .25))
        if not np.isfinite(scale) or scale < 1e-9:
            scale = float(np.std(target_obs)) or 1.0
        frame = pd.DataFrame(features, columns=[f"f{i}" for i in range(features.shape[1])])
        frame["target"] = (np.asarray(target_obs) - median) / scale
        frames.append(frame)
        print(f"[tuning 2/4] {col}: {len(frame)} lignes, "
              f"{features.shape[1]} features", flush=True)
    if not frames:
        raise RuntimeError("Aucune donnée utilisable pour le tuning")
    return pd.concat(frames, ignore_index=True), tune_cols


def select_global_ensemble(X, y, donors):
    expected = {"artifact_version": ARTIFACT_VERSION, "native_nan": True,
                "candidates": CANDIDATES,
                "num_bag_folds": NUM_BAG_FOLDS,
                "tuning_time_limit": TUNING_TIME_LIMIT,
                "n_tuning_columns": N_TUNING_COLUMNS}
    if TUNING_PATH.exists():
        saved = json.loads(TUNING_PATH.read_text(encoding="utf-8"))
        if all(saved.get(k) == value for k, value in expected.items()):
            print(f"[tuning] Artefact compatible rechargé: {TUNING_PATH}")
            print(f"[tuning] Ensemble figé: {[c['id'] for c in saved['selected']]}")
            return saved
        print("[tuning] Artefact incompatible ignoré (non supprimé).", flush=True)

    from autogluon.tabular import TabularPredictor
    frame, tune_cols = tuning_frame(X, y, donors)
    tmp_root = tempfile.mkdtemp(prefix="ag_v14_tuning_")
    model_path = os.path.join(tmp_root, "models")
    started = time.time()
    try:
        print(f"[tuning 3/4] CV={NUM_BAG_FOLDS}, MAE, budget total="
              f"{TUNING_TIME_LIMIT}s, candidats={len(CANDIDATES)}", flush=True)
        predictor = TabularPredictor(
            label="target", eval_metric="mean_absolute_error",
            problem_type="regression", path=model_path, verbosity=2).fit(
                frame, time_limit=TUNING_TIME_LIMIT,
                num_bag_folds=NUM_BAG_FOLDS, num_bag_sets=1,
                hyperparameters=candidate_hyperparameters(CANDIDATES),
                ag_args_fit={"num_cpus": 4})
        leaderboard = predictor.leaderboard(silent=True)
        print("[tuning 4/4] Leaderboard CV (score = -MAE normalisée):")
        print(leaderboard.to_string(index=False), flush=True)
        standalone = leaderboard[
            ~leaderboard["model"].str.contains("WeightedEnsemble", case=False, na=False)]
        ranked = []
        for model_name in standalone.sort_values("score_val", ascending=False)["model"]:
            candidate = next((c for c in CANDIDATES if c["id"] in str(model_name)), None)
            if candidate and candidate["id"] not in {c["id"] for c in ranked}:
                ranked.append(candidate)
        if not ranked:
            raise RuntimeError("Impossible d'associer le leaderboard aux candidats")
        selected = ranked[:min(NUM_SELECTED_MODELS, len(ranked))]
        records = json.loads(leaderboard.to_json(orient="records"))
        artifact = {**expected, "tuning_columns": tune_cols, "n_rows": len(frame),
                    "n_features": len(frame.columns) - 1,
                    "elapsed_s": time.time() - started,
                    "leaderboard": records, "selected": selected}
        artifact["signature"] = config_signature(selected)
        atomic_json_dump(artifact, TUNING_PATH)
        print(f"[tuning] Ensemble retenu: {[c['id'] for c in selected]}")
        print(f"[tuning] Signature: {artifact['signature']}", flush=True)
        return artifact
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


class AutoGluonReg:
    """Régresseur branchable dans ``predict_column_v14``."""
    selected = None
    fit_counter = 0

    def __init__(self, time_limit=FINAL_FIT_TIME_LIMIT, num_bag_folds=NUM_BAG_FOLDS):
        self.time_limit, self.num_bag_folds = time_limit, num_bag_folds
        self.tmp_dir = None

    def fit(self, X, y):
        from autogluon.tabular import TabularPredictor
        if not self.selected:
            raise RuntimeError("Composition AutoGluon non sélectionnée")
        self.cols = [f"f{i}" for i in range(X.shape[1])]
        if not np.isfinite(y).all():
            raise ValueError("La cible AutoGluon contient des NaN/Inf")
        df = pd.DataFrame(X, columns=self.cols)
        df["target"] = y
        self.tmp_dir = tempfile.mkdtemp(prefix="ag_v14_final_")
        model_path = os.path.join(self.tmp_dir, "models")
        type(self).fit_counter += 1
        step = type(self).fit_counter
        feature_nan = int(np.isnan(np.asarray(X, dtype=float)).sum())
        print(f"      [fit {step}] {len(df)} lignes, {len(self.cols)} features, "
              f"NaN={feature_nan} ({100 * feature_nan / np.asarray(X).size:.3f}%), "
              f"ensemble={[c['id'] for c in self.selected]}, CV={self.num_bag_folds}", flush=True)
        try:
            self.predictor = TabularPredictor(
                label="target", eval_metric="mean_absolute_error",
                problem_type="regression", path=model_path, verbosity=1).fit(
                    df, time_limit=self.time_limit,
                    num_bag_folds=self.num_bag_folds, num_bag_sets=1,
                    hyperparameters=candidate_hyperparameters(self.selected),
                    ag_args_fit={"num_cpus": 4})
            board = self.predictor.leaderboard(silent=True)
            cols = [c for c in ("model", "score_val", "fit_time") if c in board]
            print(f"      [fit {step}] leaderboard:\n{board[cols].to_string(index=False)}", flush=True)
            return self
        except Exception:
            self.close()
            raise

    def predict(self, X):
        try:
            missing = int(np.isnan(np.asarray(X, dtype=float)).sum())
            print(f"      [predict] NaN transmis à AutoGluon: {missing}", flush=True)
            return self.predictor.predict(pd.DataFrame(X, columns=self.cols)).to_numpy()
        finally:
            self.close()

    def close(self):
        if self.tmp_dir:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            self.tmp_dir = None


def load_progress(signature):
    empty = {"artifact_version": ARTIFACT_VERSION, "signature": signature, "repeats": {}}
    if not PROGRESS_PATH.exists():
        return empty
    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    if progress.get("signature") != signature:
        print("[reprise] Checkpoint incompatible ignoré (non supprimé).", flush=True)
        return empty
    return progress


def eval_split_checkpointed(X, y, donors, dev_cols, val_cols, opts,
                             progress, repeat_key, signature):
    pool_vals, km, cluster_col_idx, tfeat = build_context(X, y, donors, dev_cols, val_cols)
    rep_progress = progress["repeats"].setdefault(repeat_key, {})
    all_errs, fallbacks = [], 0
    for position, col in enumerate(val_cols, 1):
        saved = rep_progress.get(col)
        if saved and saved.get("signature") == signature:
            all_errs.append(np.asarray(saved["abs_err"]))
            fallbacks += int(saved.get("fallback", False))
            print(f"    [{repeat_key} {position}/{len(val_cols)}] {col}: reprise, "
                  f"MAE={np.mean(saved['abs_err']):.2f} W", flush=True)
            continue
        started = time.time()
        _, abs_err, error = v14.complete_column_v14(
            col, X[col], y[col], pool_vals, km, cluster_col_idx,
            tfeat, X.index, opts)
        elapsed = time.time() - started
        if abs_err is None:
            print(f"    [{repeat_key}] {col}: aucun trou", flush=True)
            continue
        all_errs.append(abs_err)
        saved = {"signature": signature, "abs_err": abs_err.tolist(),
                 "mae": float(abs_err.mean()), "fallback": bool(error),
                 "error": error, "time_s": elapsed}
        rep_progress[col] = saved
        atomic_json_dump(progress, PROGRESS_PATH)
        suffix = f", FALLBACK: {error}" if error else ""
        print(f"    [{repeat_key} {position}/{len(val_cols)}] {col}: "
              f"MAE={saved['mae']:.2f} W ({elapsed:.1f}s{suffix})", flush=True)
        fallbacks += int(bool(error))
    if not all_errs:
        raise RuntimeError(f"Aucune erreur calculée pour {repeat_key}")
    return float(np.concatenate(all_errs).mean()), fallbacks


def main():
    warnings.filterwarnings("default")
    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]
    tuning = select_global_ensemble(X, y, donors)
    selected, signature = tuning["selected"], tuning["signature"]
    AutoGluonReg.selected = selected
    opts = {**BEST, "regressor_cls": AutoGluonReg}
    progress = load_progress(signature)

    print(f"\n=== ÉVALUATION AUTOGluon native_nan {N_REPEATS}x{N_VAL_SUB} ===")
    print("Aucun ffill/bfill, aucune interpolation, aucun blend/fallback manuel.", flush=True)
    print(f"Configuration figée: {[c['id'] for c in selected]} | "
          f"signature={signature} | CV={NUM_BAG_FOLDS}", flush=True)
    holed_cols, scores, repeat_details = list(y.columns), [], []
    for i in range(N_REPEATS):
        seed_i, repeat_key = RANDOM_STATE + i, f"repeat{i}"
        dev_cols, val_cols_full = train_test_split(
            holed_cols, test_size=.1, random_state=seed_i)
        rng = np.random.RandomState(seed_i)
        val_cols = list(rng.choice(
            val_cols_full, size=min(N_VAL_SUB, len(val_cols_full)), replace=False))
        started = time.time()
        mae, fallbacks = eval_split_checkpointed(
            X, y, donors, dev_cols, val_cols, opts, progress, repeat_key, signature)
        elapsed = time.time() - started
        scores.append(mae)
        repeat_details.append({"repeat": i + 1, "seed": seed_i, "columns": val_cols,
                               "mae": mae, "fallbacks": fallbacks, "time_s": elapsed})
        print(f"  [final] repeat {i + 1}/{N_REPEATS}: MAE={mae:.2f} W "
              f"({elapsed:.1f}s, fallbacks={fallbacks})", flush=True)

    scores_array = np.asarray(scores)
    result = {"artifact_version": ARTIFACT_VERSION, "native_nan": True,
              "signature": signature,
              "selected": selected, "tuning_path": str(TUNING_PATH),
              "num_bag_folds": NUM_BAG_FOLDS, "scores": scores,
              "mean": float(scores_array.mean()), "std": float(scores_array.std()),
              "repeats": repeat_details}
    atomic_json_dump(result, RESULTS_PATH)
    print(f"\n[final] MAE moyenne: {result['mean']:.2f} +/- {result['std']:.2f} W", flush=True)

    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["scores"]
        print("\nrepeat | baseline TabPFN V14 | AutoGluon figé | delta")
        wins = losses = ties = 0
        for i, (base, score) in enumerate(zip(baseline, scores), 1):
            delta = score - base
            tag = "gagné" if delta < -1e-9 else ("perdu" if delta > 1e-9 else "égalité")
            wins += tag == "gagné"; losses += tag == "perdu"; ties += tag == "égalité"
            print(f"{i:6d} | {base:19.2f} | {score:15.2f} | {delta:+7.2f} ({tag})")
        print(f"{wins} gagnés / {losses} perdus / {ties} égalités")
    print(f"Résultats écrits dans {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
