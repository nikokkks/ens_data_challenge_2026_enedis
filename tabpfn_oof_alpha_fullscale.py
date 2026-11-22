"""Reprise à pleine échelle du blend alpha appris (cf. `tabpfn_oof_alpha.py`,
rejeté à 60 puis 300 colonnes d'entraînement -- benchmark.md §14/§15ter, le
repo concluait explicitement que le budget d'entraînement était insuffisant
sans jamais l'avoir testé à l'échelle des 1000 colonnes disponibles).

Réutilise telles quelles les fonctions de `tabpfn_oof_alpha.py`
(`build_rows_for_column`, `build_context`, `collect_rows`) -- seule
l'ÉCHELLE change (700 train / 150 test / 150 helper au lieu de 60-300/100).
Split meta-train/meta-test sur seed=999 (disjoint des seeds 42-51 déjà
utilisés pour V12/V14, donc pas de fuite).

Deux évaluations :
  1. "petite" (rapide) : comparaison directe sur meta_test (interp seul /
     TabPFN seul / blend V14 / blend appris Ridge / blend appris GBM),
     comme dans `tabpfn_oof_alpha.py` mais avec 700 colonnes d'entraînement.
  2. "standard" (si le blend appris bat V14 sur meta_test) : le meta-modèle
     gagnant est injecté dans le pipeline V14 réel via l'option
     `learned_blend_model` (cf. `tabpfn_v14_ablation.predict_column_v14`) et
     confirmé sur le protocole robuste 10x10 standard, comparé à la
     référence V14 déjà enregistrée (63.24 +/- 30.46 W, mêmes seeds).
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import tabpfn_client
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from evaluate import load_data
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14
from tabpfn_oof_alpha import build_context, collect_rows, RANDOM_STATE_OOF

RESULTS_PATH = "tabpfn_oof_alpha_fullscale_results.json"
MODEL_CACHE = "tabpfn_oof_alpha_fullscale_meta_model.pkl"
ROWS_CACHE_TRAIN = "tabpfn_oof_alpha_fullscale_rows_train.pkl"
ROWS_CACHE_TEST = "tabpfn_oof_alpha_fullscale_rows_test.pkl"


def cached_collect(cols, X, y, donors, pool_vals, km, cluster_col_idx, tfeat, label, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if set(r["col"] for r in cached) == set(cols):
            print(f"  [{label}] repris du cache : {len(cached)} colonnes", flush=True)
            return cached
    rows = collect_rows(cols, X, y, donors, pool_vals, km, cluster_col_idx, tfeat, label)
    with open(cache_path, "wb") as f:
        pickle.dump(rows, f)
    return rows


def stack(rows):
    interp = np.concatenate([r["interp"] for r in rows])
    model = np.concatenate([r["model"] for r in rows])
    meta = np.concatenate([r["meta"] for r in rows])
    yv = np.concatenate([r["y"] for r in rows])
    return interp, model, meta, yv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=700)
    p.add_argument("--n-test", type=int, default=150)
    p.add_argument("--n-helper", type=int, default=145)  # laisse une marge (<1000 strict, contrainte train_test_split)
    p.add_argument("--confirm", action="store_true", help="rejoue le protocole 10x10 standard avec le meta-modele gagnant")
    a = p.parse_args()

    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant")
    tabpfn_client.set_access_token(token)
    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]
    holed = list(y.columns)

    N_TRAIN, N_TEST, N_HELPER = a.n_train, a.n_test, a.n_helper
    assert N_TRAIN + N_TEST + N_HELPER <= len(holed)
    meta_train_cols, rest = train_test_split(holed, test_size=1 - N_TRAIN / len(holed), random_state=RANDOM_STATE_OOF)
    meta_test_cols, rest2 = train_test_split(rest, test_size=1 - N_TEST / len(rest), random_state=RANDOM_STATE_OOF + 1)
    helper_cols, _ = train_test_split(rest2, test_size=1 - N_HELPER / len(rest2), random_state=RANDOM_STATE_OOF + 2)
    assert not (set(meta_train_cols) & set(meta_test_cols) & set(helper_cols))
    print(f"meta_train={len(meta_train_cols)} meta_test={len(meta_test_cols)} helper={len(helper_cols)}", flush=True)

    pool_vals, km, cluster_col_idx, tfeat = build_context(X, y, donors, meta_train_cols + meta_test_cols, helper_cols)

    t0 = time.time()
    train_rows = cached_collect(meta_train_cols, X, y, donors, pool_vals, km, cluster_col_idx, tfeat, "meta_train", ROWS_CACHE_TRAIN)
    test_rows = cached_collect(meta_test_cols, X, y, donors, pool_vals, km, cluster_col_idx, tfeat, "meta_test", ROWS_CACHE_TEST)
    print(f"Collecte terminée en {time.time()-t0:.1f}s", flush=True)

    interp_tr, model_tr, meta_tr, y_tr = stack(train_rows)
    interp_te, model_te, meta_te, y_te = stack(test_rows)
    print(f"points d'entraînement meta : {len(y_tr)}  points de test meta : {len(y_te)}", flush=True)

    X_meta_tr = np.column_stack([interp_tr, model_tr, meta_tr])
    X_meta_te = np.column_stack([interp_te, model_te, meta_te])

    meta_ridge = Ridge(alpha=1.0, positive=True)
    meta_ridge.fit(X_meta_tr, y_tr)
    mae_ridge = float(np.abs(meta_ridge.predict(X_meta_te) - y_te).mean())

    meta_gbm = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=300, learning_rate=.05,
        max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, random_state=42)
    meta_gbm.fit(X_meta_tr, y_tr)
    mae_gbm = float(np.abs(meta_gbm.predict(X_meta_te) - y_te).mean())

    mae_interp_only = float(np.abs(interp_te - y_te).mean())
    mae_model_only = float(np.abs(model_te - y_te).mean())

    runs_norm, edges_norm, period_agree, vol = meta_te[:, 0], meta_te[:, 1], meta_te[:, 2], meta_te[:, 3]
    runs, edges = runs_norm * 50.0, edges_norm * 20.0
    base_alpha = np.array([feb.continuous_alpha(r, e, True) for r, e in zip(runs, edges)])
    alpha_v14 = base_alpha * (1.0 - 1.0 * period_agree)
    alpha_v14 = alpha_v14 * np.array([feb.volatility_alpha_mult(v_, vol_weight=0.8, cap=0.6) for v_ in vol])
    alpha_v14 = np.clip(alpha_v14, 0.0, 0.95)
    pred_v14 = alpha_v14 * interp_te + (1 - alpha_v14) * model_te
    mae_v14 = float(np.abs(pred_v14 - y_te).mean())

    print(f"\n=== Comparaison sur meta_test ({len(meta_test_cols)} colonnes, {len(y_te)} points, "
          f"entraîné sur {len(meta_train_cols)} colonnes / {len(y_tr)} points) ===", flush=True)
    print(f"interpolation seule            : {mae_interp_only:8.3f} W", flush=True)
    print(f"TabPFN seul (modele)           : {mae_model_only:8.3f} W", flush=True)
    print(f"blend V14 (formule main)       : {mae_v14:8.3f} W", flush=True)
    print(f"blend appris (Ridge OOF)       : {mae_ridge:8.3f} W", flush=True)
    print(f"blend appris (GBM OOF)         : {mae_gbm:8.3f} W", flush=True)

    winner_name = "ridge" if mae_ridge < mae_gbm else "gbm"
    winner_model = meta_ridge if mae_ridge < mae_gbm else meta_gbm
    winner_mae = min(mae_ridge, mae_gbm)

    result = {
        "n_train_cols": len(meta_train_cols), "n_test_cols": len(meta_test_cols),
        "n_train_points": len(y_tr), "n_test_points": len(y_te),
        "mae_interp_only": mae_interp_only, "mae_model_only": mae_model_only,
        "mae_v14_formula": mae_v14, "mae_learned_ridge": mae_ridge, "mae_learned_gbm": mae_gbm,
        "winner": winner_name, "winner_beats_v14_on_meta_test": bool(winner_mae < mae_v14),
    }
    json.dump(result, open(RESULTS_PATH, "w"), indent=2)
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump({"name": winner_name, "model": winner_model}, f)

    if winner_mae >= mae_v14:
        print(f"\nLe meilleur blend appris ({winner_name}, {winner_mae:.3f} W) ne bat PAS "
              f"la formule V14 ({mae_v14:.3f} W) même à cette échelle -- pas de confirmation "
              f"10x10 lancée (inutile de dépenser du budget dessus).", flush=True)
        return
    print(f"\nLe blend appris ({winner_name}) bat la formule V14 sur meta_test "
          f"({winner_mae:.3f} vs {mae_v14:.3f} W) -- lancer avec --confirm pour la "
          f"confirmation 10x10 standard.", flush=True)

    if a.confirm:
        print("\n=== Confirmation 10x10 standard (pipeline V14 réel + blend appris) ===", flush=True)
        opts = {**v14.DEFAULT_OPTS, "volatility_alpha": True, "vol_weight": 0.8,
                "periodic_alpha_strength": 1.0, "extra_lags": (24,), "neighbor_rolling": True,
                "learned_blend_model": winner_model}
        scores = v14.run_config(f"oof_alpha_fullscale_{winner_name}", opts, X, y, donors, n_repeats=10, n_val_sub=10)
        try:
            v14_confirm = json.load(open("tabpfn_v14_confirm_results.json"))
            delta = scores - np.array(v14_confirm["scores"])
            print(f"\nV14 (référence)         : {v14_confirm['mean']:.3f} +/- {v14_confirm['std']:.3f} W", flush=True)
            print(f"blend appris ({winner_name}, fullscale) : {scores.mean():.3f} +/- {scores.std():.3f} W  "
                  f"delta={scores.mean()-v14_confirm['mean']:+.3f} W  wins={int((delta<0).sum())}/10", flush=True)
        except FileNotFoundError:
            pass
        result["confirm_10x10_scores"] = scores.tolist()
        result["confirm_10x10_mean"] = float(scores.mean())
        result["confirm_10x10_std"] = float(scores.std())
        json.dump(result, open(RESULTS_PATH, "w"), indent=2)


if __name__ == "__main__":
    main()
