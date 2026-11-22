"""Ensembling TabPFN + LightGBM sur le pipeline V14 (mêmes features, deux
régresseurs différents), blend pondéré des prédictions finales déjà mélangées
à l'interpolation (pas seulement des `model_pred` bruts -- on ensemble la
sortie complète de chaque pipeline).

Motivation (cf. discussion 2026-08-17) : 24 configurations d'ajout/retrait de
features testées sur V14 seul cette session, 0 gain confirmé (benchmark.md
§16-18) -- un levier structurel jamais retenté avec une méthodologie propre
est d'ensembler TabPFN avec un second régresseur diversifié (le seul essai
historique, §5/§7 "stacking classique", manquait d'OOF et datait d'avant le
blend alpha actuel).

Protocole : repeat 1 (seed=42) pour le balayage de poids (rapide), puis
confirmation 10x10 du meilleur poids sur le protocole standard, comparée à
la référence V14 déjà enregistrée (63.24 +/- 30.46 W).
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tabpfn_client
from sklearn.model_selection import train_test_split

from evaluate import load_data
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14
from tabpfn_v15_bench import LightGBMMae

RESULTS_PATH = "tabpfn_gbm_ensemble_results.json"
RANDOM_STATE = 42
V14_BASE = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                extra_lags=(24,), neighbor_rolling=True)


def predict_both(col, s, y_true_col, pool_vals, km, cluster_col_idx, tfeat, index):
    """Renvoie (col, filled_tabpfn, filled_lgbm, y_true, missing_mask) pour
    une colonne -- les DEUX pipelines partagent exactement les mêmes
    features/alpha, seul `regressor_cls` change."""
    opts_tabpfn = {**V14_BASE}
    opts_lgbm = {**V14_BASE, "regressor_cls": LightGBMMae, "iterative_passes": 2}
    _, filled_t, err_t = v14.predict_column_v14(col, s, pool_vals, km, cluster_col_idx, tfeat, index, opts_tabpfn)
    _, filled_g, err_g = v14.predict_column_v14(col, s, pool_vals, km, cluster_col_idx, tfeat, index, opts_lgbm)
    missing = s.isna().to_numpy() & y_true_col.notna().to_numpy()
    return col, filled_t.to_numpy()[missing], filled_g.to_numpy()[missing], y_true_col.to_numpy()[missing], bool(err_t)


def eval_split_ensemble(X, y, donors, dev, val, weights, n_clusters=15):
    donor_df = X[donors]
    dev_true_df = y[dev]
    val_holed_df = X[val]
    pool = pd.concat([donor_df, dev_true_df], axis=1)
    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=n_clusters, random_state=RANDOM_STATE)
    dev_filled = dev_true_df.ffill().bfill()
    Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
    pool_labels = pd.concat([donor_labels, dev_labels])
    pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
    cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(val_holed_df.index, ("hour", "dow", "dom", "year"))

    tabpfn_all, gbm_all, y_all = [], [], []
    fallbacks = 0
    with ThreadPoolExecutor(max_workers=v14.MAX_WORKERS) as executor:
        futures = [executor.submit(predict_both, c, val_holed_df[c], y[c], pool_vals, km,
                                    cluster_col_idx, tfeat, val_holed_df.index) for c in val]
        for fut in as_completed(futures):
            col, ft, fg, yt, err = fut.result()
            tabpfn_all.append(ft); gbm_all.append(fg); y_all.append(yt)
            fallbacks += bool(err)
    tabpfn_all = np.concatenate(tabpfn_all); gbm_all = np.concatenate(gbm_all); y_all = np.concatenate(y_all)

    out = {}
    for w in weights:
        pred = w * tabpfn_all + (1 - w) * gbm_all
        out[w] = float(np.abs(pred - y_all).mean())
    out["tabpfn_only"] = float(np.abs(tabpfn_all - y_all).mean())
    out["gbm_only"] = float(np.abs(gbm_all - y_all).mean())
    return out, fallbacks


def run_search(X, y, donors, weights, nval):
    seed = RANDOM_STATE
    dev, val_full = train_test_split(list(y.columns), test_size=.1, random_state=seed)
    val = list(np.random.RandomState(seed).choice(val_full, min(nval, len(val_full)), replace=False))
    t0 = time.time()
    out, fb = eval_split_ensemble(X, y, donors, dev, val, weights)
    print(f"[repeat1, nval={nval}] {time.time()-t0:.1f}s fallbacks={fb}", flush=True)
    for w in weights:
        print(f"  poids tabpfn={w:.2f} (gbm={1-w:.2f}) : MAE={out[w]:.3f} W", flush=True)
    print(f"  tabpfn seul (w=1.0, référence V14) : {out['tabpfn_only']:.3f} W", flush=True)
    print(f"  lgbm seul (w=0.0)                  : {out['gbm_only']:.3f} W", flush=True)
    json.dump({"repeat1": out}, open(RESULTS_PATH, "w"), indent=2)
    return out


def run_confirm(X, y, donors, weight, n_repeats=10, nval=10):
    results = {}
    try:
        results = json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        pass
    key = f"confirm_w{weight}"
    if key in results:
        print(f"[{key}] déjà fait -- skip", flush=True)
        return
    scores = []
    for i in range(n_repeats):
        seed = RANDOM_STATE + i
        dev, val_full = train_test_split(list(y.columns), test_size=.1, random_state=seed)
        val = list(np.random.RandomState(seed).choice(val_full, min(nval, len(val_full)), replace=False))
        t0 = time.time()
        out, fb = eval_split_ensemble(X, y, donors, dev, val, [weight])
        scores.append(out[weight])
        print(f"  repeat {i+1}/{n_repeats}: MAE={out[weight]:.3f} W ({time.time()-t0:.1f}s, fallbacks={fb})", flush=True)
    scores = np.array(scores)
    results[key] = {"weight": weight, "scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}
    json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    try:
        v14_confirm = json.load(open("tabpfn_v14_confirm_results.json"))
        delta = scores - np.array(v14_confirm["scores"])
        print(f"\n=== Comparaison finale (10x10, mêmes seeds) ===", flush=True)
        print(f"  V14 seul (référence)        : {v14_confirm['mean']:.3f} +/- {v14_confirm['std']:.3f} W", flush=True)
        print(f"  ensemble w={weight}           : {scores.mean():.3f} +/- {scores.std():.3f} W  "
              f"delta={scores.mean()-v14_confirm['mean']:+.3f} W  wins={int((delta<0).sum())}/10", flush=True)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nval", type=int, default=10)
    p.add_argument("--confirm", type=float, default=None, help="poids tabpfn à confirmer sur 10x10")
    a = p.parse_args()

    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant")
    tabpfn_client.set_access_token(token)
    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]

    if a.confirm is not None:
        run_confirm(X, y, donors, a.confirm)
    else:
        run_search(X, y, donors, [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0], a.nval)
