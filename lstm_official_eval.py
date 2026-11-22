"""Évaluation OFFICIELLE du meilleur LSTM du sweep, protocole `10x10`.

Le sweep `lstm_bench.py` sélectionne la configuration sur la validation
TEMPORELLE (early stopping, MAE normalisée). Ce script prend le meilleur
checkpoint et le mesure sur le protocole officiel par COLONNES -- 10 repeats
x 10 colonnes `holed_*`, `random_state=42` -- donc directement comparable à
TabPFN I4 (66,24 W) et R0_lightgbm (69,75 W), avec un tableau par repeat.

Deux variantes sont mesurées :
  `standalone` : la prédiction aux trous est la sortie brute du réseau ;
  `blend`      : mélange alpha-interpolation identique au reste du dépôt,
                 pour ne pas pénaliser le réseau sur les trous courts où
                 l'interpolation est déjà quasi optimale.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch

from evaluate import load_data, official_mae, fill_nan_with_interpolation
import feature_engineering_bench as feb
import lstm_bench as lb

RANDOM_STATE = 42
N_REPEATS = 10
TEST_SIZE = 0.10
N_VAL_SUB = 10
RESULTS_PATH = "lstm_official_results.json"


def best_config():
    prog = json.load(open(lb.PROGRESS_PATH))
    ranked = sorted(prog.items(), key=lambda kv: kv[1]["best_val_mae_norm"])
    return ranked[0][0], ranked[0][1], ranked


def make_model_fn(net, tfeat, window, residual, blend):
    def fn(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            missing = s.isna().values
            if not missing.any():
                preds[col] = s
                continue
            filled = lb.impute_curve(net, s.values.astype(np.float64), tfeat,
                                     window, residual)
            if blend:
                miss_idx = np.where(missing)[0]
                interp = fill_nan_with_interpolation(s).values[miss_idx]
                run_len = feb.run_lengths_for(missing)
                edge = feb.edge_distance_for(missing)
                alphas = np.array([feb.continuous_alpha(run_len[m], edge[m], adaptive=True)
                                   for m in miss_idx])
                filled[miss_idx] = alphas * interp + (1 - alphas) * filled[miss_idx]
            preds[col] = pd.Series(filled, index=s.index)
        return pd.DataFrame(preds, index=val_holed_df.index)
    return fn


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    name, entry, ranked = best_config()
    print("Classement des configurations sur la validation TEMPORELLE :")
    for n, e in ranked:
        print(f"  {n:24s} val_mae_norm={e['best_val_mae_norm']:.4f}  "
              f"({e['train_seconds']:.0f}s)")
    print(f"\nMeilleure configuration : {name} -> {entry['config']}\n", flush=True)

    cfg = entry["config"]
    window, residual = cfg["window"], cfg.get("residual", False)

    X_train, y_train, _ = load_data()
    tfeat = lb.time_channels(X_train.index)
    net = lb.SeqImputer(**cfg)
    net.load_state_dict(torch.load(f"lstm_{name}.pt"))
    net.eval()

    baseline = json.load(open("advanced_regressors_results.json"))["R0_lightgbm_reference"]
    tabpfn = json.load(open("tabpfn_five_improvements_results.json"))["I4_confidence_alpha"]

    results = {}
    for variant, blend in [("standalone", False), ("blend_alpha_interp", True)]:
        print(f"=== LSTM {name} / {variant} ===", flush=True)
        model = make_model_fn(net, tfeat, window, residual, blend)
        scores = []
        holed_cols = list(y_train.columns)
        for i in range(N_REPEATS):
            seed_i = RANDOM_STATE + i
            dev_cols, val_full = train_test_split(holed_cols, test_size=TEST_SIZE,
                                                  random_state=seed_i)
            val_cols = list(np.random.RandomState(seed_i).choice(val_full,
                            size=N_VAL_SUB, replace=False))
            t0 = time.time()
            y_pred = model(donor_df=None, dev_holed_df=None, dev_true_df=None,
                           val_holed_df=X_train[val_cols])
            mae = official_mae(y_train[val_cols], y_pred, X_train[val_cols].isna())
            scores.append(mae)
            print(f"  repeat {i+1}/{N_REPEATS}: {mae:8.2f} W | R0_lightgbm = "
                  f"{baseline[i]:8.2f} W | TabPFN I4 = {tabpfn[i]:7.2f} W  "
                  f"({time.time()-t0:.1f}s)", flush=True)
        s = np.array(scores)
        print(f"\n  LSTM {variant:20s} : {s.mean():.2f} +/- {s.std():.2f} W")
        print(f"  R0_lightgbm            : {np.mean(baseline):.2f} W")
        print(f"  TabPFN I4              : {np.mean(tabpfn):.2f} W\n", flush=True)
        results[f"{name}_{variant}"] = s.tolist()
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
