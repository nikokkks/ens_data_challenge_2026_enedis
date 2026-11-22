"""Confirmation détaillée 10x10 du meilleur pipeline TabPFN V14.

Même configuration, mêmes splits et mêmes graines que
``tabpfn_v14_confirm.py``. La seule différence est le reporting : les dix
MAE de colonnes sont affichées dans l'ordre après chaque repeat, puis les
deux agrégations (moyenne des colonnes et MAE globale point par point).

Définir ``V14_DETAILED_SMOKE=1`` pour vérifier le flux en 1 repeat x 1
colonne. Ce mode utilise toujours TabPFN et exige donc ``TABPFN_API_KEY``.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tabpfn_client
from sklearn.model_selection import train_test_split

from evaluate import load_data
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14


RANDOM_STATE = 42
SMOKE = os.environ.get("V14_DETAILED_SMOKE", "0") == "1"
N_REPEATS = 1 if SMOKE else 10
N_VAL_SUB = 1 if SMOKE else 10
RESULTS_PATH = Path("tabpfn_v14_confirm_detailed_results.json")

BEST = dict(
    volatility_alpha=True,
    vol_weight=0.8,
    periodic_alpha_strength=1.0,
    extra_lags=(24,),
    neighbor_rolling=True,
)


def atomic_json_dump(value, path=RESULTS_PATH):
    """Écrit le checkpoint sans risquer de laisser un JSON tronqué."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def select_validation_columns(holed_cols, repeat_index, n_val_sub=N_VAL_SUB):
    """Reproduit exactement la sélection de ``v14.run_config``."""
    seed = RANDOM_STATE + repeat_index
    dev_cols, val_cols_full = train_test_split(
        holed_cols, test_size=.1, random_state=seed)
    rng = np.random.RandomState(seed)
    val_cols = list(rng.choice(
        val_cols_full, size=min(n_val_sub, len(val_cols_full)), replace=False))
    return seed, dev_cols, val_cols


def evaluate_repeat(X, y, donor_cols, dev_cols, val_cols, opts):
    """Évalue en parallèle, mais restitue les résultats dans ``val_cols``."""
    donor_df = X[donor_cols]
    dev_true_df = y[dev_cols]
    val_holed_df = X[val_cols]

    pool = pd.concat([donor_df, dev_true_df], axis=1)
    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    km, donor_labels = feb.fit_shape_clusters(
        donor_df, n_clusters=opts.get("n_clusters", 15), random_state=RANDOM_STATE)
    dev_filled = dev_true_df.ffill().bfill()
    normalized_dev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = pd.Series(km.predict(normalized_dev.T.values), index=dev_true_df.columns)
    pool_labels = pd.concat([donor_labels, dev_labels])
    positions = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
    cluster_col_idx = {
        cluster: positions[index].values
        for cluster, index in pool_labels.groupby(pool_labels).groups.items()
    }
    time_features = feb.time_features_full_seasonality(
        val_holed_df.index, ("hour", "dow", "dom", "year"))

    by_column = {}
    with ThreadPoolExecutor(max_workers=v14.MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                v14.complete_column_v14,
                column,
                val_holed_df[column],
                y[column],
                pool_vals,
                km,
                cluster_col_idx,
                time_features,
                val_holed_df.index,
                opts,
            ): column
            for column in val_cols
        }
        for future in as_completed(futures):
            column, abs_err, error = future.result()
            by_column[column] = {
                "abs_err": abs_err,
                "error": error,
            }

    ordered = []
    for column in val_cols:
        if column not in by_column:
            raise RuntimeError(f"Résultat manquant pour {column}")
        abs_err = by_column[column]["abs_err"]
        error = by_column[column]["error"]
        if abs_err is None or len(abs_err) == 0:
            raise RuntimeError(f"Aucune erreur calculée pour {column}")
        ordered.append({
            "column": str(column),
            "mae": float(np.mean(abs_err)),
            "n_missing": int(len(abs_err)),
            "fallback": bool(error),
            "error": error,
            "abs_err": abs_err,
        })

    column_mean = float(np.mean([item["mae"] for item in ordered]))
    weighted_mae = float(np.mean(np.concatenate([item["abs_err"] for item in ordered])))
    return ordered, column_mean, weighted_mae


def serializable_repeat(repeat_number, seed, elapsed, ordered,
                        column_mean, weighted_mae):
    return {
        "repeat": repeat_number,
        "seed": seed,
        "time_s": elapsed,
        "columns": [
            {
                "position": position,
                "column": item["column"],
                "mae": item["mae"],
                "n_missing": item["n_missing"],
                "fallback": item["fallback"],
                "error": item["error"],
            }
            for position, item in enumerate(ordered, 1)
        ],
        "column_mae_mean": column_mean,
        "weighted_mae": weighted_mae,
        "fallbacks": sum(item["fallback"] for item in ordered),
    }


def print_repeat(result):
    print(f"\n=== REPEAT {result['repeat']}/{N_REPEATS} "
          f"(seed={result['seed']}) ===")
    for item in result["columns"]:
        suffix = f" | FALLBACK: {item['error']}" if item["fallback"] else ""
        print(f"  {item['position']:2d}/{len(result['columns'])} "
              f"{item['column']:>12s} : MAE={item['mae']:9.2f} W "
              f"(n={item['n_missing']}){suffix}")
    print(f"  -> moyenne des {len(result['columns'])} MAE colonnes : "
          f"{result['column_mae_mean']:.2f} W")
    print(f"  -> MAE globale pondérée (officielle V14) : "
          f"{result['weighted_mae']:.2f} W")
    print(f"  -> temps={result['time_s']:.1f}s, fallbacks={result['fallbacks']}", flush=True)


def print_final_summary(repeats):
    column_means = np.asarray([item["column_mae_mean"] for item in repeats])
    weighted_maes = np.asarray([item["weighted_mae"] for item in repeats])
    print("\n" + "=" * 66)
    print("RÉCAPITULATIF DES REPEATS")
    for result in repeats:
        print(f"  repeat {result['repeat']:2d}/{N_REPEATS} : moyenne colonnes="
              f"{result['column_mae_mean']:9.2f} W | MAE pondérée="
              f"{result['weighted_mae']:9.2f} W")
    print("-" * 66)
    print(f"MAE finale — moyenne des moyennes de colonnes : "
          f"{column_means.mean():.2f} +/- {column_means.std():.2f} W")
    print(f"MAE finale — moyenne des MAE pondérées V14  : "
          f"{weighted_maes.mean():.2f} +/- {weighted_maes.std():.2f} W")
    print("=" * 66, flush=True)
    return column_means, weighted_maes


def main():
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant")
    tabpfn_client.set_access_token(token)

    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [column for column in X if not str(column).startswith("holed_")]
    holed_cols = list(y.columns)
    print(f"=== CONFIRM DÉTAILLÉ v14_best {N_REPEATS}x{N_VAL_SUB}: {BEST} ===", flush=True)

    repeats = []
    for repeat_index in range(N_REPEATS):
        seed, dev_cols, val_cols = select_validation_columns(holed_cols, repeat_index)
        print(f"\n[repeat {repeat_index + 1}/{N_REPEATS}] calcul parallèle de "
              f"{len(val_cols)} colonnes...", flush=True)
        started = time.time()
        ordered, column_mean, weighted_mae = evaluate_repeat(
            X, y, donors, dev_cols, val_cols, BEST)
        result = serializable_repeat(
            repeat_index + 1, seed, time.time() - started,
            ordered, column_mean, weighted_mae)
        repeats.append(result)
        print_repeat(result)

        # Checkpoint après chaque repeat terminé.
        atomic_json_dump({
            "status": "running",
            "config": BEST,
            "n_repeats": N_REPEATS,
            "n_val_sub": N_VAL_SUB,
            "repeats": repeats,
        })

    column_means, weighted_maes = print_final_summary(repeats)
    final = {
        "status": "complete",
        "config": BEST,
        "n_repeats": N_REPEATS,
        "n_val_sub": N_VAL_SUB,
        "repeats": repeats,
        "column_mae_means": column_means.tolist(),
        "weighted_maes": weighted_maes.tolist(),
        "final_column_mean": float(column_means.mean()),
        "final_column_std": float(column_means.std()),
        "final_weighted_mean": float(weighted_maes.mean()),
        "final_weighted_std": float(weighted_maes.std()),
    }
    atomic_json_dump(final)
    print(f"Résultats détaillés écrits dans {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
