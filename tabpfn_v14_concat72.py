"""TabPFN V14 enrichi par concaténation des 72 features zero-start.

Le modèle conserve la sélection de voisins, les features et le blend final de
V14. Les 72 features indépendantes de ``tabpfn_zero_start`` sont ajoutées au
design matrix juste avant le fit TabPFN. Le protocole par défaut est le 10x10
historique de V14 (seeds 42..51, 10 colonnes par repeat).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tabpfn_client
from sklearn.model_selection import train_test_split

import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14
import tabpfn_zero_start as zero
from evaluate import load_data


RANDOM_STATE = 42
MAX_WORKERS = 20
MAX_RETRIES = 3
DEFAULT_OUTPUT = Path("tabpfn_v14_concat72_results.json")
REFERENCE_RESULTS = Path("tabpfn_v14_confirm_results.json")
BEST_V14 = {
    "volatility_alpha": True,
    "vol_weight": 0.8,
    "periodic_alpha_strength": 1.0,
    "extra_lags": (24,),
    "neighbor_rolling": True,
}


def prepare_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    donor_cols: list[str],
    dev_cols: list[str],
    val_cols: list[str],
) -> tuple[np.ndarray, object, dict, np.ndarray, zero.DonorPool]:
    """Build all split-level state without exposing validation truths."""
    donor_df = X[donor_cols]
    dev_true = y[dev_cols]
    pool = pd.concat([donor_df, dev_true], axis=1).ffill().bfill()

    km, donor_labels = feb.fit_shape_clusters(
        donor_df, n_clusters=15, random_state=RANDOM_STATE
    )
    dev_filled = dev_true.ffill().bfill()
    dev_z = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = pd.Series(km.predict(dev_z.T.to_numpy()), index=dev_cols)
    pool_labels = pd.concat([donor_labels, dev_labels])
    positions = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
    cluster_positions = {
        cluster: positions[index].to_numpy()
        for cluster, index in pool_labels.groupby(pool_labels).groups.items()
    }
    time_features = feb.time_features_full_seasonality(
        X[val_cols].index, ("hour", "dow", "dom", "year")
    )
    # Same authorized pool as V14, but with the independent robust scaling and
    # global neighbour ranking used by the 72-feature pipeline.
    independent_pool = zero.DonorPool(pool)
    # Reuse the float32 raw matrix already owned by DonorPool instead of
    # retaining a second ~177 MB float64 copy for V14.
    pool_values = independent_pool.raw
    return pool_values, km, cluster_positions, time_features, independent_pool


def build_concat_features(
    column: str,
    series: pd.Series,
    pool_values: np.ndarray,
    km: object,
    cluster_positions: dict,
    time_features: np.ndarray,
    independent_pool: zero.DonorPool,
) -> tuple:
    built = v14.build_v14_features(
        column,
        series,
        pool_values,
        km,
        cluster_positions,
        time_features,
        series.index,
        BEST_V14,
    )
    if built is None:
        return None
    datetime_index = pd.DatetimeIndex(pd.to_datetime(series.index))
    zero_frame, zero_meta, _, _ = zero.build_features(
        datetime_index,
        series.to_numpy(dtype=float),
        independent_pool,
        zero.CONFIGS["full"],
    )
    zero_values = zero_frame.to_numpy(dtype=np.float32)
    if zero_values.shape != (len(series), 72):
        raise AssertionError(f"Expected 72 zero-start features, got {zero_values.shape}")
    return (*built, zero_values, zero_meta)


def predict_column(
    column: str,
    series: pd.Series,
    pool_values: np.ndarray,
    km: object,
    cluster_positions: dict,
    time_features: np.ndarray,
    independent_pool: zero.DonorPool,
    dry_run: bool = False,
) -> tuple[str, np.ndarray | None, str | None, dict]:
    built = build_concat_features(
        column,
        series,
        pool_values,
        km,
        cluster_positions,
        time_features,
        independent_pool,
    )
    if built is None:
        return column, None, None, {"n_features": 0}

    (
        obs_idx,
        miss_idx,
        target_obs,
        interpolation,
        knn_prediction,
        scaled_v14,
        extra_v14,
        missing,
        lags,
        zero_values,
        _,
    ) = built
    design = np.hstack([scaled_v14, extra_v14, zero_values])
    n_v14 = scaled_v14.shape[1] + extra_v14.shape[1]
    diagnostics = {
        "v14_features": int(n_v14),
        "zero_features": int(zero_values.shape[1]),
        "n_features": int(design.shape[1]),
        "n_observed": int(len(obs_idx)),
        "n_missing": int(len(miss_idx)),
    }
    # V14's own neighbour count is adaptive (5..40, corr_threshold=0.15), so
    # n_v14 -- and therefore the concatenated width -- varies per column; 147
    # is only the width observed on the smoke-test column, not a repo
    # invariant. Only the zero-start block (asserted at 72 in
    # build_concat_features) and the resulting hstack width are guaranteed.
    if zero_values.shape[1] != 72 or design.shape[1] != n_v14 + 72:
        raise AssertionError(
            f"Inconsistent concatenated design: v14={n_v14}, zero={zero_values.shape[1]}, "
            f"total={design.shape[1]}"
        )

    prediction = None
    error = None
    if dry_run:
        prediction = knn_prediction
    elif len(obs_idx) >= 15:
        for attempt in range(MAX_RETRIES):
            try:
                regressor = v14.MedianTabPFN(random_state=RANDOM_STATE)
                regressor.fit(design[obs_idx], target_obs)
                prediction = regressor.predict(design[miss_idx])
                break
            except Exception as exc:  # network/API fallback mirrors V14
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(2 * (attempt + 1))
    if prediction is None:
        prediction = knn_prediction

    # Preserve the complete V14 final blend unchanged.
    runs = feb.run_lengths_for(missing)
    edges = feb.edge_distance_for(missing)
    alpha = np.array(
        [feb.continuous_alpha(runs[i], edges[i], True) for i in miss_idx]
    )
    periodic_agreement = feb.periodic_agreement(
        series.to_numpy(dtype=float), lags=lags
    )[miss_idx]
    alpha *= 1.0 - BEST_V14["periodic_alpha_strength"] * periodic_agreement
    volatility = feb.curve_volatility(series.to_numpy(dtype=float))
    alpha *= feb.volatility_alpha_mult(
        volatility, vol_weight=BEST_V14["vol_weight"], cap=0.6
    )
    alpha = np.clip(alpha, 0.0, 0.95)
    blended = (
        alpha * interpolation.to_numpy()[miss_idx]
        + (1.0 - alpha) * np.asarray(prediction)
    )
    return column, blended, error, diagnostics


def evaluate_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    donor_cols: list[str],
    dev_cols: list[str],
    val_cols: list[str],
    dry_run: bool = False,
) -> tuple[float, int, list[dict]]:
    state = prepare_split(X, y, donor_cols, dev_cols, val_cols)
    pool_values, km, cluster_positions, time_features, independent_pool = state
    errors: list[np.ndarray] = []
    diagnostics: list[dict] = []
    fallbacks = 0

    def work(column: str):
        return predict_column(
            column,
            X[column],
            pool_values,
            km,
            cluster_positions,
            time_features,
            independent_pool,
            dry_run=dry_run,
        )

    workers = 1 if dry_run else min(MAX_WORKERS, len(val_cols))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(work, column): column for column in val_cols}
        for future in as_completed(futures):
            column, prediction, error, info = future.result()
            missing = X[column].isna().to_numpy()
            truth = y[column].to_numpy(dtype=float)[missing]
            valid = np.isfinite(truth)
            errors.append(np.abs(prediction[valid] - truth[valid]))
            diagnostics.append({"column": column, **info, "fallback": bool(error)})
            fallbacks += int(bool(error))
    return float(np.concatenate(errors).mean()), fallbacks, diagnostics


def paired_statistics(scores: list[float], reference_path: Path) -> dict | None:
    if not reference_path.exists():
        return None
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_scores = np.asarray(reference.get("scores", []), dtype=float)
    current = np.asarray(scores, dtype=float)
    if len(reference_scores) < len(current):
        return None
    reference_scores = reference_scores[: len(current)]
    delta = current - reference_scores
    rng = np.random.default_rng(RANDOM_STATE)
    bootstrap = np.array(
        [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(100_000)]
    )
    return {
        "reference_scores": reference_scores.tolist(),
        "delta_scores": delta.tolist(),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "wins": int((delta < 0).sum()),
        "bootstrap_ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def save_progress(
    output: Path,
    scores: list[float],
    details: list[dict],
    repeats: int,
    columns: int,
    dry_run: bool,
) -> dict:
    result = {
        "model": "TabPFN V14 + 72 zero-start features",
        "v14_options": {**BEST_V14, "extra_lags": list(BEST_V14["extra_lags"])},
        "zero_features": 72,
        "v14_features": "adaptive (5-40 neighbors + fixed extras), see per-column diagnostics",
        "repeats": repeats,
        "columns_per_repeat": columns,
        "scores": scores,
        "mean": float(np.mean(scores)) if scores else None,
        "std": float(np.std(scores)) if scores else None,
        "details": details,
        "dry_run": dry_run,
        "comparison_v14": None if dry_run else paired_statistics(scores, REFERENCE_RESULTS),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run(args: argparse.Namespace) -> dict:
    X, y, X_test = load_data()
    # The benchmark never uses X_test; release its ~300 MB payload before
    # constructing the two neighbour representations.
    del X_test
    donor_cols = [c for c in X if not str(c).startswith("holed_")]
    target_cols = list(y.columns)
    scores: list[float] = []
    details: list[dict] = []

    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            previous.get("repeats") == args.repeats
            and previous.get("columns_per_repeat") == args.columns
            and previous.get("dry_run") == args.dry_run
        ):
            scores = list(previous.get("scores", []))
            details = list(previous.get("details", []))

    for repeat in range(len(scores), args.repeats):
        seed = RANDOM_STATE + repeat
        dev_cols, validation_pool = train_test_split(
            target_cols, test_size=0.1, random_state=seed
        )
        rng = np.random.RandomState(seed)
        val_cols = list(
            rng.choice(
                validation_pool,
                size=min(args.columns, len(validation_pool)),
                replace=False,
            )
        )
        started = time.time()
        score, fallbacks, feature_info = evaluate_split(
            X, y, donor_cols, dev_cols, val_cols, dry_run=args.dry_run
        )
        elapsed = time.time() - started
        scores.append(score)
        details.append(
            {
                "repeat": repeat + 1,
                "seed": seed,
                "columns": val_cols,
                "mae": score,
                "fallbacks": fallbacks,
                "seconds": elapsed,
                "feature_diagnostics": feature_info,
            }
        )
        result = save_progress(
            args.output, scores, details, args.repeats, args.columns, args.dry_run
        )
        print(
            f"repeat={repeat + 1}/{args.repeats} MAE={score:.6f} W "
            f"fallbacks={fallbacks} time={elapsed:.1f}s",
            flush=True,
        )
    return save_progress(
        args.output, scores, details, args.repeats, args.columns, args.dry_run
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        token = os.environ.get("TABPFN_API_KEY")
        if not token:
            raise SystemExit("TABPFN_API_KEY manquant")
        tabpfn_client.set_access_token(token)
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
