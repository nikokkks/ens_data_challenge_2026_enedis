"""Pipeline TabPFN reconstruit depuis les donnees brutes du challenge Enedis.

Le module est volontairement autonome : il n'importe aucun pipeline du depot.
Il permet une verification sans API, un benchmark rapide 3x10 et une
confirmation 10x10. La MAE est calculee uniquement aux positions masquees
pour lesquelles y_train fournit une verite.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SEED = 42
DAILY = 48
PERIODS = (48, 96, 144, 336)


@dataclass(frozen=True)
class Config:
    name: str = "zero_start_full"
    k_neighbors: int = 40
    use_periodic: bool = True
    use_neighbors: bool = True
    blend: bool = True


CONFIGS = {
    "calendar": Config("calendar", use_periodic=False, use_neighbors=False),
    "periodic": Config("periodic", use_neighbors=False),
    "full_no_blend": Config("full_no_blend", blend=False),
    "full": Config(),
}


def locate_data(data_dir: Path) -> tuple[Path, Path, Path]:
    def one(pattern: str) -> Path:
        matches = sorted(data_dir.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one {pattern!r} in {data_dir}, got {matches}")
        return matches[0]

    return one("X_train*.csv"), one("y_train*.csv"), one("X_test*.csv")


def load_train(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_path, y_path, _ = locate_data(data_dir)
    X = pd.read_csv(x_path, index_col=0, parse_dates=True)
    y = pd.read_csv(y_path, index_col=0, parse_dates=True)
    targets = [c for c in X.columns if c.startswith("holed_")]
    if not X.index.equals(y.index) or targets != list(y.columns):
        raise ValueError("X_train/y_train index or target columns do not align")
    if len(targets) != 1000:
        raise ValueError(f"Expected 1000 holed columns, got {len(targets)}")
    return X, y


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = np.nanmedian(values, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(values - location), axis=0)
    q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
    scale = np.maximum.reduce([mad, (q75 - q25) / 1.349, np.full_like(mad, 10.0)])
    scale[~np.isfinite(scale)] = 10.0
    return location, scale


def robust_z(values: np.ndarray, location: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((values - location) / scale, -20.0, 20.0)


def gap_geometry(missing: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = np.zeros(len(missing), dtype=np.float32)
    edge = np.zeros(len(missing), dtype=np.float32)
    position = np.zeros(len(missing), dtype=np.float32)
    changes = np.diff(np.r_[False, missing, False].astype(np.int8))
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        n = stop - start
        x = np.arange(n)
        length[start:stop] = n
        edge[start:stop] = np.minimum(x + 1, n - x)
        position[start:stop] = (x + 0.5) / n
    return length, edge, position


def shifted(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float32)
    if lag > 0:
        out[lag:] = values[:-lag]
    else:
        out[:lag] = values[-lag:]
    return out


def calendar_features(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    minute = index.hour.to_numpy() * 60 + index.minute.to_numpy()
    hour_angle = 2 * np.pi * minute / (24 * 60)
    dow = index.dayofweek.to_numpy()
    dow_angle = 2 * np.pi * dow / 7
    day_angle = 2 * np.pi * (index.day.to_numpy() - 1) / 31
    return {
        "hour_sin": np.sin(hour_angle), "hour_cos": np.cos(hour_angle),
        "dow_sin": np.sin(dow_angle), "dow_cos": np.cos(dow_angle),
        "day_sin": np.sin(day_angle), "day_cos": np.cos(day_angle),
        "weekend": (dow >= 5).astype(np.float32),
        "trend": np.linspace(-1.0, 1.0, len(index), dtype=np.float32),
    }


class DonorPool:
    """Robustly scaled donors and leak-free nearest-neighbour lookup."""

    def __init__(self, frame: pd.DataFrame):
        self.columns = list(frame.columns)
        self.raw = frame.to_numpy(dtype=np.float32)
        self.location, self.scale = robust_location_scale(self.raw)
        self.z = robust_z(self.raw, self.location, self.scale).astype(np.float32)

    def nearest(self, target: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        observed = np.isfinite(target)
        target_loc, target_scale = robust_location_scale(target[:, None])
        zt = robust_z(target, target_loc[0], target_scale[0]).astype(np.float32)
        zt_obs = zt[observed]
        donors = self.z[observed]
        zt_centered = zt_obs - zt_obs.mean()
        donor_centered = donors - np.nanmean(donors, axis=0)
        numerator = np.nansum(donor_centered * zt_centered[:, None], axis=0)
        denominator = np.sqrt(
            np.nansum(donor_centered**2, axis=0) * np.sum(zt_centered**2)
        )
        corr = np.divide(numerator, denominator, out=np.full_like(numerator, -np.inf), where=denominator > 1e-8)
        finite = np.isfinite(corr)
        candidates = np.flatnonzero(finite)
        if len(candidates) < k:
            # Pearson is undefined for a flat target. In that case rank donors
            # by robust shape distance on jointly observed timestamps.
            distance = np.nanmean((donors - zt_obs[:, None]) ** 2, axis=0)
            candidates = np.flatnonzero(np.isfinite(distance))
            if len(candidates) < k:
                raise ValueError(f"Only {len(candidates)} valid donors for k={k}")
            selected = candidates[np.argpartition(distance[candidates], k)[:k]]
            selected = selected[np.argsort(distance[selected])]
            return self.z[:, selected], np.zeros(k, dtype=np.float32)
        selected = candidates[np.argpartition(corr[candidates], -k)[-k:]]
        selected = selected[np.argsort(corr[selected])[::-1]]
        return self.z[:, selected], corr[selected]


def build_features(
    index: pd.DatetimeIndex,
    target: np.ndarray,
    pool: DonorPool,
    config: Config,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], float, float]:
    target_loc_a, target_scale_a = robust_location_scale(target[:, None])
    target_loc, target_scale = float(target_loc_a[0]), float(target_scale_a[0])
    target_z = robust_z(target, target_loc, target_scale).astype(np.float32)
    missing = ~np.isfinite(target)
    run_length, edge_distance, run_position = gap_geometry(missing)
    features = calendar_features(index)

    periodic_stack: list[np.ndarray] = []
    if config.use_periodic:
        for period in PERIODS:
            for direction in (-1, 1):
                lag = direction * period
                value = shifted(target_z, lag)
                features[f"self_lag_{lag}"] = value
                features[f"self_lag_{lag}_available"] = np.isfinite(value).astype(np.float32)
                periodic_stack.append(value)
        with np.errstate(all="ignore"):
            stack = np.stack(periodic_stack)
            features["periodic_median"] = np.nanmedian(stack, axis=0)
            features["periodic_mad"] = np.nanmedian(
                np.abs(stack - features["periodic_median"][None, :]), axis=0
            )
            features["periodic_count"] = np.isfinite(stack).sum(axis=0).astype(np.float32)

    if config.use_neighbors:
        neighbors, correlations = pool.nearest(target, config.k_neighbors)
        for rank in range(config.k_neighbors):
            features[f"neighbor_{rank + 1}"] = neighbors[:, rank]
        with np.errstate(all="ignore"):
            features["neighbor_median"] = np.nanmedian(neighbors, axis=1)
            features["neighbor_mad"] = np.nanmedian(
                np.abs(neighbors - features["neighbor_median"][:, None]), axis=1
            )
            features["neighbor_mean"] = np.nanmean(neighbors, axis=1)
        features["neighbor_available"] = np.isfinite(neighbors).sum(axis=1).astype(np.float32)
        features["best_neighbor_corr"] = np.full(len(index), correlations[0], dtype=np.float32)

    # Gap geometry is used by the blend only: for observed training rows it
    # would be identically zero and would create a train/query distribution shift.
    metadata = {
        "run_length": run_length,
        "edge_distance": edge_distance,
        "run_position": run_position,
        "target_z": target_z,
    }
    return pd.DataFrame(features, index=index, dtype=np.float32), metadata, target_loc, target_scale


def interpolation(target: np.ndarray) -> np.ndarray:
    return pd.Series(target).interpolate(limit_direction="both").to_numpy(dtype=np.float32)


def blend_predictions(
    model_prediction: np.ndarray,
    interpolation_prediction: np.ndarray,
    features: pd.DataFrame,
    metadata: dict[str, np.ndarray],
    missing: np.ndarray,
) -> np.ndarray:
    edge = metadata["edge_distance"][missing]
    run = metadata["run_length"][missing]
    # Interpolation dominates isolated points and gap boundaries; its weight
    # decays toward the centre of long gaps. Periodic agreement reduces it.
    alpha = 0.12 + 0.78 * np.exp(-np.maximum(edge - 1.0, 0.0) / 6.0)
    alpha = np.where(run <= 1, 0.92, alpha)
    if "periodic_count" in features:
        count = features.loc[missing, "periodic_count"].to_numpy()
        mad = features.loc[missing, "periodic_mad"].to_numpy()
        agreement = np.exp(-np.nan_to_num(mad, nan=10.0, posinf=10.0) / 1.5)
        alpha *= 1.0 - 0.35 * agreement * np.clip(count / 4.0, 0.0, 1.0)
    alpha = np.clip(alpha, 0.05, 0.95)
    return alpha * interpolation_prediction + (1.0 - alpha) * model_prediction


def predict_column(
    target: np.ndarray,
    index: pd.DatetimeIndex,
    pool: DonorPool,
    config: Config,
    random_state: int,
    dry_run: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    features, metadata, location, scale = build_features(index, target, pool, config)
    observed = np.isfinite(target)
    missing = ~observed
    if not observed.any() or not missing.any():
        raise ValueError("Target must contain both observed and missing values")
    if dry_run:
        model_z = features.loc[missing].get("neighbor_median", pd.Series(0.0, index=features.index[missing])).to_numpy()
    else:
        import tabpfn_client
        from tabpfn_client import TabPFNRegressor


        regressor = TabPFNRegressor(random_state=random_state,thinking_mode=True, thinking_metric="mean_absolute_error")
        regressor.fit(features.loc[observed], metadata["target_z"][observed])
        model_z = np.asarray(regressor.predict(features.loc[missing], output_type="median"), dtype=float)
    model_w = location + scale * model_z
    linear_w = interpolation(target)[missing]
    prediction = model_w
    if config.blend:
        prediction = blend_predictions(model_w, linear_w, features, metadata, missing)
    prediction = np.maximum(prediction, 0.0)
    diagnostics = {
        "n_observed": int(observed.sum()), "n_missing": int(missing.sum()),
        "n_features": int(features.shape[1]), "location": location, "scale": scale,
    }
    return prediction, diagnostics


def choose_columns(n_targets: int, repeats: int, n_columns: int, seed: int) -> list[np.ndarray]:
    from sklearn.model_selection import train_test_split

    result = []
    for repeat in range(repeats):
        all_columns = np.arange(n_targets)
        _, validation = train_test_split(
            all_columns, test_size=0.1, random_state=seed + repeat, shuffle=True
        )
        result.append(validation[:n_columns])
    return result


def evaluate(
    X: pd.DataFrame,
    y: pd.DataFrame,
    config: Config,
    repeats: int,
    n_columns: int,
    seed: int,
    dry_run: bool = False,
) -> dict:
    target_columns = list(y.columns)
    # Only the 20,000 complete/non-target columns are donors. Evaluated target
    # truths never enter the donor pool.
    donor_columns = [c for c in X.columns if not c.startswith("holed_")]
    pool = DonorPool(X[donor_columns])
    splits = choose_columns(len(target_columns), repeats, n_columns, seed)
    repeat_scores: list[float] = []
    details: list[dict] = []
    for repeat, columns in enumerate(splits):
        errors: list[np.ndarray] = []
        for ordinal, column_idx in enumerate(columns, start=1):
            name = target_columns[column_idx]
            target = X[name].to_numpy(dtype=float)
            truth = y[name].to_numpy(dtype=float)
            missing = ~np.isfinite(target)
            prediction, diagnostics = predict_column(
                target, X.index, pool, config, seed + repeat, dry_run=dry_run
            )
            score_mask = np.isfinite(truth[missing])
            errors.append(np.abs(prediction[score_mask] - truth[missing][score_mask]))
            print(
                f"repeat={repeat + 1}/{repeats} column={ordinal}/{n_columns} "
                f"name={name} features={diagnostics['n_features']}", flush=True
            )
        score = float(np.concatenate(errors).mean())
        repeat_scores.append(score)
        details.append({"repeat": repeat, "columns": columns.tolist(), "mae": score})
        print(f"repeat={repeat + 1} MAE={score:.6f} W", flush=True)
    return {
        "config": config.__dict__, "seed": seed, "repeats": repeats,
        "n_columns": n_columns, "scores": repeat_scores,
        "mean": float(np.mean(repeat_scores)), "std": float(np.std(repeat_scores)),
        "details": details, "dry_run": dry_run,
    }


def run_eda(X: pd.DataFrame, y: pd.DataFrame) -> dict:
    targets = list(y.columns)
    observed = X[targets]
    truth = y.to_numpy(dtype=float)
    values = observed.to_numpy(dtype=float)
    scorable = ~np.isfinite(values) & np.isfinite(truth)
    linear = observed.interpolate(limit_direction="both").to_numpy(dtype=float)
    lengths = []
    for column in range(values.shape[1]):
        run, _, _ = gap_geometry(~np.isfinite(values[:, column]))
        lengths.extend(run[run > 0].tolist())
    return {
        "shape_X": list(X.shape), "shape_y": list(y.shape),
        "start": str(X.index.min()), "end": str(X.index.max()),
        "missing": int((~np.isfinite(values)).sum()), "scorable": int(scorable.sum()),
        "linear_mae": float(np.abs(linear[scorable] - truth[scorable]).mean()),
        "missing_fraction": float((~np.isfinite(values)).mean()),
        "point_weighted_gap_quantiles": np.quantile(lengths, [0, .5, .9, .95, .99, 1]).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--mode", choices=("eda", "dry-run", "quick", "full"), default="dry-run")
    parser.add_argument("--config", choices=tuple(CONFIGS), default="full")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repeats", type=int, help="Override the mode repeat count")
    parser.add_argument("--columns", type=int, help="Override columns per repeat")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y = load_train(args.data_dir)
    if args.mode == "eda":
        result = run_eda(X, y)
    else:
        repeats, n_columns = ((1, 2) if args.mode == "dry-run" else (3, 10) if args.mode == "quick" else (10, 10))
        repeats = args.repeats or repeats
        n_columns = args.columns or n_columns
        if not os.environ.get("TABPFN_API_KEY") and args.mode != "dry-run":
            raise RuntimeError("TABPFN_API_KEY is required for quick/full modes")
        result = evaluate(
            X, y, CONFIGS[args.config], repeats, n_columns, args.seed,
            dry_run=args.mode == "dry-run",
        )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    output = args.output or Path(f"tabpfn_zero_start_{args.mode}_{args.config}.json")
    output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
