"""Fast, mask-aware benchmarks for the Enedis challenge.

The existing local models are fitted on observed timestamps and are then
asked to extrapolate inside gaps.  This benchmark instead learns globally
from the *real* gaps of the development curves.  Optional augmentation puts
the same real masks on complete donor curves.

All configurations use the exact TabPFN protocol: 10 repeated 90/10 column
splits and the same 10 validation columns in every repeat.
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from evaluate import load_data, official_mae, fill_nan_with_interpolation
from feature_engineering_bench import run_lengths_for, edge_distance_for

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_REPEATS = 10
N_VAL_SUB = 10
RESULTS_PATH = "masked_supervised_results.json"


def _nearest(values):
    n = len(values)
    obs = ~np.isnan(values)
    pv = np.full(n, np.nan); pdist = np.full(n, n, dtype=float)
    nv = np.full(n, np.nan); ndist = np.full(n, n, dtype=float)
    last_v = np.nan; last_i = -n
    for i in range(n):
        if obs[i]: last_v, last_i = values[i], i
        pv[i], pdist[i] = last_v, i - last_i
    next_v = np.nan; next_i = 2 * n
    for i in range(n - 1, -1, -1):
        if obs[i]: next_v, next_i = values[i], i
        nv[i], ndist[i] = next_v, next_i - i
    return pv, pdist, nv, ndist


def curve_features(s, index, periodic=True):
    """Features available both for a real target and a pseudo-masked donor."""
    v = s.to_numpy(dtype=float)
    miss = np.isnan(v)
    idx = np.flatnonzero(miss)
    if len(idx) == 0:
        return np.empty((0, 1)), idx, 0.0, 1.0

    obs = v[~miss]
    center = float(np.median(obs))
    q25, q75 = np.quantile(obs, [0.25, 0.75])
    scale = float(max(q75 - q25, 0.1 * (abs(center) + 1.0), 1.0))
    norm = lambda x: (x - center) / scale

    interp = fill_nan_with_interpolation(s).to_numpy(dtype=float)
    prev, dprev, nxt, dnxt = _nearest(v)
    runs = run_lengths_for(miss).astype(float)
    edges = edge_distance_for(miss)
    # Position from the left edge and relative position within the gap.
    left_pos = np.zeros(len(v), dtype=float)
    for i in idx:
        left_pos[i] = min(dprev[i] - 1, runs[i] - 1)
    rel_pos = left_pos / np.maximum(runs - 1, 1)

    dt = pd.to_datetime(index)
    hour = dt.hour.to_numpy() + dt.minute.to_numpy() / 60.0
    dow = dt.dayofweek.to_numpy()
    cols = [
        norm(interp), norm(prev), norm(nxt), dprev / len(v), dnxt / len(v),
        np.log1p(runs), edges / np.maximum(runs, 1), rel_pos,
        np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7), (dow >= 5).astype(float),
        np.full(len(v), np.log1p(scale)), np.full(len(v), center / scale),
    ]
    if periodic:
        # Exact daily/weekly copies. Missing source values remain NaN and are
        # handled natively by LightGBM; no target information is introduced.
        for lag in (48, 96, 144, 336):
            back = np.full(len(v), np.nan); forward = np.full(len(v), np.nan)
            back[lag:] = v[:-lag]; forward[:-lag] = v[lag:]
            cols.extend([norm(back), norm(forward)])
        bucket = dt.hour.to_numpy() * 60 + dt.minute.to_numpy()
        tmp = pd.DataFrame({"bucket": bucket[~miss], "value": obs})
        seasonal = pd.Series(bucket).map(tmp.groupby("bucket")["value"].median()).to_numpy()
        cols.append(norm(seasonal))
    return np.column_stack(cols)[idx], idx, center, scale


def build_rows(holed, truth, periodic=True):
    xs, ys = [], []
    for col in holed.columns:
        Xc, idx, center, scale = curve_features(holed[col], holed.index, periodic)
        good = ~np.isnan(truth[col].to_numpy(dtype=float)[idx])
        if good.any():
            xs.append(Xc[good])
            ys.append((truth[col].to_numpy(dtype=float)[idx][good] - center) / scale)
    return np.vstack(xs), np.concatenate(ys)


def augment_from_donors(donor_df, masks, n_donors, periodic, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(donor_df.shape[1], size=min(n_donors, donor_df.shape[1]), replace=False)
    xs, ys = [], []
    for j, donor_pos in enumerate(chosen):
        truth = donor_df.iloc[:, donor_pos]
        mask = masks.iloc[:, j % masks.shape[1]].to_numpy()
        pseudo = truth.mask(mask)
        Xc, idx, center, scale = curve_features(pseudo, donor_df.index, periodic)
        good = ~np.isnan(truth.to_numpy(dtype=float)[idx])
        if good.any():
            xs.append(Xc[good]); ys.append((truth.to_numpy()[idx][good] - center) / scale)
    return np.vstack(xs), np.concatenate(ys)


def predict_frame(reg, frame, periodic=True):
    out = frame.copy()
    for col in frame.columns:
        Xc, idx, center, scale = curve_features(frame[col], frame.index, periodic)
        out.iloc[idx, out.columns.get_loc(col)] = center + scale * reg.predict(Xc)
    return out


def evaluate_config(X, y, donor_cols, periodic=True, artificial_donors=0):
    cols = list(y.columns); scores = []
    for repeat in range(N_REPEATS):
        dev, val = train_test_split(cols, test_size=0.10, random_state=RANDOM_STATE + repeat)
        rng = np.random.default_rng(RANDOM_STATE + repeat)
        val = list(rng.choice(val, size=N_VAL_SUB, replace=False))
        Xtr, ytr = build_rows(X[dev], y[dev], periodic)
        if artificial_donors:
            Xa, ya = augment_from_donors(
                X[donor_cols], X[dev].isna(), artificial_donors, periodic,
                RANDOM_STATE + repeat)
            Xtr = np.vstack([Xtr, Xa]); ytr = np.concatenate([ytr, ya])
        reg = LGBMRegressor(
            objective="mae", n_estimators=350, learning_rate=0.05,
            num_leaves=31, max_depth=-1, min_child_samples=80,
            subsample=0.8, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=RANDOM_STATE + repeat, n_jobs=-1, verbosity=-1)
        t0 = time.time(); reg.fit(Xtr, ytr)
        pred = predict_frame(reg, X[val], periodic)
        score = official_mae(y[val], pred, X[val].isna())
        scores.append(score)
        print(f"  repeat {repeat+1:2d}/{N_REPEATS}: {score:8.3f} W "
              f"({len(ytr):,} train rows, {time.time()-t0:.1f}s)", flush=True)
    return np.asarray(scores)


CONFIGS = {
    "S0_global_real_gaps": dict(periodic=False, artificial_donors=0),
    "S1_plus_daily_weekly_copies": dict(periodic=True, artificial_donors=0),
    "S2_plus_1000_pseudo_masked_donors": dict(periodic=True, artificial_donors=1000),
}


if __name__ == "__main__":
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    X, y, _ = load_data()
    donors = [c for c in X.columns if not str(c).startswith("holed_")]
    try:
        results = json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        results = {}
    for name in ([only] if only else CONFIGS):
        print(f"=== {name} ===", flush=True)
        scores = evaluate_config(X, y, donors, **CONFIGS[name])
        results[name] = scores.tolist()
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"{name}: {scores.mean():.3f} +/- {scores.std():.3f} W", flush=True)
