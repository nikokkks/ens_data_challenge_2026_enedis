#!/usr/bin/env python3
"""Compare TabPFN V14 avec pool de donneurs restreint (X_train seul) vs élargi
(X_train + X_test), sur les mêmes repeats/colonnes que `run_paired_benchmark`
dans `colab_retrieval_perceiver_t1.py`.

Contexte : le benchmark 10×10 de `colab_retrieval_perceiver_t1.py` donne à
TabPFN V14 un pool de donneurs restreint à X_train (~20 900 courbes, via
`eval_tabpfn_v14_repeat`), alors que le modèle Retrieval Perceiver a accès
tout du long à sa banque transductive X_train+X_test (~57 140 courbes). Ce
script teste si donner à TabPFN V14 ce même pool élargi change sa MAE — pour
savoir si l'écart observé (Retrieval largement derrière TabPFN même avec un
pool 2,7x plus petit pour ce dernier) tient toujours une fois les deux pools
alignés.

Reprend TEL QUEL le pipeline TabPFN V14 (V14_BEST, sélection de voisins,
clusters de forme, alpha de blend) de `colab_retrieval_perceiver_t1.py` —
seule la construction du pool de donneurs change entre les deux conditions
comparées. Script indépendant : ne modifie ni n'importe le script principal.

Usage :
    export TABPFN_API_KEY=...
    .venv/bin/python claude2/tabpfn_v14_pool_comparison.py --repeats 1
    .venv/bin/python claude2/tabpfn_v14_pool_comparison.py --repeats 2 --columns 10

Reprenable : les erreurs par (repeat, condition de pool) déjà calculées sont
mises en cache dans le JSON de sortie et ne sont pas relancées (mêmes appels
TabPFN payants évités en cas de relance après coupure).
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

FILES = {
    "x_train": "X_train_78VdSWL.csv",
    "y_train": "y_train_u0UkKEh.csv",
    "x_test": "X_test_XKVc4no.csv",
}


def complete_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not str(c).startswith("holed_")]


def _interactive_colab_kernel_available() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and getattr(shell, "kernel", None) is not None
    except Exception:
        return False


def load_frames(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Chargement des CSV…")
    x_train = pd.read_csv(root / FILES["x_train"], index_col=0)
    y_train = pd.read_csv(root / FILES["y_train"], index_col=0)
    x_test = pd.read_csv(root / FILES["x_test"], index_col=0)
    return x_train, y_train, x_test


# %% ------------------------------------------------------------------------
# Pipeline TabPFN V14 exact (copié de colab_retrieval_perceiver_t1.py, non
# modifié — seule la composition du `pool` passé en entrée change).
# -----------------------------------------------------------------------

V14_BEST = dict(
    k=40, k_min=5, corr_threshold=0.15, euclid_weight=0.3,
    ratio_clip=(0.1, 10.0), extra_lags=(24,), periodic_alpha_strength=1.0,
    vol_weight=0.8, vol_cap=0.6, alpha_scale=1.0,
)
_TABPFN_REGRESSOR = None


def run_lengths_for(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(len(mask), dtype=np.int32)
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        out[i:j] = j - i
        i = j
    return out


def edge_distance_for(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(len(mask), dtype=np.float32)
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        p = np.arange(j - i)
        out[i:j] = np.minimum(p, j - i - 1 - p)
        i = j
    return out


def continuous_alpha_v14(run_len: int, edge_dist: float, edge_tau: float = 1.5) -> float:
    if run_len <= 2:
        base = 0.8
    elif run_len <= 6:
        base = 0.4
    elif run_len <= 20:
        base = 0.15
    else:
        base = 0.0
    return float(np.clip(base + (1 - base) * np.exp(-edge_dist / edge_tau), 0, 0.95))


def curve_volatility_v14(values: np.ndarray) -> float:
    obs = values[np.isfinite(values)]
    if len(obs) < 3:
        return 0.0
    return float(np.std(np.diff(obs)) / (np.mean(np.abs(obs)) + 1e-9))


def time_features_v14(index: pd.Index) -> np.ndarray:
    dt = pd.DatetimeIndex(pd.to_datetime(index))
    minute = dt.hour.to_numpy() * 60.0 + dt.minute.to_numpy()
    dow = dt.dayofweek.to_numpy(float)
    dom = dt.day.to_numpy(float) - 1.0
    doy = dt.dayofyear.to_numpy(float) - 1.0 + minute / 1440.0
    def cyc(v, period):
        angle = 2 * np.pi * v / period
        return np.column_stack([np.sin(angle), np.cos(angle)])
    return np.column_stack([cyc(minute, 1440), cyc(dow, 7), cyc(dom, 31),
                            cyc(doy, 365.2425), (dow >= 5).astype(float)])


def periodic_copy_features_v14(values: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    values = np.asarray(values, float)
    blocks = []
    for lag in lags:
        back, forward = np.full(len(values), np.nan), np.full(len(values), np.nan)
        back[lag:], forward[:-lag] = values[:-lag], values[lag:]
        blocks.extend([back, forward])
    raw = np.column_stack(blocks)
    available = np.isfinite(raw).astype(float)
    obs = values[np.isfinite(values)]
    center = float(np.median(obs)) if len(obs) else 0.0
    scale = max(1.4826 * np.median(np.abs(obs - center)), 0.02 * (abs(center) + 1), 1.0)
    normalized = (raw - center) / scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(normalized, axis=1)
        mad = np.nanmedian(np.abs(normalized - median[:, None]), axis=1)
    count = np.isfinite(normalized).sum(axis=1).astype(float)
    return np.column_stack([normalized, median, mad, count, available])


def periodic_agreement_v14(values: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    raw = []
    for lag in lags:
        back, forward = np.full(len(values), np.nan), np.full(len(values), np.nan)
        back[lag:], forward[:-lag] = values[:-lag], values[lag:]
        raw.extend([back, forward])
    raw = np.column_stack(raw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(raw, axis=1)
        mad = np.nanmedian(np.abs(raw - med[:, None]), axis=1)
    obs = values[np.isfinite(values)]
    scale = max(1.4826 * np.median(np.abs(obs - np.median(obs))), 1.0)
    return np.clip(np.isfinite(raw).mean(axis=1) * np.exp(-np.nan_to_num(mad, nan=np.inf) / scale), 0, 1)


def same_curve_seasonality_v14(values: np.ndarray, index: pd.Index) -> np.ndarray:
    dt = pd.to_datetime(index)
    bucket = dt.hour.to_numpy() * 60 + dt.minute.to_numpy()
    observed = np.isfinite(values)
    frame = pd.DataFrame({"bucket": bucket[observed], "value": values[observed]})
    med = frame.groupby("bucket")["value"].median()
    fallback = float(np.nanmedian(values)) if observed.any() else 0.0
    return pd.Series(bucket).map(med).fillna(fallback).to_numpy()[:, None]


def fit_shape_clusters_v14(df: pd.DataFrame) -> Tuple[KMeans, pd.Series]:
    filled = df.ffill().bfill()
    z = (filled - filled.mean()) / (filled.std() + 1e-9)
    km = KMeans(n_clusters=15, n_init=4, max_iter=100, random_state=42)
    labels = km.fit_predict(z.T.to_numpy())
    return km, pd.Series(labels, index=df.columns)


def adaptive_neighbors_v14(target: np.ndarray, pool: np.ndarray, opts: Dict) -> Tuple[np.ndarray, np.ndarray]:
    tc = target - target.mean()
    dc = pool - pool.mean(axis=0, keepdims=True)
    corr = (dc.T @ tc) / (np.sqrt((dc * dc).sum(0) * (tc * tc).sum()) + 1e-9)
    euclid = np.sqrt(((dc - tc[:, None]) ** 2).mean(0)) / (target.std() + 1e-9)
    combined = (1 - opts["euclid_weight"]) * corr + opts["euclid_weight"] / (1 + euclid)
    order = np.argsort(-combined)
    above = order[corr[order] >= opts["corr_threshold"]]
    chosen = above[:opts["k"]] if len(above) >= opts["k_min"] else order[:opts["k_min"]]
    return chosen, corr[chosen]


def scaled_neighbors_v14(pool: np.ndarray, obs: np.ndarray, target: np.ndarray,
                         opts: Dict) -> Tuple[np.ndarray, np.ndarray]:
    chosen, corr = adaptive_neighbors_v14(target, pool[obs], opts)
    selected = pool[:, chosen]
    selected_obs = selected[obs]
    center = np.median(selected_obs, axis=0)
    spread = np.maximum(1.4826 * np.median(np.abs(selected_obs - center), axis=0),
                        0.02 * (np.abs(center) + 1))
    tc = np.median(target)
    ts = max(1.4826 * np.median(np.abs(target - tc)), 0.02 * (abs(tc) + 1))
    ratio = np.clip(ts / spread, *opts["ratio_clip"])
    return (selected - center) * ratio + tc, corr


def predict_tabpfn_v14_column(s: pd.Series, pool: np.ndarray, km: KMeans,
                              cluster_indices: Dict[int, np.ndarray], tfeat: np.ndarray,
                              index: pd.Index, opts: Dict) -> Tuple[np.ndarray, Optional[str]]:
    missing = s.isna().to_numpy()
    obs, miss = np.flatnonzero(~missing), np.flatnonzero(missing)
    values = s.to_numpy(float)
    interpolated = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    z = (interpolated - interpolated.mean()) / (interpolated.std(ddof=1) + 1e-9)
    cluster = int(km.predict(z.reshape(1, -1))[0])
    same = cluster_indices.get(cluster)
    candidate = pool[:, same] if same is not None and len(same) >= 80 else pool
    scaled, corr = scaled_neighbors_v14(candidate, obs, values[obs], opts)
    weights = np.clip(corr, 1e-3, None); weights /= weights.sum()
    fallback = scaled[miss] @ weights
    lags = (48, 96, 144, 336) + tuple(opts["extra_lags"])
    periodic = periodic_copy_features_v14(values, lags)
    seasonal = same_curve_seasonality_v14(values, index)
    consensus = pd.Series(scaled.mean(axis=1))
    rolling = np.column_stack([consensus.rolling(5, center=True, min_periods=1).mean(),
                               consensus.rolling(5, center=True, min_periods=1).std().fillna(0)])
    extra = np.hstack([tfeat, seasonal, periodic, rolling])
    pred, error = None, None
    if len(obs) >= 15:
        for attempt in range(3):
            try:
                reg = _TABPFN_REGRESSOR(random_state=42)
                reg.fit(np.hstack([scaled[obs], extra[obs]]), values[obs])
                pred = reg.predict(np.hstack([scaled[miss], extra[miss]]), output_type="median")
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(2 * (attempt + 1))
    if pred is None:
        pred = fallback
    runs, edges = run_lengths_for(missing), edge_distance_for(missing)
    alpha = np.array([continuous_alpha_v14(runs[i], edges[i]) for i in miss])
    alpha *= 1 - opts["periodic_alpha_strength"] * periodic_agreement_v14(values, lags)[miss]
    vol = min(curve_volatility_v14(values), opts["vol_cap"]) / opts["vol_cap"]
    alpha *= 1 - opts["vol_weight"] * vol
    alpha = np.clip(alpha * opts["alpha_scale"], 0, 0.95)
    return alpha * interpolated[miss] + (1 - alpha) * pred, error


def configure_tabpfn() -> None:
    global _TABPFN_REGRESSOR
    if _TABPFN_REGRESSOR is not None:
        return
    try:
        import tabpfn_client
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tabpfn-client"])
        import tabpfn_client
    key = os.environ.get("TABPFN_API_KEY")
    if not key:
        try:
            from google.colab import userdata  # type: ignore
            key = userdata.get("TABPFN_API_KEY")
        except Exception:
            key = None
    if not key and _interactive_colab_kernel_available():
        key = getpass.getpass("TABPFN_API_KEY (saisie masquée) : ").strip()
    if not key:
        raise SystemExit("TABPFN_API_KEY absent : exporte-le dans l'environnement avant de lancer ce script.")
    tabpfn_client.set_access_token(key)
    from tabpfn_client import TabPFNRegressor
    _TABPFN_REGRESSOR = TabPFNRegressor


# %% ------------------------------------------------------------------------
# Ce qui change réellement entre les deux conditions comparées : la
# composition du pool de donneurs (et donc des clusters de forme fittés
# dessus). Tout le reste (sélection de voisins, features, alpha) est
# identique et hérité tel quel du bloc ci-dessus.
# -----------------------------------------------------------------------

def build_pool_and_clusters(x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame,
                            dev_cols: Sequence[str], include_test: bool
                            ) -> Tuple[np.ndarray, KMeans, Dict[int, np.ndarray]]:
    donor_frames = [x_train[complete_columns(x_train)]]
    if include_test:
        donor_frames.append(x_test[complete_columns(x_test)])
    donor_df = pd.concat(donor_frames, axis=1)
    dev_df = y_train[list(dev_cols)]
    pool_df = pd.concat([donor_df, dev_df], axis=1)
    pool = pool_df.ffill().bfill().to_numpy(float)
    km, donor_labels = fit_shape_clusters_v14(donor_df)
    dev_filled = dev_df.ffill().bfill()
    zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = pd.Series(km.predict(zdev.T.to_numpy()), index=dev_df.columns)
    labels = pd.concat([donor_labels, dev_labels])
    positions = pd.Series(np.arange(len(labels)), index=labels.index)
    groups = {cl: positions[idx].to_numpy() for cl, idx in labels.groupby(labels).groups.items()}
    return pool, km, groups


def eval_tabpfn_v14_on_pool(x_train: pd.DataFrame, y_train: pd.DataFrame, pool: np.ndarray,
                            km: KMeans, groups: Dict[int, np.ndarray], tfeat: np.ndarray,
                            val_cols: Sequence[str]) -> Tuple[Dict[str, np.ndarray], int]:
    errors: Dict[str, np.ndarray] = {}
    fallbacks = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(predict_tabpfn_v14_column, x_train[c], pool, km, groups,
                                   tfeat, x_train.index, V14_BEST): c for c in val_cols}
        for future in as_completed(futures):
            col = futures[future]
            pred, error = future.result()
            miss = x_train[col].isna().to_numpy()
            truth = y_train[col].to_numpy()[miss]
            valid = np.isfinite(truth)
            errors[col] = np.abs(pred[valid] - truth[valid])
            if error:
                fallbacks += 1
                print(f"  [TabPFN V14] {col}: repli après {error}", flush=True)
    return errors, fallbacks


def save_json_atomic(payload: Dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


POOL_CONDITIONS = (("train_only", False), ("train_plus_test", True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Nombre de repeats (seeds 42, 43, ...) — même convention que "
                             "cfg.benchmark_repeats/run_paired_benchmark")
    parser.add_argument("--columns", type=int, default=10,
                        help="Colonnes notées par repeat — même convention que cfg.benchmark_columns")
    parser.add_argument("--output", default="claude2/tabpfn_v14_pool_comparison_results.json")
    args = parser.parse_args()

    root = Path(args.data_dir)
    x_train, y_train, x_test = load_frames(root)
    configure_tabpfn()

    columns = list(y_train.columns)
    tfeat = time_features_v14(x_train.index)

    out_path = Path(args.output)
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"repeats": {}}

    for i in range(args.repeats):
        seed = 42 + i
        # Split identique à run_paired_benchmark : dev_cols/val_cols ne
        # servent ici qu'à sélectionner les mêmes colonnes notées ; dev_cols
        # entre aussi dans le pool commun aux deux conditions (comme dans
        # eval_tabpfn_v14_repeat), donc ce n'est pas ce qui varie ici.
        dev_cols, val_full = train_test_split(columns, test_size=0.1, random_state=seed)
        val_cols = list(np.random.RandomState(seed).choice(
            val_full, size=min(args.columns, len(val_full)), replace=False))
        key = str(i)
        rec = results["repeats"].setdefault(key, {"seed": seed, "val_cols": val_cols})
        if rec["val_cols"] != val_cols:
            raise RuntimeError(f"Colonnes incohérentes au repeat {i + 1} : "
                               f"résultats existants calculés avec un autre split")

        for pool_name, include_test in POOL_CONDITIONS:
            field = f"{pool_name}_errors"
            if field in rec:
                continue
            print(f"\n[repeat {i + 1}/{args.repeats}] pool={pool_name} "
                  f"(include_test={include_test}) — seed={seed}", flush=True)
            started = time.time()
            pool, km, groups = build_pool_and_clusters(x_train, y_train, x_test, dev_cols, include_test)
            errors, fallbacks = eval_tabpfn_v14_on_pool(x_train, y_train, pool, km, groups, tfeat, val_cols)
            rec[field] = {c: e.tolist() for c, e in errors.items()}
            rec[f"{pool_name}_fallbacks"] = fallbacks
            rec[f"{pool_name}_seconds"] = time.time() - started
            rec[f"{pool_name}_pool_size"] = int(pool.shape[1])
            save_json_atomic(results, out_path)
            mae = np.concatenate([np.asarray(v) for v in errors.values()]).mean()
            print(f"[repeat {i + 1}/{args.repeats}] pool={pool_name}: MAE={mae:.2f} W "
                  f"(pool={pool.shape[1]} donneurs), temps={rec[f'{pool_name}_seconds']:.1f}s, "
                  f"replis={fallbacks}", flush=True)

    print("\n| Repeat | Seed | MAE train_only | MAE train+test | Δ | Gain |")
    print("|---:|---:|---:|---:|---:|---:|")
    all_train_only, all_train_test = [], []
    for i in range(args.repeats):
        rec = results["repeats"][str(i)]
        cols = rec["val_cols"]
        e1 = np.concatenate([np.asarray(rec["train_only_errors"][c]) for c in cols])
        e2 = np.concatenate([np.asarray(rec["train_plus_test_errors"][c]) for c in cols])
        all_train_only.append(e1)
        all_train_test.append(e2)
        delta = e2.mean() - e1.mean()
        gain = 100 * (1 - e2.mean() / e1.mean())
        print(f"| {i + 1} | {rec['seed']} | {e1.mean():.2f} | {e2.mean():.2f} | "
              f"{delta:+.2f} | {gain:+.2f}% |")
    pooled1, pooled2 = np.concatenate(all_train_only), np.concatenate(all_train_test)
    delta_pooled = pooled2.mean() - pooled1.mean()
    gain_pooled = 100 * (1 - pooled2.mean() / pooled1.mean())
    print(f"\nMAE poolée train_only    = {pooled1.mean():.2f} W")
    print(f"MAE poolée train+test    = {pooled2.mean():.2f} W")
    print(f"Δ poolé                  = {delta_pooled:+.2f} W ({gain_pooled:+.2f}%)")
    print(f"\nRésultats détaillés : {out_path}")


if __name__ == "__main__":
    main()
