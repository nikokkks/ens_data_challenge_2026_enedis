#!/usr/bin/env python3
"""TabPFN V14 sur le même split "validation finale" que colab_retrieval_perceiver_t1.py,
plutôt que le protocole 10×10 (10 colonnes, très bruité — cf. le repeat 1 à
110,87 W très au-dessus de la moyenne historique ~63 W). Donne un chiffre
bien plus stable, directement comparable au "Meilleure MAE holdout" que le
script principal affiche après `train_imputer`.

Reproduit exactement le split de `main()` dans colab_retrieval_perceiver_t1.py :
    all_ids = np.arange(len(y_train.columns))            # 1000
    fit_ids, val_ids = train_test_split(all_ids, test_size=cfg.val_fraction,
                                        random_state=cfg.seed)
    # cfg.val_fraction=0.20, cfg.seed=42 -> fit=800, val=200

`fit_ids` (800 colonnes) entrent dans le pool de donneurs TabPFN comme
`dev_cols`, exactement le rôle qu'elles jouent pour le modèle Retrieval (vues
en vérité terrain pendant son entraînement supervisé) ; `val_ids` (200
colonnes) sont notées, jamais dans le pool. Même agrégation de la MAE
(moyenne poolée point par point, pas moyenne des moyennes par colonne) que
`official_mae` (evaluate.py) et `evaluate_loader` (colab_retrieval_perceiver_t1.py)
— comparaison directe légitime.

Reprend TEL QUEL le pipeline TabPFN V14 (V14_BEST, sélection de voisins,
clusters de forme, alpha de blend) déjà validé dans les autres scripts du
repo. Script indépendant : ne modifie ni n'importe aucun autre fichier.

Usage :
    export TABPFN_API_KEY=...
    .venv/bin/python claude2/tabpfn_v14_final_holdout_eval.py
    .venv/bin/python claude2/tabpfn_v14_final_holdout_eval.py --pool train_plus_test

⚠️ Coût : 200 colonnes notées = ~20x le protocole 10×10. Le clustering
K-means (une seule fois, sur le pool complet) prend en plus ~40s (train_only)
à ~150s (train_plus_test) mesurés en local. Compte plusieurs dizaines de
minutes pour le run complet, même parallélisé sur 20 workers. Reprenable :
les colonnes déjà notées sont sauvegardées au fur et à mesure (tous les 10
résultats) et ne sont pas relancées après une coupure.
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
# Pipeline TabPFN V14 exact (identique aux autres scripts V14 du repo, non
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
# Construction du pool (identique à tabpfn_v14_pool_comparison.py) + boucle
# d'évaluation avec checkpoint incrémental (200 colonnes, coupure possible).
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


def save_json_atomic(payload: Dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def evaluate_with_checkpoint(x_train: pd.DataFrame, y_train: pd.DataFrame, pool: np.ndarray,
                             km: KMeans, groups: Dict[int, np.ndarray], tfeat: np.ndarray,
                             val_cols: Sequence[str], results: Dict, out_path: Path,
                             max_workers: int, checkpoint_every: int = 10) -> None:
    done: Dict[str, List[float]] = results.setdefault("errors", {})
    remaining = [c for c in val_cols if c not in done]
    if not remaining:
        print("Toutes les colonnes sont déjà notées (cache rechargé).")
        return
    print(f"{len(done)} colonnes déjà notées (cache), {len(remaining)} restantes.")
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(predict_tabpfn_v14_column, x_train[c], pool, km, groups,
                                   tfeat, x_train.index, V14_BEST): c for c in remaining}
        for future in as_completed(futures):
            col = futures[future]
            pred, error = future.result()
            miss = x_train[col].isna().to_numpy()
            truth = y_train[col].to_numpy()[miss]
            valid = np.isfinite(truth)
            done[col] = np.abs(pred[valid] - truth[valid]).tolist()
            if error:
                results["fallbacks"] = results.get("fallbacks", 0) + 1
                print(f"  [TabPFN V14] {col}: repli après {error}", flush=True)
            completed += 1
            if completed % checkpoint_every == 0 or completed == len(remaining):
                save_json_atomic(results, out_path)
                pooled = np.concatenate([np.asarray(v) for v in done.values()])
                print(f"  [{completed}/{len(remaining)}] MAE cumulée sur colonnes notées="
                      f"{pooled.mean():.2f} W ({len(done)}/{len(val_cols)} au total)", flush=True)
    save_json_atomic(results, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pool", choices=("train_only", "train_plus_test"), default="train_only",
                        help="train_only reproduit le protocole historique de tous les benchmarks "
                             "TabPFN du repo (comparable à la référence ~63,24 W). train_plus_test "
                             "ajoute les donneurs de X_test, comme la banque transductive du modèle "
                             "Retrieval Perceiver.")
    parser.add_argument("--val-fraction", type=float, default=0.20,
                        help="cfg.val_fraction dans colab_retrieval_perceiver_t1.py")
    parser.add_argument("--seed", type=int, default=42, help="cfg.seed dans colab_retrieval_perceiver_t1.py")
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--output", default=None,
                        help="Défaut: claude2/tabpfn_v14_final_holdout_<pool>_results.json")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else Path(
        f"claude2/tabpfn_v14_final_holdout_{args.pool}_results.json")

    root = Path(args.data_dir)
    x_train, y_train, x_test = load_frames(root)
    configure_tabpfn()

    train_targets_cols = list(y_train.columns)
    all_ids = np.arange(len(train_targets_cols))
    fit_ids, val_ids = train_test_split(all_ids, test_size=args.val_fraction, random_state=args.seed)
    fit_cols = [train_targets_cols[i] for i in fit_ids]
    val_cols = [train_targets_cols[i] for i in val_ids]
    print(f"Split identique à colab_retrieval_perceiver_t1.py (val_fraction={args.val_fraction}, "
          f"seed={args.seed}) : fit={len(fit_cols)} colonnes (pool), val={len(val_cols)} colonnes "
          f"(notées), pool={args.pool}")

    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    if results and results.get("val_cols") != val_cols:
        raise RuntimeError(f"{out_path} contient des résultats calculés avec un autre split "
                           f"(val_fraction/seed différents) — supprime le fichier ou change --output")
    results.setdefault("val_cols", val_cols)
    results.setdefault("pool", args.pool)

    tfeat = time_features_v14(x_train.index)

    print(f"\nConstruction du pool de donneurs (pool={args.pool})…")
    started = time.time()
    include_test = args.pool == "train_plus_test"
    pool, km, groups = build_pool_and_clusters(x_train, y_train, x_test, fit_cols, include_test)
    print(f"Pool prêt: {pool.shape[1]} donneurs, {len(groups)} clusters, {time.time() - started:.1f}s")

    print(f"\nÉvaluation TabPFN V14 sur {len(val_cols)} colonnes tenues à l'écart…")
    started = time.time()
    evaluate_with_checkpoint(x_train, y_train, pool, km, groups, tfeat, val_cols,
                             results, out_path, args.max_workers)
    results["seconds"] = results.get("seconds", 0.0) + (time.time() - started)
    results["pool_size"] = int(pool.shape[1])
    save_json_atomic(results, out_path)

    all_errors = np.concatenate([np.asarray(v) for v in results["errors"].values()])
    print(f"\n=== TabPFN V14 — validation finale style colab_retrieval_perceiver_t1.py "
          f"(pool={args.pool}) ===")
    print(f"MAE = {all_errors.mean():.3f} W  (n={len(all_errors)} points manquants sur "
          f"{len(val_cols)} colonnes, replis={results.get('fallbacks', 0)})")
    print(f"\nÀ comparer directement à \"Meilleure MAE holdout\" affiché par "
          f"colab_retrieval_perceiver_t1.py (train_imputer, même split "
          f"val_fraction={args.val_fraction} seed={args.seed}, même agrégation pointée-poolée).")
    print(f"Résultats : {out_path}")


if __name__ == "__main__":
    main()
