#!/usr/bin/env python3
# %% [markdown]
# # Enedis — Retrieval-conditioned Perceiver + T1/TimeMixer imputer
#
# Script Colab autonome pour GPU 80 Go. Dans Colab :
# 1. Runtime > Change runtime type > GPU (A100/H100 recommandé).
# 2. Méthode recommandée avec `!python` : téléverser d'abord les CSV dans
#    une cellule Colab :
#      from google.colab import files
#      files.upload()
#    puis lancer :
#      !python /content/colab_retrieval_perceiver_t1.py --profile full --data-dir /content
# 3. Avec `%run /content/colab_retrieval_perceiver_t1.py --profile full`, le
#    script peut ouvrir lui-même la boîte de téléversement interactive.
#
# Le modèle n'utilise ni SAITS, ni PyPOTS, ni TabPFN. Il comporte :
# - un pré-entraînement auto-supervisé sur les 57 140 courbes complètes ;
# - un retrieval top-256 avec calibration affine sur les points observés ;
# - un Perceiver global auquel toutes les références participent ;
# - un décodeur temporel multi-échelle inspiré T1/TimeMixer ;
# - quatre experts et une prédiction résiduelle par rapport à PCHIP.
#
# Le profil `full` commence désormais par une comparaison stricte et
# reprenable 10 repeats × 10 colonnes contre TabPFN V14. Le modèle principal
# reste indépendant de TabPFN ; celui-ci n'est appelé que comme benchmark.
# Ajoute `TABPFN_API_KEY` aux Secrets Colab avant de lancer `full`.

# %%
from __future__ import annotations

import argparse
import copy
import gc
import getpass
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _bootstrap_colab() -> None:
    """Installe uniquement les petites dépendances absentes de Colab."""
    try:
        import torch  # noqa: F401
        import pandas  # noqa: F401
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "numpy", "pandas", "scipy", "scikit-learn", "tqdm", "matplotlib",
        ])


_bootstrap_colab()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import PchipInterpolator
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm.auto import tqdm


# %% [markdown]
# ## Configuration

# %%
@dataclass
class Config:
    seed: int = 42
    profile: str = "full"
    n_steps: int = 1057
    k_local: int = 256
    k_coarse: int = 2048
    embed_dim: int = 192
    temporal_channels: int = 64
    latent_dim: int = 192
    n_latents: int = 96
    n_heads: int = 8
    local_feature_channels: int = 16
    decoder_layers: int = 6
    dropout: float = 0.10
    pretrain_epochs: int = 80
    finetune_epochs: int = 80
    final_epochs: int = 80
    global_pretrain_steps: int = 200
    pretrain_batch_size: int = 32
    train_batch_size: int = 64
    grad_accum: int = 2
    pretrain_samples_per_epoch: int = 30000
    synthetic_samples_per_epoch: int = 6000
    supervised_repeat: int = 5
    lr_pretrain: float = 2e-3
    lr_finetune: float = 3e-4
    weight_decay: float = 1e-3
    patience: int = 15
    val_fraction: float = 0.20
    clip_grad: float = 1.0
    num_workers: int = 2
    use_test_time_adaptation: bool = True
    tta_steps: int = 4
    tta_lr: float = 5e-5
    exact_global_perceiver: bool = True
    benchmark_repeats: int = 2
    benchmark_columns: int = 10
    benchmark_base_epochs: int = 60
    benchmark_fold_epochs: int = 60

    @classmethod
    def for_profile(cls, name: str) -> "Config":
        cfg = cls(profile=name)
        if name == "smoke":
            cfg.k_local = 16
            cfg.k_coarse = 64
            cfg.embed_dim = 64
            cfg.temporal_channels = 16
            cfg.latent_dim = 64
            cfg.n_latents = 16
            cfg.n_heads = 4
            cfg.local_feature_channels = 4
            cfg.decoder_layers = 2
            cfg.pretrain_epochs = 1
            cfg.finetune_epochs = 2
            cfg.final_epochs = 1
            cfg.global_pretrain_steps = 3
            cfg.pretrain_batch_size = 32
            cfg.train_batch_size = 2
            cfg.grad_accum = 1
            cfg.pretrain_samples_per_epoch = 256
            cfg.synthetic_samples_per_epoch = 128
            cfg.supervised_repeat = 1
            cfg.patience = 2
            cfg.use_test_time_adaptation = False
            cfg.benchmark_repeats = 1
            cfg.benchmark_columns = 2
            cfg.benchmark_base_epochs = 1
            cfg.benchmark_fold_epochs = 1
        return cfg


def config_namespace(cfg: Config) -> str:
    """Sépare strictement les poids/caches incompatibles entre architectures."""
    if cfg.profile == "smoke":
        return "smoke"
    return (
        f"{cfg.profile}_e{cfg.embed_dim}_tc{cfg.temporal_channels}"
        f"_l{cfg.latent_dim}x{cfg.n_latents}_local{cfg.local_feature_channels}"
        f"_d{cfg.decoder_layers}_b{cfg.train_batch_size}a{cfg.grad_accum}"
        f"_ep{cfg.pretrain_epochs}-{cfg.finetune_epochs}-{cfg.final_epochs}"
        f"_bench{cfg.benchmark_base_epochs}-{cfg.benchmark_fold_epochs}"
        f"_es{cfg.patience}v2"
    )


FILES = {
    "x_train": "X_train_78VdSWL.csv",
    "y_train": "y_train_u0UkKEh.csv",
    "x_test": "X_test_XKVc4no.csv",
    "sample": "new_output_sample.csv",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _has_all_csv(root: Path) -> bool:
    return all((root / name).is_file() for name in FILES.values())


def _interactive_colab_kernel_available() -> bool:
    """`files.upload()` exige le kernel IPython du notebook parent.

    `!python script.py` démarre un nouveau processus : google.colab y est
    importable, mais `get_ipython()` vaut None. Tester seulement l'import de
    google.colab provoquait donc le traceback observé dans Colab.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and getattr(shell, "kernel", None) is not None
    except Exception:
        return False


def _subprocess_upload_help(missing: Sequence[str], searched: Sequence[Path]) -> str:
    locations = ", ".join(str(p) for p in searched)
    return (
        "CSV du challenge introuvables.\n"
        f"Fichiers manquants : {', '.join(missing)}\n"
        f"Dossiers examinés : {locations}\n\n"
        "Ce script est lancé avec `!python`, donc `google.colab.files.upload()` "
        "ne peut pas ouvrir le sélecteur depuis ce sous-processus.\n"
        "Exécute d'abord cette cellule Colab :\n\n"
        "    from google.colab import files\n"
        "    files.upload()\n\n"
        "Puis relance :\n\n"
        "    !python /content/colab_retrieval_perceiver_t1.py "
        "--profile full --data-dir /content\n\n"
        "Alternative interactive :\n\n"
        "    %run /content/colab_retrieval_perceiver_t1.py --profile full\n"
    )


def upload_data_if_needed(data_dir: Optional[str]) -> Path:
    explicit = Path(data_dir).expanduser().resolve() if data_dir else None
    if explicit is not None:
        candidates = [explicit]
    elif Path("/content").exists():
        candidates = [Path("/content/enedis_data"), Path("/content")]
    else:
        candidates = [Path("data").resolve()]

    # Les CSV déjà téléversés dans /content sont découverts sans interaction.
    for candidate in candidates:
        if _has_all_csv(candidate):
            print(f"CSV détectés dans {candidate}")
            return candidate

    root = candidates[0]
    root.mkdir(parents=True, exist_ok=True)
    missing = [name for name in FILES.values() if not (root / name).is_file()]

    if not _interactive_colab_kernel_available():
        raise SystemExit(_subprocess_upload_help(missing, candidates))

    try:
        from google.colab import files  # type: ignore
    except ImportError as exc:
        raise FileNotFoundError(f"CSV absents dans {root}: {missing}") from exc

    print("Téléverse les CSV manquants du challenge :", ", ".join(missing))
    uploaded = files.upload()
    for filename, content in uploaded.items():
        (root / Path(filename).name).write_bytes(content)
    missing = [name for name in FILES.values() if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Fichiers toujours absents après upload: {missing}")
    return root


def load_frames(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Chargement des CSV…")
    frames = [pd.read_csv(root / FILES[k], index_col=0) for k in ("x_train", "y_train", "x_test", "sample")]
    x_train, y_train, x_test, sample = frames
    if len(x_train) != len(x_test) or not x_train.index.equals(x_test.index):
        raise ValueError("X_train et X_test doivent partager les mêmes 1 057 timestamps")
    if not y_train.index.equals(x_train.index) or not sample.index.equals(x_test.index):
        raise ValueError("Index temporels incompatibles")
    return x_train, y_train, x_test, sample


def complete_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if not str(c).startswith("holed_")]


def robust_curve_normalize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise indépendamment chaque ligne (courbe), uniquement sur ses valeurs finies."""
    center = np.nanmedian(x, axis=1).astype(np.float32)
    mad = np.nanmedian(np.abs(x - center[:, None]), axis=1).astype(np.float32)
    scale = np.maximum(1.4826 * mad, 0.02 * (np.abs(center) + 1.0)).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-5)] = 1.0
    center[~np.isfinite(center)] = 0.0
    z = (x - center[:, None]) / scale[:, None]
    return z.astype(np.float32), center, scale


def baseline_pchip(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=True)
    obs = np.flatnonzero(np.isfinite(values))
    if len(obs) == 0:
        return np.zeros_like(values, dtype=np.float32)
    if len(obs) == 1:
        return np.full_like(values, values[obs[0]], dtype=np.float32)
    grid = np.arange(len(values))
    try:
        pred = PchipInterpolator(obs, values[obs], extrapolate=False)(grid)
    except ValueError:
        pred = np.interp(grid, obs, values[obs])
    linear = np.interp(grid, obs, values[obs])
    pred = np.where(np.isfinite(pred), pred, linear)
    pred[obs] = values[obs]
    return pred.astype(np.float32)


def gap_features(missing: np.ndarray) -> np.ndarray:
    """Longueur, position relative et distances aux bords pour chaque timestamp."""
    t = len(missing)
    out = np.zeros((t, 4), np.float32)
    i = 0
    while i < t:
        if not missing[i]:
            i += 1
            continue
        j = i
        while j + 1 < t and missing[j + 1]:
            j += 1
        length = j - i + 1
        p = np.arange(length, dtype=np.float32)
        out[i:j + 1, 0] = np.log1p(length) / np.log1p(t)
        out[i:j + 1, 1] = p / max(length - 1, 1)
        out[i:j + 1, 2] = np.minimum(p + 1, length - p) / max(length, 1)
        out[i:j + 1, 3] = np.select([length <= 3, length <= 47, length <= 95], [0, 1, 2], default=3) / 3.0
        i = j + 1
    return out


def time_features(index: pd.Index) -> np.ndarray:
    dt = pd.to_datetime(index)
    hour = dt.hour.to_numpy() * 2 + dt.minute.to_numpy() / 30.0
    dow = dt.dayofweek.to_numpy()
    doy = dt.dayofyear.to_numpy()
    return np.stack([
        np.sin(2 * np.pi * hour / 48), np.cos(2 * np.pi * hour / 48),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        (dow >= 5).astype(np.float32),
    ], axis=1).astype(np.float32)


def observed_stats(x: np.ndarray) -> Tuple[float, float]:
    obs = x[np.isfinite(x)]
    if len(obs) == 0:
        return 0.0, 1.0
    center = float(np.median(obs))
    scale = max(float(1.4826 * np.median(np.abs(obs - center))), 0.02 * (abs(center) + 1.0), 1e-5)
    return center, scale


# %% [markdown]
# ## TabPFN V14 exact — pipeline autonome du benchmark historique

# %%
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


def eval_tabpfn_v14_repeat(x_train: pd.DataFrame, y_train: pd.DataFrame,
                           donor_cols: Sequence[str], dev_cols: Sequence[str],
                           val_cols: Sequence[str]) -> Tuple[Dict[str, np.ndarray], int]:
    donor_df, dev_df = x_train[list(donor_cols)], y_train[list(dev_cols)]
    pool_df = pd.concat([donor_df, dev_df], axis=1)
    pool = pool_df.ffill().bfill().to_numpy(float)
    km, donor_labels = fit_shape_clusters_v14(donor_df)
    dev_filled = dev_df.ffill().bfill()
    zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = pd.Series(km.predict(zdev.T.to_numpy()), index=dev_df.columns)
    labels = pd.concat([donor_labels, dev_labels])
    positions = pd.Series(np.arange(len(labels)), index=labels.index)
    groups = {cl: positions[idx].to_numpy() for cl, idx in labels.groupby(labels).groups.items()}
    tfeat = time_features_v14(x_train.index)
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
        raise SystemExit("TABPFN_API_KEY absent : ajoute-le aux Secrets Colab ou à l'environnement.")
    tabpfn_client.set_access_token(key)
    from tabpfn_client import TabPFNRegressor
    _TABPFN_REGRESSOR = TabPFNRegressor


# %% [markdown]
# ## Masques challenge-like et pré-entraînement

# %%
def extract_gap_lengths(holed: pd.DataFrame) -> np.ndarray:
    lengths: List[int] = []
    for col in holed.columns:
        m = holed[col].isna().to_numpy()
        padded = np.r_[False, m, False]
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        lengths.extend((ends - starts).tolist())
    return np.asarray(lengths or [1, 2, 48], dtype=np.int32)


def sample_challenge_mask(rng: np.random.Generator, n_steps: int, gap_lengths: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_steps, dtype=bool)
    # Nombre de blocs proche de celui d'une target réelle, borné pour éviter les masques dégénérés.
    n_blocks = int(np.clip(rng.poisson(5) + 1, 1, 20))
    for _ in range(n_blocks):
        length = int(rng.choice(gap_lengths))
        length = min(max(length, 1), n_steps - 2)
        start = int(rng.integers(0, n_steps - length + 1))
        mask[start:start + length] = True
    if mask.mean() > 0.70:
        idx = rng.choice(np.flatnonzero(mask), size=int(0.70 * n_steps), replace=False)
        mask[:] = False
        mask[idx] = True
    return mask


class PretrainDataset(Dataset):
    def __init__(self, curves: np.ndarray, gap_lengths: np.ndarray, n_samples: int, seed: int):
        self.curves, self.gap_lengths, self.n_samples, self.seed = curves, gap_lengths, n_samples, seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx * 104729)
        row = self.curves[int(rng.integers(len(self.curves)))].copy()
        original_observed = np.isfinite(row)
        hide = sample_challenge_mask(rng, len(row), self.gap_lengths) & original_observed
        if not hide.any():
            hide[int(rng.choice(np.flatnonzero(original_observed)))] = True
        center, scale = observed_stats(np.where(hide, np.nan, row))
        truth = (np.nan_to_num(row, nan=center) - center) / scale
        visible = original_observed & ~hide
        value = np.where(visible, truth, 0.0).astype(np.float32)
        return (
            torch.from_numpy(value), torch.from_numpy(visible.astype(np.float32)),
            torch.from_numpy(truth.astype(np.float32)), torch.from_numpy(hide.astype(np.float32)),
        )


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.dw = nn.Conv1d(channels, channels, kernel, padding=pad, dilation=dilation, groups=channels)
        self.pw1 = nn.Conv1d(channels, channels * 4, 1)
        self.pw2 = nn.Conv1d(channels * 4, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.pw2(self.dropout(F.gelu(self.pw1(y))))
        return self.norm(x + self.dropout(y))


class SharedCurveEncoder(nn.Module):
    """Encodeur mask-aware partagé par toutes les courbes."""
    def __init__(self, cfg: Config):
        super().__init__()
        c = cfg.temporal_channels
        self.stem = nn.Conv1d(2, c, kernel_size=9, padding=4)
        self.blocks = nn.Sequential(*[
            ResidualTCNBlock(c, k, d, cfg.dropout)
            for k, d in ((7, 1), (11, 2), (15, 4), (23, 8))
        ])
        self.pool = nn.AdaptiveAvgPool1d(16)
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(c * 16, cfg.embed_dim), nn.LayerNorm(cfg.embed_dim))
        self.reconstruct = nn.Sequential(
            nn.Conv1d(c, c, 5, padding=2), nn.GELU(), nn.Conv1d(c, 1, 1),
        )

    def features(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(torch.stack([value, mask], dim=1)))

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(value, mask)
        return self.embedding(self.pool(feat)), self.reconstruct(feat).squeeze(1)


def masked_l1(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool() & torch.isfinite(truth)
    error = torch.where(valid, (pred - truth).abs(), torch.zeros_like(pred))
    return error.sum() / valid.sum().clamp_min(1)


def accumulation_divisor(step: int, total_steps: int, accumulation: int) -> int:
    """Taille réelle du groupe accumulé, y compris pour le dernier groupe incomplet."""
    group_start = (step // accumulation) * accumulation
    return min(accumulation, total_steps - group_start)


def autocast_context(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def pretrain_encoder(cfg: Config, encoder: SharedCurveEncoder, curves: np.ndarray,
                     gap_lengths: np.ndarray, device: torch.device, work: Path) -> None:
    ckpt = work / "encoder_pretrained.pt"
    if ckpt.exists():
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        encoder.load_state_dict(saved["model"] if isinstance(saved, dict) and "model" in saved else saved)
        print("Checkpoint encodeur chargé:", ckpt)
        return
    ds = PretrainDataset(curves, gap_lengths, cfg.pretrain_samples_per_epoch, cfg.seed)
    val_ds = PretrainDataset(curves, gap_lengths, min(2048, cfg.pretrain_samples_per_epoch // 4),
                             cfg.seed + 10_000_000)
    loader = DataLoader(ds, batch_size=cfg.pretrain_batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.pretrain_batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    opt = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr_pretrain, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=False)
    encoder.to(device)
    best, bad, history = float("inf"), 0, []
    for epoch in range(cfg.pretrain_epochs):
        encoder.train()
        total, count = 0.0, 0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(loader, desc=f"prétrain {epoch + 1}/{cfg.pretrain_epochs}")
        for step, (value, mask, truth, hide) in enumerate(bar):
            value, mask, truth, hide = [v.to(device, non_blocking=True) for v in (value, mask, truth, hide)]
            with autocast_context(device):
                _, pred = encoder(value, mask)
                raw_loss = masked_l1(pred, truth, hide)
                loss = raw_loss / accumulation_divisor(step, len(loader), cfg.grad_accum)
            scaler.scale(loss).backward()
            if (step + 1) % cfg.grad_accum == 0 or step + 1 == len(loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(encoder.parameters(), cfg.clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            total += float(raw_loss.detach())
            count += 1
            bar.set_postfix(l1=total / count)
        encoder.eval()
        val_total, val_count = 0.0, 0
        with torch.inference_mode():
            for value, mask, truth, hide in val_loader:
                value, mask, truth, hide = [v.to(device, non_blocking=True) for v in (value, mask, truth, hide)]
                with autocast_context(device):
                    _, pred = encoder(value, mask)
                    val_loss = masked_l1(pred, truth, hide)
                val_total += float(val_loss); val_count += 1
        val_mean = val_total / max(val_count, 1)
        train_mean = total / max(count, 1)
        history.append({"epoch": epoch + 1, "train_l1": train_mean, "val_l1": val_mean})
        print(f"prétrain epoch={epoch + 1}: train={train_mean:.4f}, val={val_mean:.4f}")
        if val_mean < best:
            best, bad = val_mean, 0
            torch.save({"model": encoder.state_dict(), "best_epoch": epoch + 1,
                        "best_val_l1": best, "history": history}, ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stopping encodeur: patience={cfg.patience}, best epoch={history[-bad-1]['epoch']}")
                break
    saved = torch.load(ckpt, map_location=device, weights_only=False)
    encoder.load_state_dict(saved["model"])


@torch.inference_mode()
def encode_reference_bank(cfg: Config, encoder: SharedCurveEncoder, curves_z: np.ndarray,
                          device: torch.device, work: Path) -> np.ndarray:
    cache = work / "reference_embeddings.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.shape == (len(curves_z), cfg.embed_dim):
            print("Embeddings chargés:", cache, emb.shape)
            return emb
    encoder.eval().to(device)
    chunks = []
    bs = 512 if device.type == "cuda" else 32
    for start in tqdm(range(0, len(curves_z), bs), desc="encodage des 57k références"):
        x = torch.from_numpy(curves_z[start:start + bs]).to(device)
        m = torch.isfinite(x).float()
        x = torch.nan_to_num(x)
        with autocast_context(device):
            e, _ = encoder(x, m)
        chunks.append(F.normalize(e.float(), dim=-1).cpu().numpy())
    emb = np.concatenate(chunks).astype(np.float32)
    np.save(cache, emb)
    return emb


# %% [markdown]
# ## Retrieval coarse-to-fine et cache anti-fuite

# %%
@torch.inference_mode()
def encode_targets(cfg: Config, encoder: SharedCurveEncoder, targets: np.ndarray,
                   device: torch.device) -> np.ndarray:
    rows = []
    for raw in targets:
        center, scale = observed_stats(raw)
        z = ((raw - center) / scale).astype(np.float32)
        x = torch.from_numpy(np.nan_to_num(z)[None]).to(device)
        m = torch.from_numpy(np.isfinite(z).astype(np.float32)[None]).to(device)
        with autocast_context(device):
            e, _ = encoder(x, m)
        rows.append(F.normalize(e.float(), dim=-1).cpu().numpy()[0])
    return np.asarray(rows, np.float32)


def coarse_neighbors(target_emb: np.ndarray, ref_emb: np.ndarray, k: int,
                     device: torch.device) -> np.ndarray:
    refs = torch.from_numpy(ref_emb).to(device)
    out = []
    batch = 64
    with torch.inference_mode():
        for start in tqdm(range(0, len(target_emb), batch), desc="retrieval coarse"):
            q = torch.from_numpy(target_emb[start:start + batch]).to(device)
            score = q @ refs.T
            out.append(torch.topk(score, k=min(k, len(ref_emb)), dim=1).indices.cpu().numpy())
    del refs
    return np.concatenate(out)


def affine_rerank_one(target: np.ndarray, ref_curves: np.ndarray, candidate_idx: np.ndarray,
                      k_local: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reranking exact sur points observés. Retourne indices et [D,rho,a,b]."""
    obs = np.isfinite(target)
    y = target[obs].astype(np.float64)
    x = ref_curves[candidate_idx][:, obs].astype(np.float64)
    finite = np.isfinite(x)
    x_mean = np.nanmean(x, axis=1)
    y_mean = float(y.mean()) if len(y) else 0.0
    xc = np.where(finite, x - x_mean[:, None], 0.0)
    yc = y - y_mean
    denom_x = np.sum(xc * xc, axis=1)
    cov = np.sum(xc * yc[None], axis=1)
    a = cov / np.maximum(denom_x, 1e-8)
    a = np.clip(a, -5.0, 5.0)
    b = y_mean - a * x_mean
    calibrated = a[:, None] * x + b[:, None]
    residual = np.abs(calibrated - y[None])
    mae = np.nanmean(residual, axis=1)
    rho = cov / np.sqrt(np.maximum(denom_x * np.sum(yc * yc), 1e-8))
    y_scale = max(1.4826 * np.median(np.abs(y - np.median(y))), 1.0)
    score = mae / y_scale + 0.25 * (1.0 - np.clip(rho, -1, 1))
    score[~np.isfinite(score)] = np.inf
    order = np.argsort(score)[:k_local]
    idx = candidate_idx[order]
    meta = np.stack([score[order], rho[order], a[order], b[order]], axis=1).astype(np.float32)
    return idx.astype(np.int32), meta


def build_retrieval_cache(name: str, cfg: Config, encoder: SharedCurveEncoder,
                          targets: np.ndarray, ref_curves: np.ndarray, ref_emb: np.ndarray,
                          device: torch.device, work: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = work / f"retrieval_{name}_k{cfg.k_local}_c{cfg.k_coarse}.npz"
    if path.exists():
        z = np.load(path)
        if len(z["indices"]) == len(targets):
            print("Cache retrieval chargé:", path)
            return z["indices"], z["meta"]
    target_emb = encode_targets(cfg, encoder, targets, device)
    candidates = coarse_neighbors(target_emb, ref_emb, cfg.k_coarse, device)
    indices, metadata = [], []
    for i in tqdm(range(len(targets)), desc=f"reranking affine {name}"):
        idx, meta = affine_rerank_one(targets[i], ref_curves, candidates[i], cfg.k_local)
        indices.append(idx)
        metadata.append(meta)
    indices_a, metadata_a = np.asarray(indices, np.int32), np.asarray(metadata, np.float32)
    np.savez_compressed(path, indices=indices_a, meta=metadata_a)
    return indices_a, metadata_a


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=0)
    sorted_v = np.take_along_axis(values, order, axis=0)
    sorted_w = np.take_along_axis(np.broadcast_to(weights[:, None], values.shape), order, axis=0)
    cdf = np.cumsum(sorted_w, axis=0)
    threshold = 0.5 * sorted_w.sum(axis=0, keepdims=True)
    pos = (cdf >= threshold).argmax(axis=0)
    return sorted_v[pos, np.arange(values.shape[1])]


def make_model_sample(raw_target: np.ndarray, truth: np.ndarray, ref_curves: np.ndarray,
                      ref_idx: np.ndarray, meta: np.ndarray, tf: np.ndarray) -> Dict[str, np.ndarray]:
    observed = np.isfinite(raw_target)
    missing = ~observed
    center, scale = observed_stats(raw_target)
    base = baseline_pchip(raw_target)
    target_z = np.where(observed, (raw_target - center) / scale, 0.0).astype(np.float32)
    base_z = ((base - center) / scale).astype(np.float32)
    refs = ref_curves[ref_idx].astype(np.float32)
    # Calibration estimée avant toute utilisation des timestamps cachés.
    calibrated = meta[:, 2, None] * refs + meta[:, 3, None]
    refs_z = ((calibrated - center) / scale).astype(np.float32)
    refs_z = np.nan_to_num(refs_z)
    weights = np.exp(-np.clip(meta[:, 0], 0, 20))
    weights /= max(weights.sum(), 1e-8)
    neighbor = weighted_median(refs_z, weights).astype(np.float32)
    truth_z = ((truth - center) / scale).astype(np.float32)
    model_meta = meta.copy()
    # a est sans dimension ; b doit être exprimé dans l'échelle robuste de la cible.
    model_meta[:, 3] = (model_meta[:, 3] - center) / scale
    return {
        "target": target_z, "observed": observed.astype(np.float32), "baseline": base_z,
        "neighbors": refs_z, "neighbor_median": neighbor, "metadata": model_meta.astype(np.float32),
        "time": tf, "gap": gap_features(missing), "truth": truth_z,
        "loss_mask": (missing & np.isfinite(truth)).astype(np.float32),
        "center": np.float32(center), "scale": np.float32(scale),
    }


class RetrievalDataset(Dataset):
    def __init__(self, targets: np.ndarray, truths: np.ndarray, refs: np.ndarray,
                 indices: np.ndarray, meta: np.ndarray, tf: np.ndarray,
                 selected: Sequence[int], repeat: int = 1):
        self.targets, self.truths, self.refs = targets, truths, refs
        self.indices, self.meta, self.tf = indices, meta, tf
        self.selected = np.asarray(selected, dtype=np.int64)
        self.repeat = repeat

    def __len__(self) -> int:
        return len(self.selected) * self.repeat

    def __getitem__(self, item: int):
        i = int(self.selected[item % len(self.selected)])
        sample = make_model_sample(self.targets[i], self.truths[i], self.refs,
                                   self.indices[i], self.meta[i], self.tf)
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v)
                for k, v in sample.items()}


class SyntheticRetrievalDataset(Dataset):
    """Transforme des références complètes en problèmes d'imputation réalistes.

    Le reranking mis en cache n'utilise que les points laissés visibles et la
    cible elle-même est absente des candidats.
    """
    def __init__(self, refs: np.ndarray, target_ids: np.ndarray, masked_targets: np.ndarray,
                 indices: np.ndarray, metadata: np.ndarray, tf: np.ndarray):
        self.refs, self.target_ids, self.masked_targets = refs, target_ids, masked_targets
        self.indices, self.metadata, self.tf = indices, metadata, tf

    def __len__(self) -> int:
        return len(self.target_ids)

    def __getitem__(self, item: int):
        target_id = int(self.target_ids[item])
        truth = self.refs[target_id].copy()
        sample = make_model_sample(self.masked_targets[item], truth, self.refs,
                                   self.indices[item], self.metadata[item], self.tf)
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v)
                for k, v in sample.items()}


def build_synthetic_cache(cfg: Config, refs: np.ndarray, target_ids: np.ndarray,
                          candidates: np.ndarray, gap_lengths: np.ndarray,
                          work: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = work / f"synthetic_training_{len(target_ids)}_k{cfg.k_local}.npz"
    if path.exists():
        z = np.load(path)
        return z["masked"], z["indices"], z["metadata"]
    masked, indices, metadata = [], [], []
    for item, target_id in enumerate(tqdm(target_ids, desc="cache synthétique challenge-like")):
        truth = refs[int(target_id)]
        rng = np.random.default_rng(cfg.seed + 9000 + item * 13007)
        hide = sample_challenge_mask(rng, len(truth), gap_lengths) & np.isfinite(truth)
        raw = truth.copy()
        raw[hide] = np.nan
        candidate = candidates[item]
        candidate = candidate[candidate != int(target_id)]
        idx, meta = affine_rerank_one(raw, refs, candidate, cfg.k_local)
        masked.append(raw)
        indices.append(idx)
        metadata.append(meta)
    arrays = (np.asarray(masked, np.float32), np.asarray(indices, np.int32),
              np.asarray(metadata, np.float32))
    np.savez_compressed(path, masked=arrays[0], indices=arrays[1], metadata=arrays[2])
    return arrays


# %% [markdown]
# ## Perceiver global, T1 local et décodeur TimeMixer

# %%
class PerceiverGlobal(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(cfg.n_latents, cfg.latent_dim) * 0.02)
        self.ref_proj = nn.Linear(cfg.embed_dim, cfg.latent_dim)
        self.cross = nn.MultiheadAttention(cfg.latent_dim, cfg.n_heads, cfg.dropout, batch_first=True)
        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(cfg.latent_dim, cfg.n_heads, cfg.dropout, batch_first=True)
            for _ in range(2)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.latent_dim) for _ in range(3)])
        self.ff = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim * 4), nn.GELU(),
                                nn.Dropout(cfg.dropout), nn.Linear(cfg.latent_dim * 4, cfg.latent_dim))

    def forward(self, reference_embeddings: torch.Tensor) -> torch.Tensor:
        refs = self.ref_proj(reference_embeddings).unsqueeze(0)
        q = self.latents.unsqueeze(0)
        x, _ = self.cross(q, refs, refs, need_weights=False)
        x = self.norms[0](q + x)
        for i, attn in enumerate(self.self_attn):
            y, _ = attn(x, x, x, need_weights=False)
            x = self.norms[i + 1](x + y + self.ff(x))
        return x


class T1TargetAttention(nn.Module):
    """Channel-head binding avec requête cible uniquement : O(K*C*L), pas O(K²)."""
    def __init__(self, cfg: Config):
        super().__init__()
        c = cfg.local_feature_channels
        self.c = c
        self.stem = nn.Conv1d(2, c, kernel_size=9, stride=4, padding=4)
        self.q_small = nn.Conv1d(c, c, 5, padding=2, groups=c)
        self.q_large = nn.Conv1d(c, c, 17, padding=8, groups=c)
        self.k_small = nn.Conv1d(c, c, 5, padding=2, groups=c)
        self.k_large = nn.Conv1d(c, c, 17, padding=8, groups=c)
        self.v_small = nn.Conv1d(c, c, 5, padding=2, groups=c)
        self.v_large = nn.Conv1d(c, c, 17, padding=8, groups=c)
        self.meta = nn.Sequential(nn.Linear(4, c), nn.Tanh())
        self.out = nn.Conv1d(c, cfg.latent_dim, 1)

    def forward(self, target: torch.Tensor, observed: torch.Tensor,
                neighbors: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        b, k, t = neighbors.shape
        tgt = self.stem(torch.stack([target, observed], dim=1))
        n_in = torch.stack([neighbors, torch.ones_like(neighbors)], dim=2).reshape(b * k, 2, t)
        nbr = self.stem(n_in).reshape(b, k, self.c, -1)
        q = self.q_small(tgt) + self.q_large(tgt)                       # B,C,L
        flat = nbr.reshape(b * k, self.c, -1)
        keys = (self.k_small(flat) + self.k_large(flat)).reshape_as(nbr)
        vals = (self.v_small(flat) + self.v_large(flat)).reshape_as(nbr)
        # Chaque canal CNN correspond à une tête : score indépendant par canal.
        logits = torch.einsum("bcl,bkcl->bkc", q, keys) / math.sqrt(q.shape[-1])
        logits = logits + self.meta(metadata)
        attn = logits.softmax(dim=1)
        fused = torch.einsum("bkc,bkcl->bcl", attn, vals)
        return self.out(fused)


class MultiScaleMixer(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.scales = (1, 2, 4, 8, 24, 48, 96, 336)
        self.mix = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels, 3, padding=1, groups=channels),
                          nn.Conv1d(channels, channels, 1), nn.GELU())
            for _ in self.scales
        ])
        self.fuse = nn.Sequential(nn.Conv1d(channels * len(self.scales), channels * 2, 1),
                                  nn.GELU(), nn.Dropout(dropout), nn.Conv1d(channels * 2, channels, 1))
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[-1]
        levels = []
        for scale, block in zip(self.scales, self.mix):
            if scale == 1:
                y = x
            else:
                y = F.avg_pool1d(x, kernel_size=min(scale, t), stride=min(scale, t), ceil_mode=True)
            y = block(y)
            levels.append(F.interpolate(y, size=t, mode="linear", align_corners=False))
        return self.norm(x + self.fuse(torch.cat(levels, dim=1)))


class RetrievalPerceiverImputer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.latent_dim
        self.global_pool = PerceiverGlobal(cfg)
        self.local_t1 = T1TargetAttention(cfg)
        self.target_stem = nn.Conv1d(15, d, 9, padding=4)
        self.local_up = nn.Conv1d(d, d, 1)
        self.global_cross = nn.MultiheadAttention(d, cfg.n_heads, cfg.dropout, batch_first=True)
        self.mixers = nn.ModuleList([MultiScaleMixer(d, cfg.dropout) for _ in range(cfg.decoder_layers)])
        self.experts = nn.ModuleList([nn.Conv1d(d, 1, 1) for _ in range(4)])
        self.gate = nn.Sequential(nn.Conv1d(d + 4, d // 2, 1), nn.GELU(), nn.Conv1d(d // 2, 4, 1))

    def global_tokens(self, reference_embeddings: torch.Tensor) -> torch.Tensor:
        return self.global_pool(reference_embeddings)

    def forward(self, batch: Dict[str, torch.Tensor], reference_embeddings: torch.Tensor,
                cached_global: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        target, observed, baseline = batch["target"], batch["observed"], batch["baseline"]
        local = self.local_t1(target, observed, batch["neighbors"], batch["metadata"])
        local = F.interpolate(self.local_up(local), size=target.shape[-1], mode="linear", align_corners=False)
        inp = torch.cat([
            target[:, None], observed[:, None], baseline[:, None], batch["neighbor_median"][:, None],
            batch["time"].transpose(1, 2), batch["gap"].transpose(1, 2),
        ], dim=1)
        x = self.target_stem(inp) + local
        global_tokens = cached_global if cached_global is not None else self.global_tokens(reference_embeddings)
        g = global_tokens.expand(x.shape[0], -1, -1)
        q = x.transpose(1, 2)
        cross, _ = self.global_cross(q, g, g, need_weights=False)
        x = (q + cross).transpose(1, 2)
        for mixer in self.mixers:
            x = mixer(x)
        expert = torch.cat([head(x) for head in self.experts], dim=1)
        gates = self.gate(torch.cat([x, batch["gap"].transpose(1, 2)], dim=1)).softmax(dim=1)
        residual = (expert * gates).sum(dim=1)
        return baseline + residual, gates


def pretrain_global_perceiver(cfg: Config, model: RetrievalPerceiverImputer,
                              ref_emb: torch.Tensor, device: torch.device, work: Path) -> None:
    """Apprend le bottleneck global comme autoencodeur d'ensemble.

    Les mini-ensembles changent à chaque pas. Après cet apprentissage, un passage
    final avec les 57 140 embeddings fabrique le contexte réellement utilisé.
    """
    path = work / "global_perceiver_pretrained.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model.global_pool.load_state_dict(saved["model"] if isinstance(saved, dict) and "model" in saved else saved)
        print("Checkpoint Perceiver global chargé:", path)
        return
    model.global_pool.to(device)
    ref_emb = ref_emb.to(device)
    decoder = nn.MultiheadAttention(cfg.latent_dim, cfg.n_heads, cfg.dropout, batch_first=True).to(device)
    query_proj = nn.Linear(cfg.embed_dim, cfg.latent_dim).to(device)
    opt = torch.optim.AdamW(list(model.global_pool.parameters()) + list(decoder.parameters()) +
                            list(query_proj.parameters()), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    set_size = min(8192, len(ref_emb))
    query_size = min(512, set_size)
    val_size = min(8192, len(ref_emb))
    val_idx = torch.randperm(len(ref_emb), generator=gen, device=device)[:val_size]
    val_subset = ref_emb[val_idx]
    val_query = torch.arange(min(512, val_size), device=device)
    validate_every = 10
    best, bad, history = float("inf"), 0, []
    opt.zero_grad(set_to_none=True)
    for step in tqdm(range(cfg.global_pretrain_steps), desc="prétrain Perceiver global"):
        idx = torch.randperm(len(ref_emb), generator=gen, device=device)[:set_size]
        subset = ref_emb[idx]
        q_idx = torch.randperm(set_size, generator=gen, device=device)[:query_size]
        with autocast_context(device):
            latents = model.global_pool(subset)
            q = query_proj(subset[q_idx]).unsqueeze(0)
            reconstructed, _ = decoder(q, latents, latents, need_weights=False)
            # embed_dim == latent_dim dans les profils fournis : cible fixe,
            # ce qui interdit la solution triviale ref_proj -> zéro.
            target = subset[q_idx].unsqueeze(0).detach()
            loss = F.smooth_l1_loss(reconstructed, target) / accumulation_divisor(
                step, cfg.global_pretrain_steps, cfg.grad_accum)
        loss.backward()
        if (step + 1) % cfg.grad_accum == 0 or step + 1 == cfg.global_pretrain_steps:
            nn.utils.clip_grad_norm_(model.global_pool.parameters(), cfg.clip_grad)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if (step + 1) % validate_every == 0 or step + 1 == cfg.global_pretrain_steps:
            model.global_pool.eval(); decoder.eval(); query_proj.eval()
            with torch.inference_mode(), autocast_context(device):
                val_latents = model.global_pool(val_subset)
                vq = query_proj(val_subset[val_query]).unsqueeze(0)
                val_reconstructed, _ = decoder(vq, val_latents, val_latents, need_weights=False)
                val_loss = float(F.smooth_l1_loss(
                    val_reconstructed, val_subset[val_query].unsqueeze(0)).float())
            history.append({"step": step + 1, "val_l1": val_loss})
            print(f"Perceiver step={step + 1}: validation L1={val_loss:.6f}")
            if val_loss < best:
                best, bad = val_loss, 0
                torch.save({"model": model.global_pool.state_dict(), "best_step": step + 1,
                            "best_val_l1": best, "history": history}, path)
            else:
                bad += 1
            model.global_pool.train(); decoder.train(); query_proj.train()
            if bad >= cfg.patience:
                print(f"Early stopping Perceiver: patience={cfg.patience}")
                break
    saved = torch.load(path, map_location=device, weights_only=False)
    model.global_pool.load_state_dict(saved["model"])


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def evaluate_loader(model: RetrievalPerceiverImputer, loader: DataLoader, ref_emb: torch.Tensor,
                    device: torch.device) -> Tuple[float, Dict[int, List[float]]]:
    model.eval()
    errors, by_class = [], {0: [], 1: [], 2: [], 3: []}
    with torch.inference_mode():
        with autocast_context(device):
            global_tokens = model.global_tokens(ref_emb)
        for batch in loader:
            batch = to_device(batch, device)
            with autocast_context(device):
                pred, _ = model(batch, ref_emb, global_tokens)
            valid = batch["loss_mask"].bool() & torch.isfinite(batch["truth"])
            err = torch.where(valid, (pred.float() - batch["truth"]).abs(), torch.zeros_like(pred.float()))
            scale = batch["scale"][:, None]
            errors.extend((err * scale)[valid].cpu().tolist())
            cls = torch.round(batch["gap"][:, :, 3] * 3).long()
            for c in range(4):
                m = valid & (cls == c)
                by_class[c].extend((err * scale)[m].cpu().tolist())
    return float(np.mean(errors)) if errors else float("inf"), by_class


def train_imputer(cfg: Config, model: RetrievalPerceiverImputer, train_loader: DataLoader,
                  val_loader: DataLoader, ref_emb: torch.Tensor, device: torch.device,
                  work: Path, epochs: int, checkpoint_name: str) -> float:
    ckpt = work / checkpoint_name
    model.to(device)
    ref_emb = ref_emb.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=False)
    best, bad = float("inf"), 0
    for epoch in range(epochs):
        model.train()
        # Le contexte est commun à tous les exemples. Le calculer une fois par
        # epoch évite des milliers de cross-attentions inutiles sur 57k tokens.
        with torch.no_grad(), autocast_context(device):
            cached_global = model.global_tokens(ref_emb).detach()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        bar = tqdm(train_loader, desc=f"finetune {epoch + 1}/{epochs}")
        for step, batch in enumerate(bar):
            batch = to_device(batch, device)
            with autocast_context(device):
                pred, _ = model(batch, ref_emb, cached_global)
                raw_loss = masked_l1(pred, batch["truth"], batch["loss_mask"])
                loss = raw_loss / accumulation_divisor(step, len(train_loader), cfg.grad_accum)
            scaler.scale(loss).backward()
            if (step + 1) % cfg.grad_accum == 0 or step + 1 == len(train_loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += float(raw_loss.detach())
            bar.set_postfix(l1=running / (step + 1))
        val_mae, categories = evaluate_loader(model, val_loader, ref_emb, device)
        cat_means = {k: float(np.mean(v)) if v else None for k, v in categories.items()}
        print(f"epoch={epoch + 1} validation MAE={val_mae:.3f} W catégories={cat_means}")
        if val_mae < best:
            best, bad = val_mae, 0
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "val_mae": best}, ckpt)
        else:
            bad += 1
            if bad >= cfg.patience:
                print("Early stopping")
                break
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return best


def train_fixed_epochs(cfg: Config, model: RetrievalPerceiverImputer, loader: DataLoader,
                       ref_emb: torch.Tensor, device: torch.device, epochs: int,
                       description: str, val_loader: Optional[DataLoader] = None,
                       checkpoint: Optional[Path] = None) -> Dict:
    """Entraînement borné à `epochs`, avec early stopping si val_loader est fourni."""
    model.to(device)
    ref_emb = ref_emb.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_finetune, weight_decay=cfg.weight_decay)
    best, bad, history = float("inf"), 0, []
    for epoch in range(epochs):
        model.train()
        with torch.no_grad(), autocast_context(device):
            global_tokens = model.global_tokens(ref_emb).detach()
        opt.zero_grad(set_to_none=True)
        bar = tqdm(loader, desc=f"{description} {epoch + 1}/{epochs}")
        running = 0.0
        for step, batch in enumerate(bar):
            batch = to_device(batch, device)
            with autocast_context(device):
                pred, _ = model(batch, ref_emb, global_tokens)
                raw_loss = masked_l1(pred, batch["truth"], batch["loss_mask"])
                loss = raw_loss / accumulation_divisor(step, len(loader), cfg.grad_accum)
            loss.backward()
            if (step + 1) % cfg.grad_accum == 0 or step + 1 == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                opt.step()
                opt.zero_grad(set_to_none=True)
            running += float(raw_loss.detach())
            bar.set_postfix(l1=running / (step + 1))
        train_mean = running / max(len(loader), 1)
        if val_loader is None:
            history.append({"epoch": epoch + 1, "train_l1": train_mean})
            continue
        val_mae, categories = evaluate_loader(model, val_loader, ref_emb, device)
        history.append({"epoch": epoch + 1, "train_l1": train_mean, "val_mae": val_mae})
        print(f"{description}: epoch={epoch + 1}, validation MAE={val_mae:.3f} W")
        if val_mae < best:
            best, bad = val_mae, 0
            if checkpoint is not None:
                torch.save({"model": model.state_dict(), "best_epoch": epoch + 1,
                            "best_val_mae": best, "history": history}, checkpoint)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"Early stopping {description}: patience={cfg.patience}")
                break
    best_epoch = epochs
    if val_loader is not None and checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        best_epoch, best = int(saved["best_epoch"]), float(saved["best_val_mae"])
        history = saved.get("history", history)
    return {"best_epoch": best_epoch, "best_val_mae": best if val_loader is not None else None,
            "history": history, "stopped_early": len(history) < epochs}


@torch.inference_mode()
def retrieval_errors_by_column(model: RetrievalPerceiverImputer, dataset: RetrievalDataset,
                               columns: Sequence[str], ref_emb: torch.Tensor,
                               device: torch.device) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    model.eval().to(device)
    ref_emb = ref_emb.to(device)
    with autocast_context(device):
        global_tokens = model.global_tokens(ref_emb).detach()
    errors, categories = {}, {"1": [], "2-3": [], "4-47": [], "48-95": [], ">=96": []}
    for item, col in enumerate(columns):
        batch = {k: v[None] for k, v in dataset[item].items()}
        batch = to_device(batch, device)
        with autocast_context(device):
            pred, _ = model(batch, ref_emb, global_tokens)
        valid = batch["loss_mask"][0].bool() & torch.isfinite(batch["truth"][0])
        err = ((pred[0].float() - batch["truth"][0]).abs() * batch["scale"]).cpu().numpy()
        errors[col] = err[valid.cpu().numpy()]
        lengths = run_lengths_for((batch["observed"][0] < 0.5).cpu().numpy())
        valid_np = valid.cpu().numpy()
        for name, select in {
            "1": lengths == 1, "2-3": (lengths >= 2) & (lengths <= 3),
            "4-47": (lengths >= 4) & (lengths <= 47),
            "48-95": (lengths >= 48) & (lengths <= 95), ">=96": lengths >= 96,
        }.items():
            categories[name].extend(err[valid_np & select].tolist())
    return errors, categories


def save_json_atomic(payload: Dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def categorize_column_errors(errors: Dict[str, np.ndarray], x_train: pd.DataFrame,
                             y_train: pd.DataFrame) -> Dict[str, List[float]]:
    categories = {"1": [], "2-3": [], "4-47": [], "48-95": [], ">=96": []}
    for col, values in errors.items():
        missing = x_train[col].isna().to_numpy()
        truth_valid = np.isfinite(y_train[col].to_numpy()[missing])
        lengths = run_lengths_for(missing)[missing][truth_valid]
        values = np.asarray(values)
        if len(values) != len(lengths):
            raise RuntimeError(f"Alignement erreurs/masque invalide pour {col}")
        selectors = {
            "1": lengths == 1, "2-3": (lengths >= 2) & (lengths <= 3),
            "4-47": (lengths >= 4) & (lengths <= 47),
            "48-95": (lengths >= 48) & (lengths <= 95), ">=96": lengths >= 96,
        }
        for name, select in selectors.items():
            categories[name].extend(values[select].tolist())
    return categories


def print_benchmark_progress(progress: Dict, total_repeats: int, started_at: float) -> None:
    rows, all_tb, all_rt = [], [], []
    for key in sorted(progress.get("repeats", {}), key=lambda x: int(x)):
        rec = progress["repeats"][key]
        if not (rec.get("tabpfn_complete") and rec.get("retrieval_complete")):
            continue
        cols = rec["val_cols"]
        tb = np.concatenate([np.asarray(rec["tabpfn"][c]) for c in cols])
        rt = np.concatenate([np.asarray(rec["retrieval"][c]) for c in cols])
        rows.append((int(key) + 1, tb.mean(), rt.mean(), rt.mean() - tb.mean()))
        all_tb.append(tb); all_rt.append(rt)
    if not rows:
        return
    print("\nRésultats cumulés provisoires")
    print("| Repeat | TabPFN V14 | Retrieval | Δ | Gagnant |")
    print("|---:|---:|---:|---:|:---|")
    for repeat, tb, rt, delta in rows:
        winner = "Retrieval" if rt < tb else "TabPFN" if tb < rt else "Égalité"
        print(f"| {repeat} | {tb:.2f} | {rt:.2f} | {delta:+.2f} | {winner} |")
    pooled_tb, pooled_rt = np.concatenate(all_tb), np.concatenate(all_rt)
    elapsed = time.time() - started_at
    eta = elapsed / len(rows) * max(total_repeats - len(rows), 0)
    column_wins = {"retrieval": 0, "tabpfn": 0, "tie": 0}
    for key in progress["repeats"]:
        rec = progress["repeats"][key]
        if not (rec.get("tabpfn_complete") and rec.get("retrieval_complete")):
            continue
        for col in rec["val_cols"]:
            mt, mr = np.mean(rec["tabpfn"][col]), np.mean(rec["retrieval"][col])
            column_wins["retrieval" if mr < mt else "tabpfn" if mt < mr else "tie"] += 1
    print(f"Pool provisoire: TabPFN={pooled_tb.mean():.2f} W, Retrieval={pooled_rt.mean():.2f} W; "
          f"victoires colonnes={column_wins}; ETA≈{eta / 3600:.2f} h\n", flush=True)


def run_paired_benchmark(cfg: Config, base_model: RetrievalPerceiverImputer,
                         x_train: pd.DataFrame, y_train: pd.DataFrame,
                         train_raw: np.ndarray, train_truth: np.ndarray,
                         refs: np.ndarray, retrieval_idx: np.ndarray, retrieval_meta: np.ndarray,
                         tf: np.ndarray, synthetic_ds: SyntheticRetrievalDataset,
                         ref_emb: torch.Tensor, device: torch.device, work: Path) -> Dict:
    """Protocole strict de tabpfn_v14_confirm.py : mêmes seeds et colonnes."""
    configure_tabpfn()
    donor_cols = [c for c in x_train.columns if not str(c).startswith("holed_")]
    columns = list(y_train.columns)
    position = {c: i for i, c in enumerate(columns)}
    progress_path = work / "paired_10x10_progress.json"
    signature = {
        "schema": 3, "seeds": list(range(42, 42 + cfg.benchmark_repeats)),
        "columns_per_repeat": cfg.benchmark_columns,
        "base_epochs": cfg.benchmark_base_epochs, "fold_epochs": cfg.benchmark_fold_epochs,
        "config": asdict(cfg),
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("signature") != signature:
            raise RuntimeError("Le benchmark sauvegardé utilise une configuration différente")
    else:
        progress = {"signature": signature, "repeats": {}}

    # Base commune autorisée : uniquement masques synthétiques, jamais y_train.
    base_path = work / "benchmark_synthetic_base.pt"
    if base_path.exists():
        base_saved = torch.load(base_path, map_location="cpu", weights_only=False)
        synthetic_state = base_saved["model"]
        base_training = base_saved.get("training", {})
    else:
        synthetic_model = copy.deepcopy(base_model)
        synth_train_ids, synth_val_ids = train_test_split(
            np.arange(len(synthetic_ds)), test_size=0.1, random_state=cfg.seed)
        synth_loader = DataLoader(Subset(synthetic_ds, synth_train_ids), batch_size=cfg.train_batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True)
        synth_val_loader = DataLoader(Subset(synthetic_ds, synth_val_ids), batch_size=cfg.train_batch_size,
                                      shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        base_training = train_fixed_epochs(
            cfg, synthetic_model, synth_loader, ref_emb, device, cfg.benchmark_base_epochs,
            "base synthétique", val_loader=synth_val_loader,
            checkpoint=work / "benchmark_synthetic_base_best.pt")
        synthetic_state = {k: v.detach().cpu() for k, v in synthetic_model.state_dict().items()}
        torch.save({"model": synthetic_state, "config": asdict(cfg),
                    "training": base_training}, base_path)
        del synthetic_model
        torch.cuda.empty_cache()

    benchmark_started = time.time()
    for repeat in range(cfg.benchmark_repeats):
        seed = 42 + repeat
        dev_cols, val_full = train_test_split(columns, test_size=0.1, random_state=seed)
        val_cols = list(np.random.RandomState(seed).choice(
            val_full, size=min(cfg.benchmark_columns, len(val_full)), replace=False))
        rec = progress["repeats"].setdefault(str(repeat), {
            "seed": seed, "val_cols": val_cols, "tabpfn": {}, "retrieval": {},
            "tabpfn_categories": {}, "retrieval_categories": {},
            "tabpfn_complete": False, "retrieval_complete": False,
        })
        if rec["val_cols"] != val_cols:
            raise RuntimeError(f"Colonnes sauvegardées incohérentes au repeat {repeat + 1}")

        if not rec["tabpfn_complete"]:
            tab_started = time.time()
            tab_errors, fallbacks = eval_tabpfn_v14_repeat(x_train, y_train, donor_cols, dev_cols, val_cols)
            rec["tabpfn"] = {c: e.tolist() for c, e in tab_errors.items()}
            rec["tabpfn_categories"] = categorize_column_errors(tab_errors, x_train, y_train)
            rec["tabpfn_seconds"] = time.time() - tab_started
            rec["tabpfn_fallbacks"] = fallbacks
            rec["tabpfn_complete"] = True
            save_json_atomic(progress, progress_path)
        tab_values = np.concatenate([np.asarray(rec["tabpfn"][c]) for c in val_cols])
        print(f"\n[repeat {repeat + 1}/{cfg.benchmark_repeats}] TabPFN V14: "
              f"MAE={tab_values.mean():.2f} W, temps={rec.get('tabpfn_seconds', 0):.1f}s, "
              f"replis={rec.get('tabpfn_fallbacks', 0)}, seed={seed}", flush=True)

        if not rec["retrieval_complete"]:
            retrieval_started = time.time()
            fold_model = copy.deepcopy(base_model)
            fold_model.load_state_dict(synthetic_state)
            fold_train_cols, fold_internal_val_cols = train_test_split(
                list(dev_cols), test_size=0.1, random_state=seed + 100_000)
            dev_ids = np.asarray([position[c] for c in fold_train_cols], dtype=np.int64)
            internal_val_ids = np.asarray([position[c] for c in fold_internal_val_cols], dtype=np.int64)
            val_ids = np.asarray([position[c] for c in val_cols], dtype=np.int64)
            # Assertion anti-fuite explicite avant toute construction du loader.
            if np.intersect1d(dev_ids, val_ids).size:
                raise AssertionError("Fuite : une colonne notée figure dans le train du repeat")
            if np.intersect1d(internal_val_ids, val_ids).size:
                raise AssertionError("Fuite : une colonne notée figure dans la validation interne")
            dev_ds = RetrievalDataset(train_raw, train_truth, refs, retrieval_idx,
                                      retrieval_meta, tf, dev_ids, repeat=1)
            internal_val_ds = RetrievalDataset(train_raw, train_truth, refs, retrieval_idx,
                                               retrieval_meta, tf, internal_val_ids)
            fold_train = ConcatDataset([dev_ds, synthetic_ds])
            loader = DataLoader(fold_train, batch_size=cfg.train_batch_size, shuffle=True,
                                num_workers=cfg.num_workers, pin_memory=True)
            internal_val_loader = DataLoader(internal_val_ds, batch_size=cfg.train_batch_size,
                                             shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
            fold_path = work / f"benchmark_retrieval_repeat_{repeat + 1:02d}.pt"
            training = train_fixed_epochs(
                cfg, fold_model, loader, ref_emb, device, cfg.benchmark_fold_epochs,
                f"repeat {repeat + 1}", val_loader=internal_val_loader,
                checkpoint=work / f"benchmark_retrieval_repeat_{repeat + 1:02d}_best.pt")
            torch.save({"model": fold_model.state_dict(), "seed": seed,
                        "val_cols": val_cols, "internal_val_cols": fold_internal_val_cols,
                        "training": training}, fold_path)
            val_ds = RetrievalDataset(train_raw, train_truth, refs, retrieval_idx,
                                      retrieval_meta, tf, val_ids)
            ret_errors, cat = retrieval_errors_by_column(
                fold_model, val_ds, val_cols, ref_emb, device)
            rec["retrieval"] = {c: e.tolist() for c, e in ret_errors.items()}
            rec["retrieval_categories"] = cat
            rec["retrieval_training"] = training
            rec["retrieval_seconds"] = time.time() - retrieval_started
            rec["retrieval_complete"] = True
            save_json_atomic(progress, progress_path)
            del fold_model
            gc.collect(); torch.cuda.empty_cache()

        tb = np.concatenate([np.asarray(rec["tabpfn"][c]) for c in val_cols])
        rt = np.concatenate([np.asarray(rec["retrieval"][c]) for c in val_cols])
        delta = rt.mean() - tb.mean()
        gain = 100 * (1 - rt.mean() / tb.mean())
        winner = "Retrieval" if delta < 0 else "TabPFN V14" if delta > 0 else "égalité"
        training = rec.get("retrieval_training", {})
        print(f"[repeat {repeat + 1}/{cfg.benchmark_repeats}] Retrieval: "
              f"MAE officielle={rt.mean():.2f} W, meilleure epoch interne={training.get('best_epoch')}, "
              f"MAE interne={training.get('best_val_mae')}, temps={rec.get('retrieval_seconds', 0):.1f}s\n"
              f"[repeat {repeat + 1}] Δ Retrieval-TabPFN={delta:+.2f} W, "
              f"gain={gain:+.2f}%, gagnant={winner}", flush=True)
        print_benchmark_progress(progress, cfg.benchmark_repeats, benchmark_started)

    rows, all_tb, all_rt = [], [], []
    wins = {"tabpfn": 0, "retrieval": 0, "tie": 0}
    tab_category_values: Dict[str, List[float]] = {k: [] for k in ("1", "2-3", "4-47", "48-95", ">=96")}
    ret_category_values: Dict[str, List[float]] = {k: [] for k in tab_category_values}
    for repeat in range(cfg.benchmark_repeats):
        rec = progress["repeats"][str(repeat)]
        cols = rec["val_cols"]
        tb = np.concatenate([np.asarray(rec["tabpfn"][c]) for c in cols])
        rt = np.concatenate([np.asarray(rec["retrieval"][c]) for c in cols])
        all_tb.append(tb); all_rt.append(rt)
        for c in cols:
            mt, mr = np.mean(rec["tabpfn"][c]), np.mean(rec["retrieval"][c])
            wins["retrieval" if mr < mt else "tabpfn" if mt < mr else "tie"] += 1
        for name, values in rec.get("tabpfn_categories", {}).items():
            tab_category_values[name].extend(values)
        for name, values in rec.get("retrieval_categories", {}).items():
            ret_category_values[name].extend(values)
        rows.append({"repeat": repeat + 1, "tabpfn": float(tb.mean()),
                     "retrieval": float(rt.mean()), "delta": float(rt.mean() - tb.mean())})
    pooled_tb, pooled_rt = np.concatenate(all_tb), np.concatenate(all_rt)
    result = {
        "rows": rows, "tabpfn_pooled_mae": float(pooled_tb.mean()),
        "retrieval_pooled_mae": float(pooled_rt.mean()),
        "tabpfn_repeat_mean": float(np.mean([r["tabpfn"] for r in rows])),
        "tabpfn_repeat_std": float(np.std([r["tabpfn"] for r in rows])),
        "retrieval_repeat_mean": float(np.mean([r["retrieval"] for r in rows])),
        "retrieval_repeat_std": float(np.std([r["retrieval"] for r in rows])),
        "relative_gain_percent": float(100 * (1 - pooled_rt.mean() / pooled_tb.mean())),
        "wins": wins,
        "tabpfn_mae_by_gap": {k: float(np.mean(v)) if v else None for k, v in tab_category_values.items()},
        "retrieval_mae_by_gap": {k: float(np.mean(v)) if v else None for k, v in ret_category_values.items()},
    }
    result_path = work / "paired_10x10_results.json"
    save_json_atomic(result, result_path)
    print("\n| Repeat | TabPFN V14 | Retrieval | Δ |\n|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['repeat']} | {row['tabpfn']:.2f} | {row['retrieval']:.2f} | {row['delta']:+.2f} |")
    print(f"MAE poolée: TabPFN={result['tabpfn_pooled_mae']:.2f} W, "
          f"Retrieval={result['retrieval_pooled_mae']:.2f} W")
    if abs(result["tabpfn_repeat_mean"] - 63.24) > 8.0:
        print("AVERTISSEMENT: le score TabPFN s'écarte de plus de 8 W de la référence 63,24 W.")
    return result


# %% [markdown]
# ## Inférence, adaptation légère et soumission

# %%
def adapt_target(model: RetrievalPerceiverImputer, raw: np.ndarray, refs: np.ndarray,
                 ref_indices: np.ndarray, tf: np.ndarray, ref_emb: torch.Tensor,
                 global_tokens: torch.Tensor, device: torch.device, cfg: Config) -> RetrievalPerceiverImputer:
    """Adapte seulement le gate et les experts sur des points observés artificiellement cachés."""
    if not cfg.use_test_time_adaptation:
        return model
    adapted = copy.deepcopy(model)
    for p in adapted.parameters():
        p.requires_grad_(False)
    params = list(adapted.gate.parameters()) + list(adapted.experts.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=cfg.tta_lr)
    observed_idx_np = np.flatnonzero(np.isfinite(raw))
    if len(observed_idx_np) < 20:
        return model
    adapted.train()
    for step in range(cfg.tta_steps):
        rng = np.random.default_rng(cfg.seed + step)
        take_np = rng.choice(observed_idx_np, size=max(8, len(observed_idx_np) // 20), replace=False)
        artificial = raw.copy()
        artificial[take_np] = np.nan
        # Recalcule a/b et PCHIP sans les points artificiellement cachés : aucune fuite TTA.
        idx, meta = affine_rerank_one(artificial, refs, ref_indices, len(ref_indices))
        sample_np = make_model_sample(artificial, raw, refs, idx, meta, tf)
        b = {k: (torch.from_numpy(v)[None] if isinstance(v, np.ndarray) else torch.tensor([v]))
             for k, v in sample_np.items()}
        b = to_device(b, device)
        opt.zero_grad(set_to_none=True)
        with autocast_context(device):
            pred, _ = adapted(b, ref_emb, global_tokens)
            loss = masked_l1(pred, b["truth"], b["loss_mask"])
        loss.backward()
        opt.step()
    return adapted


def predict_targets(cfg: Config, model: RetrievalPerceiverImputer, targets: np.ndarray,
                    refs: np.ndarray, indices: np.ndarray, meta: np.ndarray, tf: np.ndarray,
                    ref_emb: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    output = targets.copy().astype(np.float32)
    with torch.inference_mode(), autocast_context(device):
        global_tokens = model.global_tokens(ref_emb).detach()
    for i in tqdm(range(len(targets)), desc="imputation test"):
        raw = targets[i]
        # Vérité factice : seuls les points observés sont utilisés par la TTA ; les vrais trous restent NaN.
        sample_np = make_model_sample(raw, raw, refs, indices[i], meta[i], tf)
        sample_np["truth"] = np.where(np.isfinite(sample_np["truth"]),
                                      sample_np["truth"], sample_np["baseline"])
        sample = {k: (torch.from_numpy(v)[None] if isinstance(v, np.ndarray) else torch.tensor([v]))
                  for k, v in sample_np.items()}
        sample = to_device(sample, device)
        active = adapt_target(model, raw, refs, indices[i], tf, ref_emb,
                              global_tokens, device, cfg)
        active.eval()
        with torch.inference_mode(), autocast_context(device):
            pred_z, _ = active(sample, ref_emb, global_tokens)
        pred = pred_z[0].float().cpu().numpy() * float(sample_np["scale"]) + float(sample_np["center"])
        missing = ~np.isfinite(raw)
        output[i, missing] = pred[missing]
        output[i, ~missing] = raw[~missing]
        del active, sample
        if i % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    return output


def validate_submission(pred: pd.DataFrame, sample: pd.DataFrame, x_test_holed: pd.DataFrame) -> None:
    assert pred.shape == sample.shape, (pred.shape, sample.shape)
    assert pred.index.equals(sample.index)
    assert pred.columns.equals(sample.columns)
    assert np.isfinite(pred.to_numpy()).all()
    observed = np.isfinite(x_test_holed.to_numpy())
    if not np.allclose(pred.to_numpy()[observed], x_test_holed.to_numpy()[observed], rtol=0, atol=1e-5):
        raise AssertionError("Les valeurs observées ont été modifiées")


def fingerprint_frames(x_train: pd.DataFrame, x_test: pd.DataFrame) -> str:
    payload = f"{x_train.shape}|{x_test.shape}|{x_train.columns[0]}|{x_test.columns[-1]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# %% [markdown]
# ## Pipeline principal

# %%
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full", "predict"), default="full")
    parser.add_argument(
        "--data-dir", default=None,
        help="Dossier des CSV; avec !python, téléverser d'abord dans une cellule puis utiliser /content",
    )
    parser.add_argument("--work-dir", default="/content/enedis_retrieval_work" if Path("/content").exists() else "claude2/retrieval_work")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    cfg = Config.for_profile("full" if args.profile == "predict" else args.profile)
    cfg.use_test_time_adaptation &= not args.no_tta
    seed_everything(cfg.seed)
    if not torch.cuda.is_available() and cfg.profile == "full":
        raise RuntimeError("Le profil full requiert CUDA. Choisis une A100/H100 80 Go dans Colab.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, VRAM={props.total_memory / 2**30:.1f} GiB")

    root = upload_data_if_needed(args.data_dir)
    # Sépare les caches incompatibles par dimensions ET régime d'entraînement.
    namespace = config_namespace(cfg)
    work = Path(args.work_dir) / namespace
    work.mkdir(parents=True, exist_ok=True)
    print(
        "Configuration Retrieval:\n"
        f"  namespace={namespace}\n"
        f"  embed/latent={cfg.embed_dim}/{cfg.latent_dim}, temporal={cfg.temporal_channels}, "
        f"latents={cfg.n_latents}, local_channels={cfg.local_feature_channels}, "
        f"decoder_layers={cfg.decoder_layers}, heads={cfg.n_heads}\n"
        f"  batch prétrain={cfg.pretrain_batch_size}, batch train={cfg.train_batch_size}, "
        f"accumulation={cfg.grad_accum}, batch effectif={cfg.train_batch_size * cfg.grad_accum}\n"
        f"  epochs prétrain/finetune/final={cfg.pretrain_epochs}/{cfg.finetune_epochs}/{cfg.final_epochs}, "
        f"benchmark base/fold={cfg.benchmark_base_epochs}/{cfg.benchmark_fold_epochs}, "
        f"Perceiver={cfg.global_pretrain_steps} steps",
        flush=True,
    )
    x_train, y_train, x_test, sample = load_frames(root)
    cfg.n_steps = len(x_train)
    train_ref_cols, test_ref_cols = complete_columns(x_train), complete_columns(x_test)
    train_targets_cols = list(y_train.columns)
    test_targets_cols = list(sample.columns)
    print(f"train refs={len(train_ref_cols)}, test refs={len(test_ref_cols)}, "
          f"train targets={len(train_targets_cols)}, test targets={len(test_targets_cols)}")

    # Courbes en lignes : (N, T). La banque n'inclut jamais y_train.
    ref_raw = np.concatenate([
        x_train[train_ref_cols].to_numpy(np.float32).T,
        x_test[test_ref_cols].to_numpy(np.float32).T,
    ], axis=0)
    ref_z, _, _ = robust_curve_normalize(ref_raw)
    train_raw = x_train[train_targets_cols].to_numpy(np.float32).T
    train_truth = y_train[train_targets_cols].to_numpy(np.float32).T
    test_raw = x_test[test_targets_cols].to_numpy(np.float32).T
    gaps = extract_gap_lengths(x_train[train_targets_cols])
    tf = time_features(x_train.index)
    print("Banque transductive:", ref_raw.shape, "distribution gaps:",
          {"<=3": int((gaps <= 3).sum()), "4-47": int(((gaps >= 4) & (gaps <= 47)).sum()),
           "48-95": int(((gaps >= 48) & (gaps <= 95)).sum()), ">=96": int((gaps >= 96).sum())})

    manifest = {"fingerprint": fingerprint_frames(x_train, x_test), "config": asdict(cfg),
                "reference_count": len(ref_raw)}
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    encoder = SharedCurveEncoder(cfg)
    pretrain_encoder(cfg, encoder, ref_raw, gaps, device, work)
    ref_embeddings_np = encode_reference_bank(cfg, encoder, ref_z, device, work)
    encoder.cpu()
    del encoder, ref_z
    gc.collect()
    torch.cuda.empty_cache()

    # Recharge l'encodeur pré-entraîné pour encoder les targets sans le garder ensuite en VRAM.
    target_encoder = SharedCurveEncoder(cfg)
    encoder_saved = torch.load(work / "encoder_pretrained.pt", map_location="cpu", weights_only=False)
    target_encoder.load_state_dict(
        encoder_saved["model"] if isinstance(encoder_saved, dict) and "model" in encoder_saved else encoder_saved)
    target_encoder.to(device).eval()
    train_idx, train_meta = build_retrieval_cache("train", cfg, target_encoder, train_raw,
                                                   ref_raw, ref_embeddings_np, device, work)
    test_idx, test_meta = build_retrieval_cache("test", cfg, target_encoder, test_raw,
                                                 ref_raw, ref_embeddings_np, device, work)
    del target_encoder
    torch.cuda.empty_cache()

    all_ids = np.arange(len(train_raw))
    fit_ids, val_ids = train_test_split(all_ids, test_size=cfg.val_fraction, random_state=cfg.seed)
    supervised_ds = RetrievalDataset(train_raw, train_truth, ref_raw, train_idx, train_meta, tf,
                                     fit_ids, repeat=cfg.supervised_repeat)
    rng = np.random.default_rng(cfg.seed)
    n_synth = min(cfg.synthetic_samples_per_epoch, len(ref_raw))
    synth_ids = rng.choice(len(ref_raw), size=n_synth, replace=False).astype(np.int32)
    synth_cache = work / f"synthetic_candidates_{n_synth}_k{cfg.k_local}.npy"
    if synth_cache.exists():
        synth_candidates = np.load(synth_cache)
    else:
        # k+1 garantit qu'il reste k voisins après retrait de la cible elle-même.
        synth_candidates = coarse_neighbors(ref_embeddings_np[synth_ids], ref_embeddings_np,
                                             cfg.k_local + 1, device)
        np.save(synth_cache, synth_candidates)
    synth_masked, synth_indices, synth_meta = build_synthetic_cache(
        cfg, ref_raw, synth_ids, synth_candidates, gaps, work)
    synthetic_ds = SyntheticRetrievalDataset(ref_raw, synth_ids, synth_masked,
                                              synth_indices, synth_meta, tf)
    train_ds = ConcatDataset([supervised_ds, synthetic_ds])
    val_ds = RetrievalDataset(train_raw, train_truth, ref_raw, train_idx, train_meta, tf, val_ids)
    train_loader = DataLoader(train_ds, batch_size=cfg.train_batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train_batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    def updates_per_epoch(n_samples: int) -> int:
        batches = math.ceil(n_samples / cfg.train_batch_size)
        return math.ceil(batches / cfg.grad_accum)
    main_updates = updates_per_epoch(len(train_ds)) * cfg.finetune_epochs
    final_samples = len(synthetic_ds) + len(all_ids) * max(1, cfg.supervised_repeat // 2)
    final_updates = updates_per_epoch(final_samples) * cfg.final_epochs
    benchmark_updates = (
        updates_per_epoch(len(synthetic_ds)) * cfg.benchmark_base_epochs
        + cfg.benchmark_repeats
        * updates_per_epoch(len(synthetic_ds) + 900) * cfg.benchmark_fold_epochs
    ) if cfg.profile == "full" else 0
    print(
        "Budget maximal d'optimisation: "
        f"finetune={main_updates:,}, final={final_updates:,}, "
        f"benchmark={benchmark_updates:,}, total={main_updates + final_updates + benchmark_updates:,} updates",
        flush=True,
    )
    ref_emb = torch.from_numpy(ref_embeddings_np)
    model = RetrievalPerceiverImputer(cfg)
    final_ckpt = work / "retrieval_perceiver_final.pt"

    if args.profile == "predict":
        if not final_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint final absent: {final_ckpt}")
        model.load_state_dict(torch.load(final_ckpt, map_location="cpu", weights_only=False)["model"])
    else:
        pretrain_global_perceiver(cfg, model, ref_emb, device, work)
        if args.profile == "full":
            print("\n=== BENCHMARK APPARIÉ STRICT RETRIEVAL vs TABPFN V14 (10×10) ===")
            run_paired_benchmark(
                cfg, model, x_train, y_train, train_raw, train_truth, ref_raw,
                train_idx, train_meta, tf, synthetic_ds, ref_emb, device, work,
            )
            print("\n=== BENCHMARK TERMINÉ — ENTRAÎNEMENT FINAL SUR 1 000 TARGETS ===")
        best = train_imputer(cfg, model, train_loader, val_loader, ref_emb, device, work,
                             cfg.finetune_epochs, "retrieval_perceiver_best.pt")
        print(f"Meilleure MAE holdout: {best:.3f} W")
        # Sélectionne d'abord la durée finale sur 900/100, sans perdre ensuite
        # les 100 courbes : le refit repart du meilleur modèle principal et
        # utilise les 1 000 targets pendant le nombre d'epochs sélectionné.
        main_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        final_train_ids, final_val_ids = train_test_split(
            all_ids, test_size=0.1, random_state=cfg.seed + 200_000)
        final_train_targets = RetrievalDataset(
            train_raw, train_truth, ref_raw, train_idx, train_meta, tf, final_train_ids, repeat=1)
        final_val_targets = RetrievalDataset(
            train_raw, train_truth, ref_raw, train_idx, train_meta, tf, final_val_ids)
        final_selection_train = DataLoader(
            ConcatDataset([final_train_targets, synthetic_ds]), batch_size=cfg.train_batch_size,
            shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        final_selection_val = DataLoader(
            final_val_targets, batch_size=cfg.train_batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True)
        selection_model = copy.deepcopy(model)
        final_cfg = copy.deepcopy(cfg)
        final_cfg.lr_finetune = cfg.lr_finetune * 0.35
        final_selection = train_fixed_epochs(
            final_cfg, selection_model, final_selection_train, ref_emb, device,
            cfg.final_epochs, "sélection finale", val_loader=final_selection_val,
            checkpoint=work / "retrieval_perceiver_final_selection_best.pt")
        selected_epochs = max(1, int(final_selection["best_epoch"]))
        print(f"Refit final sur 1 000 targets pendant {selected_epochs} epochs sélectionnées")
        del selection_model
        model.load_state_dict(main_state)
        full_targets = RetrievalDataset(
            train_raw, train_truth, ref_raw, train_idx, train_meta, tf,
            all_ids, repeat=max(1, cfg.supervised_repeat // 2))
        full_loader = DataLoader(
            ConcatDataset([full_targets, synthetic_ds]), batch_size=cfg.train_batch_size,
            shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        refit = train_fixed_epochs(
            final_cfg, model, full_loader, ref_emb, device, selected_epochs, "refit final")
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "holdout_mae": best,
                    "final_selection": final_selection, "final_refit": refit}, final_ckpt)

    model.to(device).eval()
    ref_emb = ref_emb.to(device)
    predictions = predict_targets(cfg, model, test_raw, ref_raw, test_idx, test_meta, tf,
                                  ref_emb, device)
    pred_df = pd.DataFrame(predictions.T, index=x_test.index, columns=test_targets_cols)
    pred_df = pred_df[sample.columns]
    validate_submission(pred_df, sample, x_test[sample.columns])
    submission = work / "submission_retrieval_perceiver_t1.csv"
    pred_df.to_csv(submission)
    print("Soumission valide écrite:", submission)

    # Le zip doit rester HORS de `work` : le mettre dedans le fait s'inclure
    # lui-même pendant l'écriture (croissance auto-référentielle observée
    # jusqu'à 29 Go en test local avant d'être tué).
    archive = shutil.make_archive(str(work.parent / f"{namespace}_artifacts"), "zip", root_dir=work)
    print("Archive:", archive)
    try:
        from google.colab import files  # type: ignore
        files.download(str(submission))
    except ImportError:
        pass


if __name__ == "__main__":
    main()

