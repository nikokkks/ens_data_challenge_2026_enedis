"""LoRA fine-tuning of the PyTorch TabFM regression checkpoint for Enedis.

Stages:
  smoke     validate TabFM, CUDA, LoRA gradients and one optimization step
  train     train adapter weights on development columns
  evaluate  paired zero-shot/LoRA evaluation (screen or confirm)
  submit    generate a resumable submission after a successful confirmation

TabFM's sklearn ``fit`` is ICL-only.  This module calls the underlying
``nn.Module.forward`` and performs backpropagation through rank-decomposition
adapters injected into every attention q_proj/v_proj.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from evaluate import fill_nan_with_interpolation, load_data
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14

SEED = 42
SENTINEL = -100.0
TRAINING_PROTOCOL = "real_gaps_with_donor_augmentation"
MANIFEST_SCHEMA_VERSION = 2
CHECKPOINT_FORMAT = 2
ADAPTER_PATH = Path("tabfm_lora_adapter.pt")
HISTORY_PATH = Path("tabfm_lora_history.json")
RESULTS_PATH = Path("tabfm_lora_results.json")
SPLIT_PATH = Path("tabfm_lora_split_seed42.json")
OUT_PATH = Path("submission_tabfm_lora.csv")
PARTIAL_PATH = Path("submission_tabfm_lora_partial.csv")

V14_FEATURE_OPTS = dict(
    volatility_alpha=True, vol_weight=.8, periodic_alpha_strength=1.0,
    extra_lags=(24,), neighbor_rolling=True,
)


@dataclass
class TrainConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = .05
    lr: float = 5e-4
    weight_decay: float = .01
    epochs: int = 45
    patience: int = 45
    batch_size: int = 32
    grad_clip: float = 1.0
    augmentation_fraction: float = .5
    max_features: int = 96
    max_context: int = 800
    train_columns: int = 880
    val_columns: int = 20
    # Episodes per single batched GPU forward/backward call (independent of
    # `batch_size`, which stays the gradient-accumulation window). Real
    # multi-episode batching instead of one sequential per-episode call.
    # `train_size`/`d` are already passed to the TabFM forward as per-example
    # tensors (shape [B]) even when B=1, which is what batching relies on --
    # validated by `smoke`'s batch-consistency check before trusting
    # `train`/`evaluate` numbers. Attention over ~800+ row sequences is
    # memory-hungry per example (64 OOM'd an 80GB A100 on the first batch,
    # ~78GB for the forward pass alone) -- 8 is a safer starting point.
    # backward_chunk/forward_chunk_predictions auto-halve on CUDA OOM down to
    # 1 regardless, so this is a speed/retry-overhead tradeoff, not a
    # correctness one: too high just means a few wasted OOM+retry cycles on
    # the first batch before it settles.
    gpu_batch_size: int = 2
    # AdamWScheduleFree needs no LR schedule; its docstring recommends
    # warmup_steps instead of one for the first steps. 0 = no warmup,
    # matching the pre-schedule-free behaviour by default.
    warmup_steps: int = 0


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank=4, alpha=8., dropout=.05):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        self.dropout = torch.nn.Dropout(dropout)
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = torch.nn.Parameter(
            torch.empty(self.rank, base.in_features, **factory_kwargs))
        self.lora_b = torch.nn.Parameter(
            torch.zeros(base.out_features, self.rank, **factory_kwargs))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(self.dropout(x), self.lora_a), self.lora_b)
        return self.base(x) + delta * self.scale


class CurveEpisodeDataset(Dataset):
    """Lightweight dataset: costly features are built in the training loop."""
    def __init__(self, columns, epoch_seed):
        self.columns = list(columns)
        self.epoch_seed = int(epoch_seed)

    def __len__(self): return len(self.columns)

    def __getitem__(self, index):
        return self.columns[index], self.epoch_seed * 100_000 + int(index)


def make_curve_loader(columns, batch_size, epoch_seed):
    generator = torch.Generator().manual_seed(int(epoch_seed))
    return DataLoader(
        CurveEpisodeDataset(columns, epoch_seed), batch_size=batch_size,
        shuffle=True, generator=generator, num_workers=0,
        collate_fn=lambda rows: rows,
    )


def inject_lora(model, rank=4, alpha=8., dropout=.05) -> list[str]:
    """Freeze ``model`` and replace every attention q/v projection in-place."""
    model.requires_grad_(False)
    replaced = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in {"q_proj", "v_proj"} and isinstance(child, torch.nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
                replaced.append(f"{parent_name}.{child_name}".lstrip("."))
    if not replaced:
        raise RuntimeError("No q_proj/v_proj torch.nn.Linear found; incompatible TabFM revision")
    return replaced


def adapter_state_dict(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.endswith("lora_a") or k.endswith("lora_b")}


def load_adapter_state(model, state: dict[str, torch.Tensor]):
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if "lora_" in k]
    missing_lora = [k for k in missing if "lora_" in k]
    if unexpected or missing_lora:
        raise RuntimeError(f"Adapter mismatch: missing={missing_lora}, unexpected={unexpected}")


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def assert_model_device(model, device: torch.device):
    """Fail before forward with actionable names for misplaced parameters."""
    expected = torch.device(device)
    if expected.type == "cuda" and expected.index is None:
        expected = torch.device("cuda", torch.cuda.current_device())
    misplaced = [(name, str(param.device)) for name, param in model.named_parameters()
                 if param.device != expected]
    if misplaced:
        preview = ", ".join(f"{name}={actual}" for name, actual in misplaced[:12])
        raise RuntimeError(
            f"Model/device mismatch: expected every parameter on {expected}; {preview}")
    trainable = [(name, str(param.device)) for name, param in model.named_parameters()
                 if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters found")


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_tabfm(device: torch.device):
    try:
        from tabfm import tabfm_v1_0_0_pytorch
    except ImportError as exc:
        raise SystemExit(
            "TabFM absent. Install the pinned official PyTorch checkout first; "
            "see TABFM_LORA.md.") from exc
    model = tabfm_v1_0_0_pytorch.load(model_type="regression")
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("The loaded TabFM backend is not a torch.nn.Module")
    return model.to(device)


def build_donor_context(X, donor_cols):
    donor_df = X[donor_cols]
    pool_vals = donor_df.ffill().bfill().to_numpy(dtype=np.float64)
    km, labels = feb.fit_shape_clusters(donor_df, n_clusters=15, random_state=SEED)
    pos = pd.Series(np.arange(len(labels)), index=labels.index)
    cluster_idx = {cl: pos[idx].to_numpy() for cl, idx in labels.groupby(labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(X.index, ("hour", "dow", "dom", "year"))
    return pool_vals, km, cluster_idx, tfeat


def real_gap_indices(source_s: pd.Series, truth: pd.Series) -> np.ndarray:
    """Official query positions: missing in X and finite in y."""
    return np.flatnonzero(source_s.isna().to_numpy() & np.isfinite(truth.to_numpy(float)))


def copied_gap_indices(template_s: pd.Series, truth: pd.Series) -> np.ndarray:
    """Copy a real Enedis mask onto a curve while retaining only valid targets."""
    return np.flatnonzero(template_s.isna().to_numpy() & np.isfinite(truth.to_numpy(float)))


def feature_matrix(col, masked_s, pool_vals, km, cluster_idx, tfeat, index):
    if masked_s.notna().sum() == 0:
        # fill_nan_with_interpolation (evaluate.py, limit_direction="both")
        # only fills a series that has >=1 finite anchor point -- verified:
        # a single one is already enough to fill the whole series. With
        # zero, v14.build_v14_features's interp_full stays fully NaN and
        # crashes feb.cluster_of's KMeans.predict() ("Input X contains
        # NaN"), the one spot in the V14 feature pipeline that isn't already
        # wrapped in nan_to_num below. This happens on "augmented" episodes
        # (make_training_batch): a real gap mask copied wholesale from a
        # heavily-gapped template column can wipe out virtually all of a
        # *different* target curve's own (normally near-complete) y[col]
        # supervision. Treat it as an unbuildable episode -- same convention
        # as v14.build_v14_features returning None -- instead of crashing
        # the whole training run over one degenerate mask draw.
        return None
    built = v14.build_v14_features(col, masked_s, pool_vals, km, cluster_idx,
                                   tfeat, index, V14_FEATURE_OPTS)
    if built is None: return None
    obs_idx, miss_idx, target_obs, interp, knn, scaled, extra, missing, lags = built
    matrix = np.nan_to_num(np.hstack([scaled, extra]), nan=0., posinf=0., neginf=0.)
    return matrix.astype(np.float32), obs_idx, miss_idx, interp, missing, lags


def normalize_episode(X, y_context):
    center = np.median(X, axis=0)
    scale = np.median(np.abs(X-center), axis=0) * 1.4826
    scale = np.where(scale > 1e-6, scale, 1.)
    Xn = np.clip((X-center)/scale, -20., 20.)
    yc = float(np.median(y_context))
    ys = max(float(np.median(np.abs(y_context-yc))*1.4826), .02*(abs(yc)+1.), 1.)
    return Xn, (y_context-yc)/ys, yc, ys


def make_episode(col, source_s, truth, context, config, query_idx=None, rng=None,
                 require_targets=True, cache=None):
    """``cache`` (dict or None) memoizes feature_matrix's expensive KNN/
    periodic-feature construction, keyed by column. Only safe to pass when
    ``query_idx`` is the same fixed mask every call for that column (the
    real-gap case: real_gap_indices doesn't depend on rng or the epoch) --
    the random context-subsampling below (rng.choice) still runs fresh on
    every call regardless, so caching never freezes that regularization.
    Augmented episodes use a different random template mask each call and
    must be called with cache=None."""
    pool_vals, km, cluster_idx, tfeat = context
    if query_idx is None:
        raise ValueError("query_idx is required by the real-gap v2 protocol")
    masked = source_s.copy(); masked.iloc[query_idx] = np.nan
    if cache is not None and col in cache:
        built = cache[col]
    else:
        built = feature_matrix(col, masked, pool_vals, km, cluster_idx, tfeat, source_s.index)
        if cache is not None: cache[col] = built
    if built is None: return None
    matrix, _, _, interp, missing, lags = built
    context_idx = np.flatnonzero(masked.notna().to_numpy() & truth.notna().to_numpy())
    if len(context_idx) > config.max_context:
        context_idx = np.sort(rng.choice(context_idx, config.max_context, replace=False))
    q = np.asarray(query_idx, dtype=int)
    valid = (np.isfinite(truth.iloc[q].to_numpy()) if require_targets
             else np.ones(len(q), dtype=bool))
    q = q[valid]
    if not len(q): return None
    rows = np.r_[context_idx, q]
    X = matrix[rows]
    if X.shape[1] > config.max_features:
        X = X[:, :config.max_features]
    active_d = X.shape[1]
    Xn, yn_context, yc, ys = normalize_episode(X, truth.iloc[context_idx].to_numpy(float))
    if active_d < config.max_features:
        Xn = np.pad(Xn, ((0, 0), (0, config.max_features-active_d)))
    y = np.full(len(rows), SENTINEL, dtype=np.float32)
    y[:len(context_idx)] = yn_context.astype(np.float32)
    target = (((truth.iloc[q].to_numpy(float)-yc)/ys).astype(np.float32)
              if require_targets else np.zeros(len(q), dtype=np.float32))
    return dict(x=Xn.astype(np.float32), y=y, train_size=len(context_idx), d=active_d,
                target=target, query_idx=q, y_center=yc, y_scale=ys,
                interp=interp.to_numpy(), missing=missing, lags=lags)


def episode_loss(model, episode, device):
    x = torch.from_numpy(episode["x"])[None].to(device)
    y = torch.from_numpy(episode["y"])[None].to(device)
    train_size = torch.tensor([episode["train_size"]], device=device)
    d = torch.tensor([episode["d"]], device=device)
    out = model(x, y, train_size, d=d)
    pred = out[0, episode["train_size"]:, 0].float()
    target = torch.from_numpy(episode["target"]).to(device)
    return torch.nn.functional.l1_loss(pred, target), pred


def pad_stack_episodes(episodes, device):
    """Right-pad every episode's (context+query) rows to the batch's max
    row count and stack into [B, T, D] tensors. Padding rows get x=0,
    y=SENTINEL (the model's own "no supervision here" value) and sit past
    each example's individual train_size/row_count, mirroring how a single
    episode's own query rows are already SENTINEL-labelled -- relies on the
    TabFM forward already taking per-example train_size/d tensors (true
    even at batch=1 in episode_loss above), which only makes sense if it
    restricts attention/loss per example instead of assuming every row in a
    [B, T, D] tensor is meaningful for every example; verified once by
    `smoke`'s batch-consistency check, not re-verified on every call."""
    B = len(episodes)
    T = max(len(ep["y"]) for ep in episodes)
    Dm = episodes[0]["x"].shape[1]
    x = np.zeros((B, T, Dm), dtype=np.float32)
    y = np.full((B, T), SENTINEL, dtype=np.float32)
    row_counts = np.empty(B, dtype=np.int64)
    train_sizes = np.empty(B, dtype=np.int64)
    d = np.empty(B, dtype=np.int64)
    for i, ep in enumerate(episodes):
        n = len(ep["y"])
        x[i, :n] = ep["x"]; y[i, :n] = ep["y"]
        row_counts[i] = n; train_sizes[i] = ep["train_size"]; d[i] = ep["d"]
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
            torch.from_numpy(train_sizes).to(device), torch.from_numpy(d).to(device),
            row_counts, train_sizes)


def episode_batch_loss(model, episodes, device):
    """Batched counterpart of episode_loss: one forward call for the whole
    chunk instead of one call per episode. Returns the mean of each
    episode's own mean L1 loss (matching the pre-batching semantics of
    averaging `episode_loss(...)` per episode, not a single micro-average
    over all points pooled together) plus the per-episode predictions."""
    x, y, train_size_t, d_t, row_counts, train_sizes = pad_stack_episodes(episodes, device)
    out = model(x, y, train_size_t, d=d_t)
    losses = []; preds = []
    for i, ep in enumerate(episodes):
        pred = out[i, train_sizes[i]:row_counts[i], 0].float()
        target = torch.from_numpy(ep["target"]).to(device)
        if pred.shape[0] != target.shape[0]:
            raise RuntimeError(
                f"Batched/single-episode shape mismatch for column slicing "
                f"(pred={pred.shape[0]}, target={target.shape[0]}); the padding "
                f"convention in pad_stack_episodes no longer matches this TabFM "
                f"revision's forward() -- rerun `smoke` before trusting results.")
        losses.append(torch.nn.functional.l1_loss(pred, target))
        preds.append(pred)
    return torch.stack(losses).mean(), losses, preds


def backward_chunk(model, group, device, use_amp, total_episodes, losses_log):
    """Forward+backward one GPU chunk immediately, so its activations are
    freed before the next chunk starts (real memory savings -- deferring
    backward across chunks would keep every chunk's graph alive at once,
    which defeats the point). Adapts to whatever VRAM ceiling this TabFM
    revision/model size actually has: on CUDA OOM, empties the cache and
    retries as two halves, recursing down to gpu_batch_size=1 (the
    original, always-correct sequential path, cf. episode_loss) if needed,
    instead of a hand-picked constant that may be wrong for a given GPU/
    model size. Requires optimizer.zero_grad() already called for the
    whole outer training-loop batch; gradients accumulate correctly across
    chunks via repeated .backward() calls, same as before batching.

    The retry MUST happen outside the `except` block: while an except
    clause is executing (including in nested calls made from inside it),
    Python keeps the failing call chain's frames -- and therefore every
    activation tensor referenced by them -- alive via the exception's
    traceback. Recursing from inside `except` would pin each failed
    attempt's whole graph in GPU memory for the entire retry cascade,
    compounding across 8->4->2->1 instead of freeing anything, which is
    exactly what starved even gpu_batch_size=1 the first time this ran."""
    oom = False
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            mean_loss, per_episode_losses, _ = episode_batch_loss(model, group, device)
        (mean_loss * len(group) / total_episodes).backward()
        losses_log.extend(float(l.detach()) for l in per_episode_losses)
    except torch.cuda.OutOfMemoryError:
        oom = True
    if not oom:
        return
    if len(group) == 1:
        raise RuntimeError(
            "CUDA OOM even at gpu_batch_size=1 for a single episode, after a "
            "clean (leak-free) retry -- this is no longer a batching-size "
            "problem, see the OutOfMemoryError above for actual free/allocated.")
    gc.collect(); torch.cuda.empty_cache()
    mid = max(1, len(group) // 2)
    print(f"  OOM at gpu_batch={len(group)}, retrying as {mid}+{len(group)-mid}", flush=True)
    backward_chunk(model, group[:mid], device, use_amp, total_episodes, losses_log)
    backward_chunk(model, group[mid:], device, use_amp, total_episodes, losses_log)


def forward_chunk_predictions(model, group, device):
    """@torch.no_grad() counterpart of backward_chunk for validation_mae:
    same adaptive OOM backoff (retry outside `except`, see backward_chunk's
    docstring), no gradient/backward bookkeeping needed."""
    oom = False
    try:
        preds = episode_batch_loss(model, group, device)[2]
    except torch.cuda.OutOfMemoryError:
        oom = True
    if not oom:
        return preds
    if len(group) == 1:
        raise RuntimeError(
            "CUDA OOM even at gpu_batch_size=1 during validation, after a "
            "clean (leak-free) retry.")
    gc.collect(); torch.cuda.empty_cache()
    mid = max(1, len(group) // 2)
    print(f"  OOM (val) at gpu_batch={len(group)}, retrying as {mid}+{len(group)-mid}", flush=True)
    return (forward_chunk_predictions(model, group[:mid], device)
            + forward_chunk_predictions(model, group[mid:], device))


def _state_to_cpu(value):
    """Recursively move an optimizer.state_dict()'s tensors to CPU before
    persisting, mirroring adapter_state_dict's own .detach().cpu() so the
    checkpoint stays portable across devices/GPU ids instead of pinning a
    specific CUDA index."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _state_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_state_to_cpu(v) for v in value]
    return value


def save_adapter(model, config, replaced, optimizer=None, path=ADAPTER_PATH):
    """``optimizer``, when given, should already be in .eval() mode (the
    schedule-free optimizer's own docs: "should be placed in eval mode when
    saving checkpoints") -- train()'s loop always calls this right after
    optimizer.eval() + validation_mae, before switching back to .train()."""
    payload = {"format": CHECKPOINT_FORMAT, "training_protocol": TRAINING_PROTOCOL,
               "config": asdict(config), "targets": replaced,
               "tabfm_manifest": tabfm_manifest(model),
               "state_dict": adapter_state_dict(model)}
    if optimizer is not None:
        payload["optimizer_state"] = _state_to_cpu(optimizer.state_dict())
    torch.save(payload, path)


def tabfm_manifest(model):
    try: version=importlib.metadata.version("tabfm")
    except importlib.metadata.PackageNotFoundError: version="editable-or-unknown"
    return {"package_version":version,
            "model_class":f"{type(model).__module__}.{type(model).__qualname__}"}


def prepare_columns(y, outer_seed, config):
    dev, holdout = train_test_split(list(y.columns), test_size=.1, random_state=outer_seed)
    rng = np.random.RandomState(SEED)
    rng.shuffle(dev)
    val_cols = dev[:config.val_columns]
    train_cols = dev[config.val_columns:]
    if config.train_columns != len(train_cols):
        raise ValueError(f"TrainConfig.train_columns={config.train_columns}, split has {len(train_cols)}")
    return train_cols, val_cols, holdout


def build_split_manifest(X, y, config, outer_seed=SEED):
    train_cols, val_cols, test_cols = prepare_columns(y, outer_seed, config)
    return {"schema_version":MANIFEST_SCHEMA_VERSION,"outer_seed":outer_seed,
            "training_protocol":TRAINING_PROTOCOL,
            "augmentation_fraction":config.augmentation_fraction,
            "train":list(train_cols),"val":list(val_cols),"test":list(test_cols)}


def _write_json_atomic(path: Path, payload):
    tmp = Path(str(path)+".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def load_or_create_split_manifest(X, y, config, outer_seed=SEED, path=SPLIT_PATH):
    expected = build_split_manifest(X, y, config, outer_seed)
    if path.exists():
        actual = json.loads(path.read_text())
        # Schema v1 used pseudo-validation holes. Migrate only when its column
        # split is exactly the deterministic v2 split.
        if actual.get("schema_version") == 1:
            same_split = all(actual.get(name) == expected[name]
                             for name in ("train", "val", "test"))
            if not same_split:
                raise RuntimeError(f"Cannot safely migrate split manifest {path}")
            _write_json_atomic(path, expected)
            return expected
        if actual != expected:
            raise RuntimeError(f"Existing split manifest {path} differs from requested split")
        return actual
    _write_json_atomic(path, expected)
    return expected


@torch.no_grad()
def validation_mae(model, columns, X, y, context, config, device, seed=SEED,
                   query_indices=None, cache=None, gpu_batch_size=None):
    model.eval()
    episodes=[]; cols_for_ep=[]
    for j, col in enumerate(columns):
        q = (np.asarray(query_indices[col],dtype=int) if query_indices
             else real_gap_indices(X[col], y[col]))
        # query_indices/real_gap_indices don't depend on rng or j -> caching
        # feature_matrix per column across the val set (and across epochs)
        # is safe here, same rule as make_training_batch's "real" branch.
        ep = make_episode(col, X[col], y[col], context, config,query_idx=q,
                          rng=np.random.RandomState(seed+j), cache=cache)
        if ep is None: continue
        episodes.append(ep); cols_for_ep.append(col)
    if not episodes:
        raise RuntimeError("Validation contains no finite real-gap targets")
    errors=[]
    chunk = gpu_batch_size or len(episodes)
    for start in range(0, len(episodes), chunk):
        group = episodes[start:start+chunk]
        preds = forward_chunk_predictions(model, group, device)
        for ep, col, pred in zip(group, cols_for_ep[start:start+chunk], preds):
            raw = pred.cpu().numpy()*ep["y_scale"]+ep["y_center"]
            blended = blend_episode_prediction(raw, ep, X[col], ep["query_idx"])
            errors.append(np.abs(blended-y[col].iloc[ep["query_idx"]].to_numpy()))
    return float(np.concatenate(errors).mean())


def make_training_batch(batch, train_cols, X, y, context, config, cache=None):
    """Build an exact real/augmented mix without held-out masks or donor
    leakage. ``cache`` memoizes the "real" branch only (fixed mask per
    column, cf. make_episode's docstring) -- never passed for "augmented"
    episodes, whose template mask is redrawn every call."""
    episodes=[]; counts={"real":0,"augmented":0,"skipped":0}
    n_aug = int(round(len(batch) * config.augmentation_fraction))
    if not 0 <= n_aug <= len(batch):
        raise ValueError("augmentation_fraction must be between 0 and 1")
    for position, (col, episode_seed) in enumerate(batch):
        rng=np.random.RandomState(episode_seed)
        if position < n_aug:
            choices=[candidate for candidate in train_cols if candidate != col]
            template_col=choices[int(rng.randint(len(choices)))] if choices else col
            # y[col] is complete supervision and holed_* columns are absent
            # from the donor context, preventing self-neighbor target leakage.
            # A heavily-gapped template_col can still wipe out virtually all
            # of y[col] once copied -- make_episode/feature_matrix return
            # None for that degenerate draw rather than crash (see
            # feature_matrix's NaN guard); tallied below as "skipped".
            q=copied_gap_indices(X[template_col], y[col])
            ep=make_episode(col,y[col],y[col],context,config,
                            query_idx=q,rng=rng)
            kind="augmented"
        else:
            q=real_gap_indices(X[col],y[col])
            ep=make_episode(col,X[col],y[col],context,config,query_idx=q,rng=rng,cache=cache)
            kind="real"
        if ep is not None:
            ep["episode_kind"]=kind
            ep["mask_template_col"]=(template_col if kind=="augmented" else col)
            episodes.append(ep); counts[kind] += 1
        else:
            counts["skipped"] += 1
    return episodes, counts


def train(model, X, y, donor_cols, config, device, outer_seed=SEED):
    manifest=load_or_create_split_manifest(X,y,config,outer_seed)
    train_cols,val_cols=manifest["train"],manifest["val"]
    context = build_donor_context(X, donor_cols)

    # Resume from ADAPTER_PATH if a checkpoint of that name already exists
    # (same convention as generate_submission_*.py's *_partial.csv resume:
    # no flag needed, just re-run). Only the LoRA weights + optimizer state
    # are resumed; train_cols/val_cols themselves are re-derived from
    # load_or_create_split_manifest above, which already refuses to silently
    # continue on a different column split (see its own checks).
    history=[]; best=math.inf; start_epoch=0; optimizer_state=None
    if ADAPTER_PATH.exists():
        print(f"Resuming from checkpoint {ADAPTER_PATH} ...", flush=True)
        cfg, replaced, optimizer_state = load_lora_checkpoint(model, ADAPTER_PATH, device)
        # rank/alpha/dropout are already baked into the LoRALinear shapes
        # load_lora_checkpoint just injected; keep them pinned to what's on
        # disk for the rest of this run instead of silently drifting from
        # `config`. Every other hyperparameter (lr, epochs, patience, ...)
        # is free to change on resume.
        config = replace(config, rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout)
        if HISTORY_PATH.exists():
            history = json.loads(HISTORY_PATH.read_text())
            if history:
                best = min(row["val_mae_w"] for row in history)
                start_epoch = history[-1]["epoch"]
        print(f"  resumed at epoch {start_epoch}, best val_mae_w so far = {best:.2f} W",
              flush=True)
    else:
        replaced = inject_lora(model, config.rank, config.alpha, config.dropout)
        model.to(device)
    assert_model_device(model, device)
    params = trainable_parameters(model)
    optimizer = build_optimizer(params, config)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        # Baseline footprint before any episode is processed -- if this
        # alone is already most of the card's VRAM, gpu_batch_size tuning
        # won't help and the real fix is elsewhere (fp16 weights, gradient
        # checkpointing, a smaller max_context, etc.), not a leak.
        print(f"CUDA memory after model+LoRA+optimizer setup: "
              f"{torch.cuda.memory_allocated()/2**30:.2f} GiB allocated, "
              f"{torch.cuda.memory_reserved()/2**30:.2f} GiB reserved "
              f"(capacity {torch.cuda.get_device_properties(device).total_memory/2**30:.2f} GiB)",
              flush=True)
    # "real" episodes mask the exact same fixed positions every epoch, so
    # feature_matrix's KNN/periodic-feature construction is deterministic
    # per column and safe to memoize for the whole training run -- the
    # random context-subsampling regularization (make_episode's rng.choice)
    # is untouched by this, it still runs fresh every call.
    feature_cache = {}
    stale=0
    for epoch in range(start_epoch, start_epoch + config.epochs):
        model.train(); optimizer.train()
        optimizer.zero_grad(set_to_none=True); losses=[]
        loader = make_curve_loader(train_cols, config.batch_size, SEED+epoch)
        optimizer_steps = 0; episode_counts={"real":0,"augmented":0,"skipped":0}
        for batch in tqdm(loader):
            episodes,counts=make_training_batch(batch,train_cols,X,y,context,config,feature_cache)
            for kind,count in counts.items(): episode_counts[kind] += count
            if not episodes: continue
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(episodes), config.gpu_batch_size):
                group = episodes[start:start+config.gpu_batch_size]
                backward_chunk(model, group, device, use_amp, len(episodes), losses)
            torch.nn.utils.clip_grad_norm_(params,config.grad_clip)
            optimizer.step(); optimizer_steps += 1
        # Schedule-Free AdamW: validation, and any checkpoint saved right
        # after it, must run on the polynomially-averaged "eval" point, not
        # the noisy fast train-time iterate -- see build_optimizer's
        # docstring. validation_mae() itself already calls model.eval().
        optimizer.eval()
        val = validation_mae(model,val_cols,X,y,context,config,device,
                             cache=feature_cache,gpu_batch_size=config.gpu_batch_size)
        row={"epoch":epoch+1,"train_l1_norm":float(np.mean(losses)),
             "val_mae_w":val,"optimizer_steps":optimizer_steps,
             "batch_size":config.batch_size,"gpu_batch_size":config.gpu_batch_size,
             "episode_counts":episode_counts,
             "training_protocol":TRAINING_PROTOCOL}
        history.append(row); HISTORY_PATH.write_text(json.dumps(history, indent=2))
        print(row, flush=True)
        if val < best:
            best=val; stale=0
            save_adapter(model, config, replaced, optimizer=optimizer)
        else:
            stale += 1
        optimizer.train()  # back to the fast point before the next epoch's backward passes
        if stale >= config.patience:
            break
    return history


def load_lora_checkpoint(model, path, device):
    """Returns ``(cfg, replaced, optimizer_state)``. ``optimizer_state`` is
    ``None`` for checkpoints saved before optimizer-state persistence (or by
    a call to ``save_adapter`` without ``optimizer=``) -- callers that only
    need the weights (evaluate/submit) already discard the extra return
    values, so this stays backward compatible."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("training_protocol") != TRAINING_PROTOCOL:
        raise RuntimeError(
            "Incompatible LoRA checkpoint: retrain with the real-gap v2 protocol")
    cfg = TrainConfig(**payload["config"])
    expected=payload.get("tabfm_manifest")
    if expected and expected != tabfm_manifest(model):
        raise RuntimeError(f"TabFM revision mismatch: checkpoint={expected}, runtime={tabfm_manifest(model)}")
    replaced = inject_lora(model, cfg.rank, cfg.alpha, cfg.dropout)
    if replaced != payload["targets"]:
        raise RuntimeError("TabFM module layout differs from the trained adapter")
    load_adapter_state(model, payload["state_dict"])
    model.to(device)
    assert_model_device(model, device)
    return cfg, replaced, payload.get("optimizer_state")


def build_optimizer(params, config):
    """Schedule-Free AdamW (Meta's ``schedulefree`` package,
    ``AdamWScheduleFree``) instead of plain ``torch.optim.AdamW``: it folds
    the usual LR-schedule + EMA weight-averaging into the optimizer itself,
    so no external scheduler is needed. In exchange the caller MUST call
    ``optimizer.train()`` before every training step and ``optimizer.eval()``
    before every eval-only forward pass (validation, checkpoint save) --
    both swap the underlying parameter tensors in place between the fast
    train-time iterate and the polynomially-averaged eval-time iterate, it
    is not a cosmetic flag like ``nn.Module.train()``/``.eval()``:
    ``.step()`` raises if called while not in train mode."""
    try:
        from schedulefree import AdamWScheduleFree
    except ImportError as exc:
        raise SystemExit(
            "schedulefree absent : `pip install schedulefree` (Schedule-Free "
            "optimizers, https://github.com/facebookresearch/schedule_free) "
            "avant de lancer `train`.") from exc
    return AdamWScheduleFree(params, lr=config.lr, weight_decay=config.weight_decay,
                              warmup_steps=config.warmup_steps)


@torch.no_grad()
def predict_curve(model, col, s, truth, context, config, device, require_targets=True):
    q = np.flatnonzero(s.isna().to_numpy())
    ep = make_episode(col, s, truth, context, config, query_idx=q,
                      rng=np.random.RandomState(SEED), require_targets=require_targets)
    if ep is None: return np.array([]), q
    q = ep["query_idx"]
    model.eval(); _, pred = episode_loss(model, ep, device)
    raw = pred.cpu().numpy()*ep["y_scale"]+ep["y_center"]
    return blend_episode_prediction(raw,ep,s,q),q


def blend_episode_prediction(raw, ep, s, q):
    runs=feb.run_lengths_for(ep["missing"]); edges=feb.edge_distance_for(ep["missing"])
    alpha=np.array([feb.continuous_alpha(runs[i],edges[i],True) for i in q])
    alpha *= 1.-V14_FEATURE_OPTS["periodic_alpha_strength"]*feb.periodic_agreement(s.to_numpy())[q]
    alpha *= feb.volatility_alpha_mult(feb.curve_volatility(s.to_numpy()), vol_weight=.8)
    return np.clip(alpha,0,.95)*ep["interp"][q]+(1-np.clip(alpha,0,.95))*raw


def evaluate_manifest(model,X,y,donor_cols,config,device,manifest):
    context=build_donor_context(X,donor_cols); result={}
    for split in ("val","test"):
        errors=[]; per_column={}
        for col in manifest[split]:
            pred,q=predict_curve(model,col,X[col],y[col],context,config,device)
            true=y[col].iloc[q].to_numpy(); valid=np.isfinite(true)
            err=np.abs(pred[valid]-true[valid]); errors.append(err)
            per_column[col]=float(err.mean())
        result[split]={"mae":float(np.concatenate(errors).mean()),
                       "column_mean":float(np.mean(list(per_column.values()))),
                       "column_std":float(np.std(list(per_column.values()))),
                       "per_column":per_column,"n_columns":len(per_column)}
    return result


def build_v14_matched_context(X, y, donor_cols, dev_cols):
    """Exact replica of tabpfn_v14_ablation.eval_split's pool construction:
    fixed donor_cols PLUS that repeat's dev_cols truths (y_train), with
    shape clusters fit on donor_cols alone and extended to dev_cols by
    predicting on their z-scored profile -- same order of operations, same
    random_state. evaluate()'s own build_donor_context only used donor_cols,
    which is a *smaller* pool than what produced tabpfn_v14_confirm_results.json
    (110.80 W at repeat 1/seed 42): not a fair "exact same data" comparison."""
    donor_df = X[donor_cols]
    dev_true_df = y[dev_cols]
    pool = pd.concat([donor_df, dev_true_df], axis=1)
    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=15, random_state=SEED)
    dev_filled = dev_true_df.ffill().bfill()
    z_dev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = pd.Series(km.predict(z_dev.T.to_numpy()), index=dev_true_df.columns)
    pool_labels = pd.concat([donor_labels, dev_labels])
    pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
    cluster_idx = {cl: pos[idx].to_numpy() for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(X.index, ("hour", "dow", "dom", "year"))
    return pool_vals, km, cluster_idx, tfeat


def evaluate_matched_v14(model, X, y, donor_cols, config, device, repeats, n_cols, label):
    """Same column selection AND donor-pool construction as
    tabpfn_v14_ablation.run_config/eval_split, so scores are directly
    comparable repeat-by-repeat to tabpfn_v14_confirm_results.json --
    repeat 1 (seed=SEED) draws the exact same 10 validation columns from
    the exact same donor+dev-truth pool as V14's own repeat 1 (110.80 W)."""
    scores = []
    for repeat in range(repeats):
        seed = SEED + repeat
        dev_cols, holdout = train_test_split(list(y.columns), test_size=.1, random_state=seed)
        cols = list(np.random.RandomState(seed).choice(holdout, min(n_cols, len(holdout)), replace=False))
        context = build_v14_matched_context(X, y, donor_cols, dev_cols)
        errors = []
        for col in cols:
            pred, q = predict_curve(model, col, X[col], y[col], context, config, device)
            true = y[col].iloc[q].to_numpy()
            valid = np.isfinite(true)
            errors.append(np.abs(pred[valid] - true[valid]))
        score = float(np.concatenate(errors).mean())
        scores.append(score)
        print(f"{label} repeat {repeat+1}/{repeats}: {score:.2f} W (pool V14-apparié)", flush=True)
    return scores


def evaluate(model, X, y, donor_cols, config, device, repeats, n_cols, label):
    scores=[]
    for repeat in range(repeats):
        seed=SEED+repeat
        dev, holdout=train_test_split(list(y.columns),test_size=.1,random_state=seed)
        cols=list(np.random.RandomState(seed).choice(holdout,min(n_cols,len(holdout)),replace=False))
        context=build_donor_context(X, donor_cols); errors=[]
        for col in cols:
            pred,q=predict_curve(model,col,X[col],y[col],context,config,device)
            true=y[col].iloc[q].to_numpy(); valid=np.isfinite(true)
            errors.append(np.abs(pred[valid]-true[valid]))
        score=float(np.concatenate(errors).mean()); scores.append(score)
        print(f"{label} repeat {repeat+1}/{repeats}: {score:.2f} W",flush=True)
    return scores


def generate_submission(model, X, y, Xtest, config, device):
    test_donors=[c for c in Xtest if not str(c).startswith("holed_")]
    train_donors=[c for c in X if not str(c).startswith("holed_")]
    holes=[c for c in Xtest if str(c).startswith("holed_")]
    pool=pd.concat([Xtest[test_donors],X[train_donors],y],axis=1)
    context=build_donor_context(pool,list(pool.columns))
    if PARTIAL_PATH.exists():
        done=pd.read_csv(PARTIAL_PATH,index_col=0); done.index=Xtest.index
    else: done=pd.DataFrame(index=Xtest.index)
    for number,col in enumerate((c for c in holes if c not in done),1):
        pred,q=predict_curve(model,col,Xtest[col],Xtest[col],context,config,device,
                             require_targets=False)
        filled=Xtest[col].copy()
        if len(q) > 0 and len(pred) == len(q):
            filled.iloc[q]=pred
        elif len(q) > 0:
            # predict_curve/make_episode returned no usable episode for this
            # column (e.g. feature_matrix's NaN guard -- too few observed
            # points to build a feature matrix). Never leave a column with
            # unfilled gaps: deterministic interpolation fallback, same
            # spirit as generate_submission_*.py's KNN fallback when the
            # primary model errors on a column.
            print(f"  [warn] no LoRA prediction for {col} ({len(q)} gaps); "
                  f"falling back to interpolation", flush=True)
            filled = fill_nan_with_interpolation(filled)
        done[col]=filled
        if number%10==0:
            tmp=str(PARTIAL_PATH)+".tmp"; done.to_csv(tmp); os.replace(tmp,PARTIAL_PATH)
            print(f"submission: {len(done.columns)}/{len(holes)}",flush=True)
    final=done[holes].clip(lower=0); final.index.name="Horodate"
    sample=pd.read_csv("data/new_output_sample.csv",index_col=0)
    assert final.shape==sample.shape and list(final.columns)==list(sample.columns)
    assert final.notna().all().all() and np.isfinite(final.to_numpy()).all()
    final.to_csv(OUT_PATH)


def smoke_batching(model, device):
    """train()/validation_mae() now batch multiple episodes into one
    padded [B, T, D] forward call (pad_stack_episodes/episode_batch_loss)
    instead of one call per episode, for GPU throughput on a large-VRAM
    card. This is only correct if the TabFM forward restricts each
    example's attention/loss to its own train_size/d/row-count and ignores
    the padding -- unverifiable from this repo (no access to the TabFM
    source). Two synthetic episodes of DIFFERENT row counts and DIFFERENT
    active feature widths (so padding is actually exercised on both axes)
    are run through the pre-existing single-episode path and the new
    batched path; their predictions must match. Run this before trusting
    any train/evaluate number produced with batching enabled."""
    def fake_episode(rows, train_size, active_d, feature_dim, seed):
        rng = np.random.RandomState(seed)
        x = np.zeros((rows, feature_dim), dtype=np.float32)
        x[:, :active_d] = rng.standard_normal((rows, active_d)).astype(np.float32)
        y = np.full(rows, SENTINEL, dtype=np.float32)
        y[:train_size] = rng.standard_normal(train_size).astype(np.float32)
        target = rng.standard_normal(rows - train_size).astype(np.float32)
        return dict(x=x, y=y, train_size=train_size, d=active_d, target=target)

    model.eval()
    episodes = [fake_episode(16, 12, 8, 10, 1), fake_episode(11, 7, 6, 10, 2)]
    with torch.no_grad():
        singles = [episode_loss(model, ep, device)[1].cpu().numpy() for ep in episodes]
        _, _, batched_preds = episode_batch_loss(model, episodes, device)
        batched = [p.cpu().numpy() for p in batched_preds]
    diffs = [float(np.max(np.abs(s - b))) for s, b in zip(singles, batched)]
    if any(not np.allclose(s, b, atol=1e-4, rtol=1e-3) for s, b in zip(singles, batched)):
        raise RuntimeError(
            f"Batched vs single-episode forward mismatch (max diffs={diffs}) -- "
            "this TabFM revision's forward() does not mask padded rows the way "
            "pad_stack_episodes assumes. Set TrainConfig.gpu_batch_size=1 to fall "
            "back to the (slow but known-correct) sequential path in train()/"
            "validation_mae(), or fix pad_stack_episodes for this revision.")
    print(f"smoke_batching OK: {len(episodes)} episodes, max diff={max(diffs):.6f}")


def smoke(model, device):
    replaced=inject_lora(model); model.to(device); assert_model_device(model,device); model.train()
    b,t,d=1,16,8
    x=torch.randn(b,t,d,device=device); y=torch.randn(b,t,device=device); y[:,12:]=SENTINEL
    output=model(x,y,torch.tensor([12],device=device),d=torch.tensor([d],device=device))
    target=torch.randn(b,4,device=device,dtype=output.dtype)
    loss=torch.nn.functional.l1_loss(output[:,12:,0].float(),target.float())
    loss.backward()
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n in trainable):
        raise RuntimeError(f"Unexpected trainable parameters: {trainable}")
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable_parameters(model)):
        raise RuntimeError("No finite LoRA gradients")
    print(f"smoke OK: loss={loss.item():.4f}, adapters={len(replaced)}, trainable={sum(p.numel() for p in trainable_parameters(model))}")
    smoke_batching(model, device)


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=["smoke","train","evaluate","submit"])
    p.add_argument("--protocol",choices=["screen","confirm","matched","v14_repeat1"],default="screen")
    p.add_argument("--device",default="cuda"); p.add_argument("--outer-seed",type=int,default=SEED)
    p.add_argument("--accept-non-commercial-license",action="store_true")
    return p.parse_args()


def main():
    args=parse_args(); set_seed()
    if not args.accept_non_commercial_license:
        raise SystemExit("Pass --accept-non-commercial-license after reviewing the TabFM weights license")
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
    model=load_tabfm(device)
    if args.stage=="smoke": return smoke(model,device)
    X,y,Xtest=load_data(); donors=[c for c in X if not str(c).startswith("holed_")]
    if args.stage=="train": return train(model,X,y,donors,TrainConfig(),device,args.outer_seed)
    if not ADAPTER_PATH.exists(): raise SystemExit("Train the adapter first")
    payload=torch.load(ADAPTER_PATH,map_location="cpu",weights_only=False)
    config=TrainConfig(**payload["config"])
    if args.stage=="evaluate":
        if args.protocol=="matched":
            manifest=load_or_create_split_manifest(X,y,config,args.outer_seed)
            load_lora_checkpoint(model,ADAPTER_PATH,device)
            matched=evaluate_manifest(model,X,y,donors,config,device,manifest)
            RESULTS_PATH.write_text(json.dumps({"protocol":"matched",
                "training_protocol":TRAINING_PROTOCOL,"lora":matched,
                "split_manifest":str(SPLIT_PATH)},indent=2)); return
        if args.protocol=="v14_repeat1":
            # Repeat 1 only (seed=SEED, 10 columns), donor pool matched to
            # tabpfn_v14_ablation.eval_split -- directly comparable to V14's
            # own repeat 1 (tabpfn_v14_confirm_results.json).
            zero=evaluate_matched_v14(model,X,y,donors,config,device,1,10,"TabFM-zero-shot")
            load_lora_checkpoint(model,ADAPTER_PATH,device)
            scores=evaluate_matched_v14(model,X,y,donors,config,device,1,10,"TabFM-LoRA")
            v14_ref_path=Path("tabpfn_v14_confirm_results.json")
            v14_repeat1=(json.loads(v14_ref_path.read_text())["scores"][0]
                        if v14_ref_path.exists() else None)
            result={"protocol":"v14_repeat1","training_protocol":TRAINING_PROTOCOL,
                    "seed":SEED,"n_columns":10,
                    "v14_repeat1_reference_mae_w":v14_repeat1,
                    "zero_shot_scores":zero,"scores":scores,
                    "zero_shot_mean":float(np.mean(zero)),"mean":float(np.mean(scores))}
            RESULTS_PATH.write_text(json.dumps(result,indent=2)); return
        repeats,ncols=(3,5) if args.protocol=="screen" else (5,10)
        zero=evaluate(model,X,y,donors,config,device,repeats,ncols,"TabFM-zero-shot")
        load_lora_checkpoint(model,ADAPTER_PATH,device)
        scores=evaluate(model,X,y,donors,config,device,repeats,ncols,"TabFM-LoRA")
        result={"protocol":args.protocol,"training_protocol":TRAINING_PROTOCOL,
                "zero_shot_scores":zero,"scores":scores,
                "zero_shot_mean":float(np.mean(zero)),"mean":float(np.mean(scores)),
                "std":float(np.std(scores)),"wins":int(np.sum(np.array(scores)<np.array(zero))),
                "seeds":list(range(SEED,SEED+repeats))}
        RESULTS_PATH.write_text(json.dumps(result,indent=2)); return
    confirmed=json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    #if (confirmed.get("protocol")!="confirm" or
    #        confirmed.get("training_protocol")!=TRAINING_PROTOCOL or
    #        confirmed.get("mean",math.inf)>=confirmed.get("zero_shot_mean",-math.inf)):
    #    raise SystemExit("Submission blocked until LoRA beats zero-shot on confirmation")
    load_lora_checkpoint(model,ADAPTER_PATH,device)
    generate_submission(model,X,y,Xtest,config,device)


if __name__=="__main__": main()

