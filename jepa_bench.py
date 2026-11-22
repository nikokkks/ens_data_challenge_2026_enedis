"""JEPA (Joint-Embedding Predictive Architecture, Yann LeCun / Meta FAIR)
appliqué à l'imputation de courbes Enedis -- un "world model" auto-supervisé
en PyTorch, demandé explicitement malgré le risque documenté ci-dessous.

Ce que fait vraiment JEPA (par opposition à un auto-encodeur masqué classique
type BERT/MAE) : au lieu de reconstruire les valeurs brutes manquantes, un
encodeur "contexte" et un encodeur "cible" (copie EMA du premier, jamais
entraînée par gradient -- stop-gradient) projettent les instants observés et
les instants à prédire dans un espace latent ; un prédicteur apprend à
deviner la représentation latente cible à partir du contexte. Une
régularisation de variance/covariance (VICReg / LeJEPA, Balestriero & LeCun
2025) empêche l'effondrement des représentations (pas de paires négatives).

AVERTISSEMENT (lu avant de coder, cf. conversation) : ce dépôt a DÉJÀ testé
la même famille d'idée -- un modèle global à poids partagés, pré-entraîné en
auto-supervisé sur les courbes donneuses avec des trous synthétiques tirés
de la distribution empirique réelle (`lstm_bench.py`, réutilisé ici tel
quel : `robust_scale`, `time_channels`, `sample_mask`, `HOLE_LENGTHS/PROBS`).
Résultat sur le protocole officiel `10x10` : 74,66 W, loin derrière
TabPFN V14 (63,24 W). Diagnostic du dépôt : les modèles globaux plafonnent
parce qu'ils n'ont accès à AUCUNE information inter-courbes au moment de
l'inférence -- alors que c'est précisément la source du gain de TabPFN/KNN
(sélection de voisins corrélés, spécifique à chaque courbe cible).

Pour donner à ce JEPA une vraie chance (et pas rejouer bêtement le même
échec), chaque instant reçoit, en plus du calendrier, un canal de
"consensus voisins" -- la même consultation KNN pondérée par corrélation
que `evaluate.make_knn_model` / la feature set V12/V14 (k=40), calculée sur
TOUTE la période (pas seulement aux trous) et fournie aussi bien au contexte
qu'aux requêtes du prédicteur. Le JEPA reste néanmoins un modèle GLOBAL à
poids partagés (contrairement à TabPFN qui ré-apprend en contexte par
colonne) : l'attente réaliste, cohérente avec §7/§7ter de `benchmark.md`,
est un score dans la zone ~70-90 W, pas un nouveau record.

Deux phases indépendantes (le pré-entraînement est cher, le fine-tuning non) :

  python jepa_bench.py pretrain [--max-steps 3000]
      Auto-supervisé sur un pool de courbes donneuses (jamais y_train),
      trous synthétiques empiriques -> écrit jepa_pretrained.pt.

  python jepa_bench.py eval [--epochs 6] [--finetune-predictor] [--no-blend]
      Charge le JEPA pré-entraîné (gelé), et pour chacun des 10 repeats du
      protocole officiel 10x10 (mêmes graines que TabPFN/LSTM) : entraîne une
      petite tête de régression (+ éventuellement le prédicteur) sur les VRAIS
      trous des colonnes dev, prédit sur les colonnes val, calcule official_mae.
      Aucune fuite : y_train des colonnes val n'est jamais vu.
"""

import argparse
import copy
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluate import load_data, official_mae, fill_nan_with_interpolation
from feature_engineering_bench import run_lengths_for, edge_distance_for, continuous_alpha
from lstm_bench import (
    robust_scale, robust_norm, time_channels, sample_mask,
    T_SPLIT, RANDOM_STATE, Z_CLIP,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))

WINDOW = 192               # 4 jours, même fenêtre que la meilleure config LSTM (L01/L13)
D_MODEL = 128
N_HEADS = 4
ENC_LAYERS = 3
PRED_LAYERS = 2
DROPOUT = 0.1
EMA_MOMENTUM = 0.996        # fixe (I-JEPA original fait croître le momentum ; simplifié ici)
VICREG_VAR_W = 1.0
VICREG_COV_W = 0.04         # ratio ~25:1 var:cov, convention VICReg
REG_SCALE = 0.25

N_PRETRAIN_DONORS = 4000    # sous-échantillon de courbes donneuses (corrélation NxN sinon trop lourde)
NEIGHBOR_K = 40             # même k que V12/V14
SIDE_DIM = 5                # sin/cos heure, sin/cos jour, consensus voisins

PRETRAIN_CKPT = "jepa_pretrained.pt"
PRETRAIN_PROGRESS = "jepa_pretrain_progress.json"
EVAL_RESULTS = "jepa_eval_results.json"

N_REPEATS = 10
N_VAL_SUB = 10
MAX_WINDOWS_PER_COL = 3     # fine-tuning : au plus 3 trous réels échantillonnés par colonne dev


# ---------------------------------------------------------------------------
# Feature "consensus voisins" (même logique que evaluate.make_knn_model / V12)
# ---------------------------------------------------------------------------

def batch_neighbor_consensus(pool_vals_T, k=NEIGHBOR_K):
    """pool_vals_T : (T, N) courbes COMPLETES. Renvoie, pour chaque courbe du
    pool, sa propre série de consensus basée sur ses k voisines les plus
    corrélées DANS LE MÊME POOL (elle-même exclue). Vectorisé : une seule
    matrice de corrélation NxN plutôt qu'une boucle par courbe cible."""
    x = pool_vals_T - pool_vals_T.mean(axis=0, keepdims=True)
    denom = np.sqrt((x ** 2).sum(axis=0)) + 1e-9
    corr = (x.T @ x) / (denom[:, None] * denom[None, :])
    np.fill_diagonal(corr, -np.inf)
    N = pool_vals_T.shape[1]
    kk = min(k, N - 1)
    top_idx = np.argpartition(-corr, kk - 1, axis=1)[:, :kk]
    mean_all = pool_vals_T.mean(axis=0)
    std_all = pool_vals_T.std(axis=0) + 1e-9
    consensus = np.zeros_like(pool_vals_T)
    for i in range(N):
        idx = top_idx[i]
        w = np.clip(corr[i, idx], 1e-3, None)
        w = w / w.sum()
        scaled = (pool_vals_T[:, idx] - mean_all[idx]) / std_all[idx] * std_all[i] + mean_all[i]
        consensus[:, i] = scaled @ w
    return consensus


def neighbor_consensus(values, pool_vals_T, k=NEIGHBOR_K):
    """Consensus voisins pour UNE courbe cible (potentiellement trouée) contre
    un pool externe fixe, calculé sur TOUTE la période (pas seulement aux
    trous) -- corrélation/échelle estimées uniquement sur les points observés
    de la cible, jamais sur ses valeurs manquantes."""
    obs = ~np.isnan(values)
    if obs.sum() < 5:
        return np.zeros_like(values, dtype=np.float32)
    target_obs = values[obs]
    donors_obs = pool_vals_T[obs, :]
    t_c = target_obs - target_obs.mean()
    d_c = donors_obs - donors_obs.mean(axis=0, keepdims=True)
    num = d_c.T @ t_c
    denom = np.sqrt((d_c ** 2).sum(axis=0) * (t_c ** 2).sum()) + 1e-9
    corr = num / denom
    kk = min(k, corr.shape[0])
    top_idx = np.argpartition(-corr, kk - 1)[:kk]
    w = np.clip(corr[top_idx], 1e-3, None)
    w = w / w.sum()
    donor_sel = pool_vals_T[:, top_idx]
    d_mean, d_std = donor_sel[obs].mean(axis=0), donor_sel[obs].std(axis=0) + 1e-9
    t_mean, t_std = target_obs.mean(), target_obs.std() + 1e-9
    scaled = (donor_sel - d_mean) / d_std * t_std + t_mean
    return (scaled * w).sum(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Batching : instants -> tokens contexte / requêtes-cibles, avec padding
# ---------------------------------------------------------------------------

def pack_context_target(x, side, mask, device):
    """x: (B,W) valeur normalisée (0 aux positions manquantes) ; side: (B,W,SIDE_DIM)
    calendrier+consensus (toujours disponible) ; mask: (B,W) True = cible JEPA."""
    B, W = x.shape
    side_dim = side.shape[-1]
    ctx_lens = (~mask).sum(axis=1)
    tgt_lens = mask.sum(axis=1)
    max_ctx = int(max(ctx_lens.max(), 1))
    max_tgt = int(max(tgt_lens.max(), 1))

    ctx_val = np.zeros((B, max_ctx, 1), dtype=np.float32)
    ctx_side = np.zeros((B, max_ctx, side_dim), dtype=np.float32)
    ctx_pos = np.zeros((B, max_ctx), dtype=np.int64)
    ctx_pad = np.ones((B, max_ctx), dtype=bool)

    tgt_side = np.zeros((B, max_tgt, side_dim), dtype=np.float32)
    tgt_pos = np.zeros((B, max_tgt), dtype=np.int64)
    tgt_pad = np.ones((B, max_tgt), dtype=bool)

    for b in range(B):
        ci = np.flatnonzero(~mask[b])
        ti = np.flatnonzero(mask[b])
        nc, nt = len(ci), len(ti)
        if nc:
            ctx_val[b, :nc, 0] = x[b, ci]
            ctx_side[b, :nc] = side[b, ci]
            ctx_pos[b, :nc] = ci
            ctx_pad[b, :nc] = False
        if nt:
            tgt_side[b, :nt] = side[b, ti]
            tgt_pos[b, :nt] = ti
            tgt_pad[b, :nt] = False

    to_t = lambda a: torch.from_numpy(a).to(device)
    return dict(
        ctx_val=to_t(ctx_val), ctx_side=to_t(ctx_side), ctx_pos=to_t(ctx_pos), ctx_pad=to_t(ctx_pad),
        tgt_side=to_t(tgt_side), tgt_pos=to_t(tgt_pos), tgt_pad=to_t(tgt_pad),
        full_val=to_t(x.astype(np.float32)), full_side=to_t(side.astype(np.float32)),
    )


def build_jepa_batch(curves, side_curves, lo, hi, window, batch_size, rng, device):
    """Fenêtres aléatoires sur le pool de pré-entraînement, trous synthétiques
    tirés de la distribution empirique réelle (`sample_mask`, cf. lstm_bench)."""
    n_curves = curves.shape[0]
    rows = rng.randint(0, n_curves, size=batch_size)
    starts = rng.randint(lo, hi - window + 1, size=batch_size)
    x = np.stack([curves[r, s:s + window] for r, s in zip(rows, starts)])
    side = np.stack([side_curves[r, s:s + window] for r, s in zip(rows, starts)])
    mask = sample_mask(x.shape, rng)
    return pack_context_target(x, side, mask, device)


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------

class TokenEmbed(nn.Module):
    def __init__(self, d_model, window, side_dim=SIDE_DIM):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.side_proj = nn.Linear(side_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(window, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

    def context_tokens(self, ctx_val, ctx_side, ctx_pos):
        return self.value_proj(ctx_val) + self.side_proj(ctx_side) + self.pos_embed[ctx_pos]

    def query_tokens(self, tgt_side, tgt_pos):
        return self.mask_token + self.side_proj(tgt_side) + self.pos_embed[tgt_pos]

    def full_tokens(self, full_val, full_side):
        B, W = full_val.shape
        pos = torch.arange(W, device=full_val.device).unsqueeze(0).expand(B, W)
        return self.value_proj(full_val.unsqueeze(-1)) + self.side_proj(full_side) + self.pos_embed[pos]


class TransformerBody(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
            batch_first=True, activation="gelu")
        self.body = nn.TransformerEncoder(layer, n_layers)

    def forward(self, tokens, key_padding_mask=None):
        return self.body(tokens, src_key_padding_mask=key_padding_mask)


class JEPA(nn.Module):
    """embed/encoder = branche "étudiant" (entraînée par gradient).
    embed_tgt/encoder_tgt = copie EMA, jamais entraînée par gradient (stop-grad),
    mise à jour uniquement via `update_target()`."""

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, enc_layers=ENC_LAYERS,
                 pred_layers=PRED_LAYERS, window=WINDOW, dropout=DROPOUT, ema=EMA_MOMENTUM):
        super().__init__()
        self.window = window
        self.d_model = d_model
        self.embed = TokenEmbed(d_model, window)
        self.encoder = TransformerBody(d_model, n_heads, enc_layers, dropout)
        self.embed_tgt = copy.deepcopy(self.embed)
        self.encoder_tgt = copy.deepcopy(self.encoder)
        for p in list(self.embed_tgt.parameters()) + list(self.encoder_tgt.parameters()):
            p.requires_grad_(False)
        self.embed_tgt.eval()
        self.encoder_tgt.eval()
        self.predictor = TransformerBody(d_model, n_heads, pred_layers, dropout)
        self.ema = ema

    def train_mode(self):
        self.embed.train(); self.encoder.train(); self.predictor.train()
        self.embed_tgt.eval(); self.encoder_tgt.eval()

    def eval_mode(self):
        self.embed.eval(); self.encoder.eval(); self.predictor.eval()
        self.embed_tgt.eval(); self.encoder_tgt.eval()

    def trainable_parameters(self):
        return list(self.embed.parameters()) + list(self.encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target(self):
        for tp, cp in zip(self.embed_tgt.parameters(), self.embed.parameters()):
            tp.mul_(self.ema).add_(cp.detach(), alpha=1 - self.ema)
        for tp, cp in zip(self.encoder_tgt.parameters(), self.encoder.parameters()):
            tp.mul_(self.ema).add_(cp.detach(), alpha=1 - self.ema)

    def encode_context(self, batch):
        tokens = self.embed.context_tokens(batch["ctx_val"], batch["ctx_side"], batch["ctx_pos"])
        return self.encoder(tokens, batch["ctx_pad"])

    def predict_targets(self, ctx_repr, batch):
        query = self.embed.query_tokens(batch["tgt_side"], batch["tgt_pos"])
        seq = torch.cat([ctx_repr, query], dim=1)
        pad = torch.cat([batch["ctx_pad"], batch["tgt_pad"]], dim=1)
        out = self.predictor(seq, pad)
        return out[:, ctx_repr.shape[1]:]

    @torch.no_grad()
    def target_representations(self, batch):
        tokens = self.embed_tgt.full_tokens(batch["full_val"], batch["full_side"])
        full_repr = self.encoder_tgt(tokens, None)
        d = full_repr.shape[-1]
        idx = batch["tgt_pos"].unsqueeze(-1).expand(-1, -1, d)
        return full_repr.gather(1, idx).detach()

    def forward(self, batch):
        ctx_repr = self.encode_context(batch)
        pred = self.predict_targets(ctx_repr, batch)
        target = self.target_representations(batch)
        return pred, target


class ValueHead(nn.Module):
    """Décodeur latent -> valeur normalisée (seule partie qui traduit la
    représentation JEPA en une prédiction exploitable en Watts)."""

    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, repr_):
        return self.mlp(repr_).squeeze(-1)


def jepa_loss(pred, target, tgt_pad, var_w=VICREG_VAR_W, cov_w=VICREG_COV_W, reg_scale=REG_SCALE):
    valid = ~tgt_pad
    p, z = pred[valid], target[valid]
    pred_loss = F.smooth_l1_loss(p, z)

    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    d = cov.shape[0]
    cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d

    reg = var_w * var_loss + cov_w * cov_loss
    return pred_loss + reg_scale * reg, dict(pred=float(pred_loss), var=float(var_loss), cov=float(cov_loss))


# ---------------------------------------------------------------------------
# Phase 1 : pré-entraînement auto-supervisé (jamais y_train)
# ---------------------------------------------------------------------------

def pretrain(max_steps=3000, batch_size=64, window=WINDOW, eval_every=100, patience=6,
             lr=2e-4, n_donors=N_PRETRAIN_DONORS, verbose=True):
    torch.manual_seed(RANDOM_STATE)
    rng = np.random.RandomState(RANDOM_STATE)

    print("Chargement des données...", flush=True)
    X_train, _, _ = load_data()
    all_donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]
    pool_rng = np.random.RandomState(RANDOM_STATE)
    donor_cols = list(pool_rng.choice(all_donor_cols, size=min(n_donors, len(all_donor_cols)), replace=False))

    donors_T = X_train[donor_cols].ffill().bfill().values.astype(np.float32)  # (T, N)
    print(f"Pool de pré-entraînement : {len(donor_cols)} courbes donneuses -> "
          f"calcul du consensus voisins (k={NEIGHBOR_K})...", flush=True)
    t0 = time.time()
    consensus_T = batch_neighbor_consensus(donors_T, k=NEIGHBOR_K)  # (T, N)
    print(f"  consensus calculé en {time.time()-t0:.1f}s", flush=True)

    curves, center, scale = robust_norm(donors_T.T)  # (N, T)
    consensus_z = np.clip((consensus_T.T - center) / scale, -Z_CLIP, Z_CLIP).astype(np.float32)  # (N, T)
    tfeat = time_channels(X_train.index)  # (T, 4)
    side_curves = np.concatenate(
        [np.broadcast_to(tfeat, (curves.shape[0], *tfeat.shape)), consensus_z[..., None]], axis=-1
    ).astype(np.float32)  # (N, T, 5)

    print(f"{curves.shape[0]} courbes x {curves.shape[1]} pas, device={DEVICE}", flush=True)

    model = JEPA(window=window).to(DEVICE)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=0.01)

    val_rng = np.random.RandomState(RANDOM_STATE + 1000)
    val_batches = [
        build_jepa_batch(curves, side_curves, T_SPLIT, curves.shape[1], window, 64, val_rng, DEVICE)
        for _ in range(6)
    ]

    best, best_state, bad, history = np.inf, None, 0, []
    t0 = time.time()
    for step in range(1, max_steps + 1):
        model.train_mode()
        batch = build_jepa_batch(curves, side_curves, 0, T_SPLIT, window, batch_size, rng, DEVICE)
        pred, target = model(batch)
        loss, parts = jepa_loss(pred, target, batch["tgt_pad"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step()
        model.update_target()

        if step % eval_every == 0:
            model.eval_mode()
            with torch.no_grad():
                vals = []
                for vb in val_batches:
                    p, t = model(vb)
                    l, _ = jepa_loss(p, t, vb["tgt_pad"])
                    vals.append(float(l))
            v = float(np.mean(vals))
            history.append({"step": step, "val_loss": v, **parts})
            if verbose:
                print(f"    step {step:4d}  train={float(loss):.4f}  val={v:.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    print(f"    early stopping au step {step}", flush=True)
                    break

    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(), "window": window, "d_model": model.d_model,
        "donor_pool_cols": donor_cols,
    }, PRETRAIN_CKPT)
    json.dump({"best_val_loss": best, "history": history, "train_seconds": time.time() - t0},
              open(PRETRAIN_PROGRESS, "w"), indent=2)
    print(f"Pré-entraînement terminé : best val JEPA loss = {best:.4f} ({time.time()-t0:.0f}s) "
          f"-> {PRETRAIN_CKPT}", flush=True)
    return model, donor_cols


# ---------------------------------------------------------------------------
# Phase 2 : fine-tuning léger de la tête + éval officielle 10x10
# ---------------------------------------------------------------------------

def prepare_column_cache(raw_by_col, pool_vals_T, tfeat_full):
    """Pré-calcule, PAR COLONNE (une fois, pas par fenêtre), l'échelle robuste
    et le canal de consensus voisins -- réutilisés pour toutes les fenêtres
    tirées de cette colonne."""
    cache = {}
    for col, raw in raw_by_col.items():
        obs = ~np.isnan(raw)
        if obs.sum() < 5:
            continue
        center, scl = robust_scale(raw[obs][None, :].astype(np.float32))
        center, scl = float(center.item()), float(scl.item())
        consensus_raw = neighbor_consensus(raw, pool_vals_T, k=NEIGHBOR_K)
        consensus_z = np.clip((consensus_raw - center) / scl, -Z_CLIP, Z_CLIP).astype(np.float32)
        z = np.where(obs, np.clip((raw - center) / scl, -Z_CLIP, Z_CLIP), 0.0).astype(np.float32)
        side = np.concatenate([tfeat_full, consensus_z[:, None]], axis=1).astype(np.float32)
        cache[col] = dict(raw=raw, z=z, side=side, center=center, scale=scl, obs=obs)
    return cache


def gap_run_starts(obs, window, max_windows, rng):
    missing = ~obs
    idx = np.flatnonzero(missing)
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    T = len(obs)
    if len(runs) > max_windows:
        chosen = [runs[i] for i in rng.choice(len(runs), max_windows, replace=False)]
    else:
        chosen = runs
    starts = [int(np.clip((a + b) // 2 - window // 2, 0, max(0, T - window))) for a, b in chosen]
    return sorted(set(starts))


def build_finetune_examples(cache, window, max_windows, rng):
    examples = []
    for col, c in cache.items():
        for s in gap_run_starts(c["obs"], window, max_windows, rng):
            examples.append((col, s))
    return examples


def make_finetune_batch(items, cache, true_by_col, window, device):
    xs, sides, masks, truths = [], [], [], []
    for col, s in items:
        c = cache[col]
        e = s + window
        seg_obs = c["obs"][s:e]
        truth_seg = true_by_col[col][s:e].astype(np.float32)
        miss = (~seg_obs) & ~np.isnan(truth_seg)
        if miss.sum() == 0:
            continue
        xs.append(c["z"][s:e])
        sides.append(c["side"][s:e])
        masks.append(miss)
        truths.append((truth_seg - c["center"]) / c["scale"])
    if not xs:
        return None
    x, side, mask, truth_z = np.stack(xs), np.stack(sides), np.stack(masks), np.stack(truths)
    batch = pack_context_target(x, side, mask, device)
    return batch, torch.from_numpy(truth_z.astype(np.float32)).to(device)


def _eval_examples(jepa, head, examples, cache, true_by_col, device, batch_size=32):
    head.eval()
    jepa.predictor.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            packed = make_finetune_batch(examples[i:i + batch_size], cache, true_by_col, jepa.window, device)
            if packed is None:
                continue
            batch, truth_z = packed
            ctx_repr = jepa.encode_context(batch)
            pred_repr = jepa.predict_targets(ctx_repr, batch)
            pred_val = head(pred_repr)
            true_at_tgt = torch.gather(truth_z, 1, batch["tgt_pos"])
            valid = ~batch["tgt_pad"]
            losses.append(float(F.l1_loss(pred_val[valid], true_at_tgt[valid])))
    return float(np.mean(losses)) if losses else float("inf")


def finetune_head(jepa, examples, cache, true_by_col, device, max_epochs=200, patience=50,
                   batch_size=32, lr=1e-3, finetune_predictor=False, seed=0,
                   inner_val_frac=0.15, verbose=False):
    """Cross-validation interne : les colonnes dev de CE repeat sont scindées
    en un sous-jeu d'entraînement et un sous-jeu de validation (par COLONNE,
    pas par fenêtre -- deux fenêtres de la même courbe partagent center/scale
    et sont corrélées, les mélanger train/val fuiterait). Entraîne jusqu'à
    `max_epochs`, arrêt anticipé si la loss de validation interne ne
    s'améliore plus pendant `patience` epochs, restaure le meilleur état."""
    rng = np.random.RandomState(seed)
    cols = sorted({col for col, _ in examples})
    rng.shuffle(cols)
    n_val_cols = max(1, int(round(len(cols) * inner_val_frac)))
    val_col_set = set(cols[:n_val_cols])
    train_examples = [e for e in examples if e[0] not in val_col_set]
    val_examples = [e for e in examples if e[0] in val_col_set]
    if not train_examples or not val_examples:
        # trop peu de colonnes/fenêtres pour scinder proprement (repli rare) :
        # pas d'arrêt anticipé possible, on entraîne sur tout pendant max_epochs.
        train_examples, val_examples = examples, examples

    head = ValueHead(jepa.d_model).to(device)
    params = list(head.parameters())
    if finetune_predictor:
        params += list(jepa.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    best_val, best_head_state, best_pred_state, bad, epoch = np.inf, None, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        head.train()
        jepa.predictor.train() if finetune_predictor else jepa.predictor.eval()
        order = rng.permutation(len(train_examples))
        for i in range(0, len(order), batch_size):
            items = [train_examples[j] for j in order[i:i + batch_size]]
            packed = make_finetune_batch(items, cache, true_by_col, jepa.window, device)
            if packed is None:
                continue
            batch, truth_z = packed
            with torch.no_grad():
                ctx_repr = jepa.encode_context(batch)
            pred_repr = jepa.predict_targets(ctx_repr, batch)
            pred_val = head(pred_repr)
            true_at_tgt = torch.gather(truth_z, 1, batch["tgt_pos"])
            valid = ~batch["tgt_pad"]
            loss = F.l1_loss(pred_val[valid], true_at_tgt[valid])
            opt.zero_grad()
            loss.backward()
            opt.step()

        val_loss = _eval_examples(jepa, head, val_examples, cache, true_by_col, device, batch_size)
        if verbose:
            print(f"      epoch {epoch:3d}  val={val_loss:.4f}  (bad={bad})", flush=True)
        if val_loss < best_val - 1e-5:
            best_val, bad = val_loss, 0
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
            if finetune_predictor:
                best_pred_state = {k: v.clone() for k, v in jepa.predictor.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    head.load_state_dict(best_head_state)
    if finetune_predictor and best_pred_state is not None:
        jepa.predictor.load_state_dict(best_pred_state)
    head.eval()
    jepa.predictor.eval()
    return head, epoch, best_val


def impute_curve_jepa(jepa, head, values, tfeat_full, pool_vals_T, window, device,
                       stride=None, alpha_blend=True):
    """Inférence sur une courbe complète, fenêtres glissantes moyennées.
    Repli sur l'interpolation linéaire si une fenêtre n'a aucun contexte
    connu (cas extrême type holed_192, ~99% manquant)."""
    stride = stride or window // 2
    n = len(values)
    missing = np.isnan(values)
    obs = ~missing
    interp = fill_nan_with_interpolation(pd.Series(values)).values

    if obs.sum() < 5:
        return interp

    center, scl = robust_scale(values[obs][None, :].astype(np.float32))
    center, scl = float(center.item()), float(scl.item())
    consensus_raw = neighbor_consensus(values, pool_vals_T, k=NEIGHBOR_K)
    consensus_z = np.clip((consensus_raw - center) / scl, -Z_CLIP, Z_CLIP).astype(np.float32)
    side_full = np.concatenate([tfeat_full, consensus_z[:, None]], axis=1).astype(np.float32)
    z = np.where(missing, 0.0, np.clip((values - center) / scl, -Z_CLIP, Z_CLIP)).astype(np.float32)

    acc, cnt = np.zeros(n), np.zeros(n)
    starts = list(range(0, max(1, n - window + 1), stride))
    if starts[-1] != max(0, n - window):
        starts.append(max(0, n - window))

    jepa.eval_mode()
    head.eval()
    with torch.no_grad():
        for s in starts:
            e = min(s + window, n)
            seg_missing = missing[s:e]
            if not seg_missing.any():
                continue
            ctx = ~seg_missing
            if ctx.sum() == 0:
                continue  # repli : reste à l'interpolation sur cette portion
            batch = pack_context_target(z[s:e][None], side_full[s:e][None], seg_missing[None], device)
            ctx_repr = jepa.encode_context(batch)
            pred_repr = jepa.predict_targets(ctx_repr, batch)
            pred_z = head(pred_repr)[0].cpu().numpy()
            tgt_idx = np.flatnonzero(seg_missing)
            acc[s:e][tgt_idx] += pred_z[:len(tgt_idx)]
            cnt[s:e][tgt_idx] += 1

    has_pred = cnt > 0
    model_pred = np.where(has_pred, (acc / np.maximum(cnt, 1)) * scl + center, interp)
    if alpha_blend:
        run_len = run_lengths_for(missing)
        edge_dist = edge_distance_for(missing)
        miss_idx = np.flatnonzero(missing)
        # continuous_alpha n'est pas vectorisé (cf. son usage partout ailleurs dans le
        # dépôt) -> appel scalaire par position manquante, comme les autres scripts.
        alpha = np.zeros(n)
        alpha[miss_idx] = [continuous_alpha(run_len[i], edge_dist[i], adaptive=True) for i in miss_idx]
        blended = alpha * interp + (1 - alpha) * model_pred
    else:
        blended = model_pred
    filled = values.copy()
    filled[missing] = blended[missing]
    return filled


def run_official_eval(max_epochs=200, patience=50, finetune_predictor=False, blend_interp=True,
                       max_repeats=None, save=True):
    from sklearn.model_selection import train_test_split

    print("Chargement des données...", flush=True)
    X_train, y_train, _ = load_data()
    tfeat_full = time_channels(X_train.index)

    # weights_only=False : checkpoint généré par ce même script (contient une liste de
    # noms de colonnes en plus du state_dict), source de confiance.
    ckpt = torch.load(PRETRAIN_CKPT, map_location=DEVICE, weights_only=False)
    jepa = JEPA(window=ckpt["window"], d_model=ckpt["d_model"]).to(DEVICE)
    jepa.load_state_dict(ckpt["state_dict"])
    jepa.eval_mode()
    for p in jepa.parameters():
        p.requires_grad_(False)
    if finetune_predictor:
        for p in jepa.predictor.parameters():
            p.requires_grad_(True)

    pool_vals_T = X_train[ckpt["donor_pool_cols"]].ffill().bfill().values.astype(np.float32)
    print(f"JEPA chargé (window={jepa.window}, pool={len(ckpt['donor_pool_cols'])} donneuses)", flush=True)

    holed_cols = list(y_train.columns)
    scores = []
    n_repeats = min(N_REPEATS, max_repeats) if max_repeats else N_REPEATS
    for repeat in range(n_repeats):
        dev_cols, val_cols = train_test_split(holed_cols, test_size=0.10, random_state=RANDOM_STATE + repeat)
        rng = np.random.RandomState(RANDOM_STATE + repeat)
        val_cols = list(rng.choice(val_cols, size=N_VAL_SUB, replace=False))

        t0 = time.time()
        dev_raw = {c: X_train[c].values.astype(np.float32) for c in dev_cols}
        dev_true = {c: y_train[c].values.astype(np.float32) for c in dev_cols}
        cache = prepare_column_cache(dev_raw, pool_vals_T, tfeat_full)
        examples = build_finetune_examples(cache, jepa.window, MAX_WINDOWS_PER_COL, rng)

        head, epochs_run, inner_val_loss = finetune_head(
            jepa, examples, cache, dev_true, DEVICE, max_epochs=max_epochs, patience=patience,
            finetune_predictor=finetune_predictor, seed=RANDOM_STATE + repeat)

        preds = {}
        for col in val_cols:
            values = X_train[col].values.astype(np.float64)
            if not np.isnan(values).any():
                preds[col] = values
                continue
            preds[col] = impute_curve_jepa(jepa, head, values, tfeat_full, pool_vals_T,
                                            jepa.window, DEVICE, alpha_blend=blend_interp)
        pred_df = pd.DataFrame(preds, index=X_train.index)
        mask = X_train[val_cols].isna()
        mae = official_mae(y_train[val_cols], pred_df, mask)
        dt = time.time() - t0
        scores.append(mae)
        stopped = "early-stop" if epochs_run < max_epochs else "max-epochs atteint"
        print(f"  repeat {repeat + 1:2d}/{n_repeats}: MAE={mae:8.2f} W  "
              f"({len(examples)} exemples dev, {epochs_run} epochs [{stopped}], "
              f"val interne={inner_val_loss:.4f}, {dt:.1f}s)", flush=True)

    scores = np.array(scores)
    print(f"\n  -> MAE moyenne {scores.mean():.2f} +/- {scores.std():.2f} W sur {len(scores)} repeats "
          f"(par repeat: {np.round(scores, 1).tolist()})", flush=True)
    if save:
        json.dump({
            "scores": scores.tolist(), "finetune_predictor": finetune_predictor,
            "blend_interp": blend_interp, "max_epochs": max_epochs, "patience": patience,
        }, open(EVAL_RESULTS, "w"), indent=2)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pretrain", "eval", "all"])
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "3000")))
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--finetune-predictor", action="store_true")
    parser.add_argument("--no-blend", action="store_true")
    parser.add_argument("--max-repeats", type=int, default=None)
    args = parser.parse_args()

    if args.mode in ("pretrain", "all"):
        pretrain(max_steps=args.max_steps)
    if args.mode in ("eval", "all"):
        run_official_eval(max_epochs=args.max_epochs, patience=args.patience,
                           finetune_predictor=args.finetune_predictor,
                           blend_interp=not args.no_blend, max_repeats=args.max_repeats)
