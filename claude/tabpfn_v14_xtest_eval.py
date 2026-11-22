"""Évaluation de TabPFN V14 en n'utilisant QUE les données de X_test
(aucune lecture de X_train ni de y_train) -- demandé pour vérifier la
performance du pipeline validé (cf. README.md, generate_submission_tabpfn_v14.py)
indépendamment des courbes donneuses de X_train.

Pourquoi pas evaluate.repeated_holdout_eval directement
---------------------------------------------------------
Ce harness marche sur X_train parce qu'on y connaît, pour les 1000 colonnes
holed_*, à la fois le VRAI masque de trous et y_train (vérité terrain) : on
peut donc découper les COLONNES en dev/val et scorer directement sur le vrai
masque. Pour X_test, on n'a pas de vérité terrain sur ses propres colonnes
holed_* -- ce sont justement celles à soumettre au challenge. Impossible donc
d'y calculer une MAE de la même façon.

Protocole utilisé ici (masquage synthétique sur les points déjà connus)
--------------------------------------------------------------------
1. Donneurs = uniquement les colonnes non "holed_*" de X_test (même pool que
   `test_donors` dans generate_submission_tabpfn_v14.py, sans y ajouter les
   donneurs de X_train ni y_train).
2. Pour chacune des 1000 colonnes holed_* de X_test, on cache artificiellement
   une partie de ses valeurs DÉJÀ CONNUES (non-NaN), par blocs dont la
   longueur est tirée de la distribution réelle des longueurs de trous
   observée sur les 1000 colonnes holed_* elles-mêmes (`real_run_length_pool`)
   -- pour rester représentatif de la vraie difficulté (trous longs vs points
   isolés) plutôt que de ne cacher que des points isolés, beaucoup plus
   faciles à interpoler et qui fausseraient les features de contexte de trou
   (gap_alpha, edge_distance, périodicité...) utilisées par le pipeline V14.
3. On complète la colonne (vrai masque + masque synthétique, tous deux NaN
   pour le modèle) avec le pipeline V14 exact (`predict_column_v14`, réutilisé
   sans modification depuis tabpfn_v14_ablation.py) sur ce pool restreint à
   X_test.
4. La MAE est calculée UNIQUEMENT aux positions masquées synthétiquement
   (jamais sur le vrai masque, dont la vérité est inconnue pour X_test).
5. On répète (masque différent à chaque repeat, seed dérivée de
   --random-state) pour obtenir une moyenne +/- écart-type, dans le même
   esprit que evaluate.repeated_holdout_eval / tabpfn_v14_ablation.run_config,
   mais adapté puisqu'ici on ne peut pas découper les COLONNES en dev/val
   (aucune vérité terrain sur aucune d'entre elles) -- on découpe donc les
   POINTS CONNUS de chaque colonne.

Checkpointé comme les autres scripts generate_submission_*.py / *_bench.py :
écrit tabpfn_v14_xtest_eval_progress.json (dans ce même dossier claude/)
après chaque colonne, et saute au redémarrage les colonnes déjà faites pour
un repeat donné -- utile car TabPFN a un quota d'appels quotidien (déjà atteint
une fois sur ce pool, cf. generate_submission_v14.log : ~176 replis KNN après
épuisement du quota lors d'un run complet sur 1000 colonnes).

Usage
-----
    export TABPFN_API_KEY=...
    .venv/bin/python claude/tabpfn_v14_xtest_eval.py
    .venv/bin/python claude/tabpfn_v14_xtest_eval.py --n-cols 100 --n-repeats 1   # run rapide
"""
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tabpfn_client

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import feature_engineering_bench as feb  # noqa: E402
from tabpfn_v14_ablation import predict_column_v14  # noqa: E402

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(REPO_ROOT, "data", "X_test_XKVc4no.csv")
HERE = os.path.dirname(os.path.abspath(__file__))
PROGRESS_PATH = os.path.join(HERE, "tabpfn_v14_xtest_eval_progress.json")
RESULTS_PATH = os.path.join(HERE, "tabpfn_v14_xtest_eval_results.json")

N_CLUSTERS = 15
# Config V14 validée (cf. README.md / generate_submission_tabpfn_v14.py) :
# inchangée ici, on ne teste QUE le changement de pool de donneurs.
V14_OPTS = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                extra_lags=(24,), neighbor_rolling=True)


# ---------------------------------------------------------------------------
# 1. Chargement -- X_test UNIQUEMENT
# ---------------------------------------------------------------------------

def load_x_test_only():
    """Charge exclusivement X_test. X_train et y_train ne sont jamais lus."""
    return pd.read_csv(DATA_PATH, index_col=0)


# ---------------------------------------------------------------------------
# 2. Distribution empirique des longueurs de trous réels (pour un masquage
#    synthétique réaliste, par blocs plutôt que par points isolés)
# ---------------------------------------------------------------------------

def extract_run_lengths(bool_mask):
    lengths = []
    n = len(bool_mask)
    i = 0
    while i < n:
        if bool_mask[i]:
            j = i
            while j < n and bool_mask[j]:
                j += 1
            lengths.append(j - i)
            i = j
        else:
            i += 1
    return lengths


def real_run_length_pool(holed_df):
    lengths = []
    mask = holed_df.isna().to_numpy()
    for j in range(mask.shape[1]):
        lengths.extend(extract_run_lengths(mask[:, j]))
    return np.array(lengths, dtype=int) if lengths else np.array([1], dtype=int)


def make_synthetic_mask(observed, run_length_pool, target_frac, rng):
    """Cache par blocs une fraction ~target_frac des positions actuellement
    observées, sans jamais chevaucher un vrai trou ni un bloc déjà posé."""
    n = len(observed)
    n_obs = int(observed.sum())
    target = int(round(target_frac * n_obs))
    synthetic = np.zeros(n, dtype=bool)
    if target <= 0 or n_obs == 0:
        return synthetic
    avail = observed.copy()
    count = 0
    attempts = 0
    max_attempts = max(200, target * 30)
    while count < target and attempts < max_attempts:
        attempts += 1
        length = int(rng.choice(run_length_pool))
        length = max(1, min(length, n))
        start = int(rng.integers(0, n - length + 1))
        end = start + length
        if avail[start:end].all():
            synthetic[start:end] = True
            avail[start:end] = False
            count += length
    return synthetic


# ---------------------------------------------------------------------------
# 3. Évaluation d'une colonne (une répétition)
# ---------------------------------------------------------------------------

def eval_column(col, series, run_length_pool, mask_frac, seed,
                 pool_vals, km, cluster_col_idx, tfeat, index, opts):
    rng = np.random.default_rng(seed)
    observed = series.notna().to_numpy()
    synth_mask = make_synthetic_mask(observed, run_length_pool, mask_frac, rng)
    if not synth_mask.any():
        return col, None, None, 0

    true_vals = series.to_numpy()[synth_mask].copy()
    work = series.copy()
    work.iloc[synth_mask] = np.nan

    _, filled, error = predict_column_v14(
        col, work, pool_vals, km, cluster_col_idx, tfeat, index, opts)
    pred_vals = filled.to_numpy()[synth_mask]
    abs_err = np.abs(pred_vals - true_vals)
    return col, abs_err, error, int(synth_mask.sum())


# ---------------------------------------------------------------------------
# 4. Boucle principale : n_repeats x (jusqu'à 1000 colonnes), checkpointée
# ---------------------------------------------------------------------------

def save_progress(progress):
    tmp = PROGRESS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f)
    os.replace(tmp, PROGRESS_PATH)


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {}


def run_repeat(rep_idx, seed, holed_cols_sorted, cols, X_test, run_length_pool,
                mask_frac, pool_vals, km, cluster_col_idx, tfeat, index, opts,
                batch_size, max_workers, progress):
    rep_key = f"repeat_{rep_idx}"
    rep_progress = progress.setdefault(rep_key, {})
    col_to_idx = {c: i for i, c in enumerate(holed_cols_sorted)}
    remaining = [c for c in cols if c not in rep_progress]
    print(f"[repeat {rep_idx}] seed={seed} {len(rep_progress)} colonnes déjà faites, "
          f"{len(remaining)} restantes / {len(cols)}", flush=True)

    fallbacks = 0
    t0 = time.time()
    for start in range(0, len(remaining), batch_size):
        batch = remaining[start:start + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    eval_column, c, X_test[c], run_length_pool, mask_frac,
                    seed * 10_000 + col_to_idx[c], pool_vals, km,
                    cluster_col_idx, tfeat, index, opts): c
                for c in batch
            }
            for fut in as_completed(futures):
                c = futures[fut]
                col, abs_err, error, n_pts = fut.result()
                if abs_err is None:
                    rep_progress[col] = {"sum_abs_err": 0.0, "n_points": 0, "fallback": False}
                else:
                    rep_progress[col] = {
                        "sum_abs_err": float(abs_err.sum()),
                        "n_points": n_pts,
                        "fallback": bool(error),
                    }
                if error:
                    fallbacks += 1
                    print(f"  repli KNN pour {col}: {error}", flush=True)
        save_progress(progress)
        done = len(rep_progress)
        elapsed = time.time() - t0
        rate = (start + len(batch)) / elapsed * 60 if elapsed else 0.0
        print(f"  [repeat {rep_idx}] {done}/{len(cols)} colonnes, "
              f"{elapsed/60:.1f} min, {rate:.1f} col/min, replis={fallbacks}", flush=True)

    sum_err = sum(rep_progress[c]["sum_abs_err"] for c in cols)
    n_pts = sum(rep_progress[c]["n_points"] for c in cols)
    n_fb = sum(1 for c in cols if rep_progress[c].get("fallback"))
    mae = sum_err / n_pts if n_pts else float("nan")
    print(f"-> [repeat {rep_idx}] MAE = {mae:.2f} W sur {n_pts} points synthétiques "
          f"masqués ({len(cols)} colonnes, {n_fb} replis KNN)\n", flush=True)
    return mae, n_pts, n_fb


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cols", type=int, default=1000,
                         help="Nb de colonnes holed_* évaluées par repeat (défaut: 1000, toutes).")
    parser.add_argument("--n-repeats", type=int, default=3,
                         help="Nb de répétitions avec masques synthétiques différents (défaut: 3).")
    parser.add_argument("--mask-frac", type=float, default=0.15,
                         help="Fraction des points déjà connus cachée par colonne (défaut: 0.15, "
                              "proche de la fraction moyenne réellement manquante ~0.122).")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant dans l'environnement")
    tabpfn_client.set_access_token(token)

    print("Chargement de X_test uniquement (ni X_train ni y_train)...", flush=True)
    X_test = load_x_test_only()
    donor_cols = [c for c in X_test.columns if not str(c).startswith("holed_")]
    holed_cols_sorted = sorted(c for c in X_test.columns if str(c).startswith("holed_"))
    print(f"X_test: {X_test.shape}, donneurs={len(donor_cols)}, "
          f"colonnes holed_*={len(holed_cols_sorted)}", flush=True)

    run_length_pool = real_run_length_pool(X_test[holed_cols_sorted])
    print(f"Distribution des longueurs de trous réels (X_test) : "
          f"n={len(run_length_pool)}, médiane={np.median(run_length_pool):.0f}, "
          f"moyenne={run_length_pool.mean():.1f}, max={run_length_pool.max()}", flush=True)

    donor_df = X_test[donor_cols]
    pool_vals = donor_df.ffill().bfill().to_numpy(dtype=np.float64)
    print("Clustering de forme sur les donneurs X_test...", flush=True)
    km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=args.n_clusters,
                                               random_state=args.random_state)
    positions = pd.Series(np.arange(len(donor_labels)), index=donor_labels.index)
    cluster_col_idx = {cl: positions[idx].to_numpy()
                        for cl, idx in donor_labels.groupby(donor_labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(X_test.index, groups=("hour", "dow", "dom", "year"))

    progress = load_progress()
    n_cols = min(args.n_cols, len(holed_cols_sorted))

    repeat_stats = []
    for i in range(args.n_repeats):
        seed = args.random_state + i
        rng_cols = np.random.default_rng(seed)
        cols = (list(rng_cols.choice(holed_cols_sorted, size=n_cols, replace=False))
                if n_cols < len(holed_cols_sorted) else list(holed_cols_sorted))
        mae, n_pts, n_fb = run_repeat(
            i, seed, holed_cols_sorted, cols, X_test, run_length_pool, args.mask_frac,
            pool_vals, km, cluster_col_idx, tfeat, X_test.index, V14_OPTS,
            args.batch_size, args.max_workers, progress)
        repeat_stats.append({"repeat": i, "seed": seed, "mae": mae,
                              "n_points": n_pts, "n_cols": len(cols), "fallbacks": n_fb})

    maes = np.array([r["mae"] for r in repeat_stats])
    print("=== Résumé ===")
    print(f"TabPFN V14, pool de donneurs = X_test uniquement ({len(donor_cols)} donneurs)")
    print(f"mask_frac={args.mask_frac}, n_cols={n_cols}, n_repeats={args.n_repeats}")
    print(f"MAE par repeat: {np.round(maes, 2).tolist()}")
    print(f"MAE = {maes.mean():.2f} +/- {maes.std():.2f} W")

    results = {
        "protocol": "synthetic masking on X_test-only observed points (no X_train/y_train)",
        "opts": V14_OPTS,
        "n_clusters": args.n_clusters,
        "mask_frac": args.mask_frac,
        "n_cols": n_cols,
        "n_repeats": args.n_repeats,
        "random_state": args.random_state,
        "n_donors": len(donor_cols),
        "repeats": repeat_stats,
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std()),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Résultats écrits dans {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
