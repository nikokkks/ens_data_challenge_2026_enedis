"""5 améliorations testées pour tabpfn_column_individual_bench.py, sur le
MÊME protocole apparié (5 repeats, mêmes graines SEED+repeat, mêmes
train/val cols via bench.split_columns) que le baseline déjà enregistré
dans tabpfn_column_individual_results.json (mean 68.17 W, scores
[135.11, 69.90, 55.05, 34.29, 46.49] pour les repeats 0-4).

Chaque idée est ablatée INDÉPENDAMMENT contre ce même baseline (une seule
variable change à la fois), dans le même esprit que les criblages V14
(benchmark.md §13) : isoler l'effet avant d'envisager de les combiner.

Idée 1 (fix_alignment) : corrige un bug de désalignement de colonnes
découvert en instrumentant le pipeline (voir commentaire sur
EXTRA_WIDTH/K_NEIGHBORS ci-dessous) -- change build_curve_rows.
Idées 2, 3, 5 : ne changent que le régresseur TabPFN (hyperparamètres/
ensemble), features identiques au baseline (bug de désalignement inclus,
pour isoler l'effet du régresseur seul).
Idée 4 : ne change que la sélection des lignes de contexte (restreintes au
cluster de forme de chaque colonne de validation), features et régresseur
identiques au baseline.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import load_data
import feature_engineering_bench as feb
import tabfm_lora_finetune as ft
import tabpfn_v14_ablation as v14
import tabpfn_column_individual_bench as bench

RESULTS_PATH = Path("tabpfn_column_individual_improvements_results.json")
BASELINE_SCORES = [135.1054131454632, 69.90134214727402, 55.0476892454131,
                    34.291273130317, 46.49315804657104]  # tabpfn_column_individual_results.json
MAX_RETRIES = 3
K_NEIGHBORS = v14.DEFAULT_OPTS["k"]  # 40 -- the neighbor-selection cap used by build_v14_features


def measure_extra_width(X, y, context) -> int:
    """Idée 1 repose sur un fait mesuré, pas supposé: le bloc de features
    'extra' (calendrier + saisonnalité propre + copies périodiques +
    consensus voisin glissant) a une largeur CONSTANTE (35 colonnes pour
    V14_FEATURE_OPTS) quel que soit le nombre de voisins retenus pour une
    courbe donnée (5 à 40 selon le seuil de corrélation). C'est justement
    ce que build_curve_rows original ne respecte pas: voir docstring de
    build_curve_rows_fixed plus bas."""
    pool_vals, km, cluster_idx, tfeat = context
    for col in y.columns:
        built = v14.build_v14_features(col, X[col], pool_vals, km, cluster_idx,
                                       tfeat, X.index, ft.V14_FEATURE_OPTS)
        if built is None:
            continue
        _, _, _, _, _, _, extra, _, _ = built
        return extra.shape[1]
    raise RuntimeError("No column produced a valid feature matrix")


def build_curve_rows_fixed(col, X, y, context, extra_width: int):
    """Idée 1 : corrige un désalignement de colonnes présent dans
    tabpfn_column_individual_bench.build_curve_rows.

    ft.feature_matrix() renvoie hstack([scaled_neighbors, extra]) où
    scaled_neighbors a une largeur VARIABLE par courbe (5 à 40 voisins
    retenus selon le seuil de corrélation -- mesuré: holed_1 en a 13,
    la plupart en ont 40) alors que extra (calendrier/saisonnalité/
    copies périodiques/consensus voisin) a toujours exactement
    `extra_width` colonnes.

    build_curve_rows original fait `matrix[q, :BASE_FEATURES]` (72 cols)
    puis pad avec des zéros à droite si la courbe est plus étroite. Donc
    pour une courbe à 40 voisins, la colonne 50 du vecteur final contient
    une valeur de calendrier (position 50-40=10 dans le bloc extra) ;
    pour une courbe à 13 voisins, cette MÊME colonne 50 contient un zéro
    de padding (le bloc extra pour cette courbe se termine à la colonne
    13+35=48). TabPFN reçoit donc, dans un seul fit(), des lignes
    provenant de ~100 courbes différentes où la colonne j n'a pas la même
    signification d'une ligne à l'autre -- un bruit de désalignement pur,
    ajouté au-dessus du signal réel. Vérifié empiriquement (voir
    measure_extra_width) : extra_width est constant (35), seul le nombre
    de voisins varie (5 à 40).

    Le fix : les features 'extra' occupent TOUJOURS les colonnes
    [0:extra_width] ; le bloc voisins occupe TOUJOURS les colonnes
    [extra_width:extra_width+K_NEIGHBORS], complété par des zéros à
    l'intérieur de CE bloc seulement (jamais après les features extra).
    """
    source = X[col]
    q = ft.real_gap_indices(source, y[col])
    if not len(q):
        return None
    built = ft.feature_matrix(col, source, *context, source.index)
    if built is None:
        return None
    matrix, _, _, interp, missing, _ = built
    n_neighbors = matrix.shape[1] - extra_width
    if n_neighbors < 0:
        raise RuntimeError(f"{col}: extra_width assumption violated "
                           f"(matrix width={matrix.shape[1]}, expected>={extra_width})")
    neighbors = matrix[:, :n_neighbors]
    extra = matrix[:, n_neighbors:]
    if n_neighbors < K_NEIGHBORS:
        neighbors = np.pad(neighbors, ((0, 0), (0, K_NEIGHBORS - n_neighbors)))
    else:
        neighbors = neighbors[:, :K_NEIGHBORS]
    aligned = np.hstack([extra, neighbors])  # extra pinned first, always
    base = aligned[q, :]
    center, scale = bench.robust_location_scale(source)
    profile = np.repeat(
        bench.curve_profile(source, source.index, center, scale)[None], len(q), axis=0)
    geometry = bench.point_geometry(source, q, interp.to_numpy(float), center, scale)
    features = np.hstack([base, profile, geometry]).astype(np.float32)
    target = ((y[col].iloc[q].to_numpy(float) - center) / scale).astype(np.float32)
    return {"col": col, "X": features, "y": target, "q": q, "center": center,
            "scale": scale, "interp": interp.to_numpy(float), "missing": missing}


class ThinkingTabPFN:
    """Idée 2 : mode 'thinking' du client TabPFN, jamais utilisé ailleurs
    dans ce dépôt (grep confirmé), avec la métrique d'optimisation alignée
    sur la métrique officielle du challenge (MAE). Budgété via
    thinking_timeout_s (mesuré : un fit sans budget dépasse 2 min sans
    terminer, impraticable pour 5 repeats x 5 variantes)."""
    def __init__(self, seed, timeout_s=45.0):
        from tabpfn_client import TabPFNRegressor
        self.reg = TabPFNRegressor(random_state=seed, thinking_mode=True,
                                   thinking_metric="mean_absolute_error",
                                   thinking_timeout_s=timeout_s)

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X, output_type="median"):
        return self.reg.predict(X, output_type=output_type)


class MeanMedianBlendTabPFN:
    """Idée 2bis (remplace l'idée 2 thinking_mode, voir NOTE ci-dessous) :
    moyenne des sorties output_type='mean' et 'median' de TabPFN plutôt que
    la médiane pure utilisée partout ailleurs dans ce dépôt -- motif déjà
    codé pour V14 (`tabpfn_v14_ablation.MeanMedianBlendTabPFN`, idée #15 de
    son criblage) mais jamais appliqué à ce pipeline column-individual.

    NOTE : l'idée 2 initialement retenue (thinking_mode=True,
    thinking_metric='mean_absolute_error') a été tentée en premier mais
    abandonnée pour raison OPÉRATIONNELLE, pas de MAE : un seul fit()
    dépassait 2 minutes sans terminer (mesuré, cf. smoke test), y compris
    avec thinking_timeout_s=45 qui n'a pas raccourci l'attente observée --
    impraticable dans le budget de cette session (5 repeats x 5 variantes)."""
    def __init__(self, seed, mean_weight=0.5):
        from tabpfn_client import TabPFNRegressor
        self.reg = TabPFNRegressor(random_state=seed)
        self.mean_weight = mean_weight

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X, output_type="median"):
        med = self.reg.predict(X, output_type="median")
        mean = self.reg.predict(X, output_type="mean")
        return self.mean_weight * mean + (1 - self.mean_weight) * med


class WideEnsembleTabPFN:
    """Idée 3 : relève n_estimators (ensemble interne de 'prompts' TabPFN)
    de 8 (défaut, jamais changé dans ce dépôt -- grep confirmé) à 32."""
    def __init__(self, seed, n_estimators=32):
        from tabpfn_client import TabPFNRegressor
        self.reg = TabPFNRegressor(random_state=seed, n_estimators=n_estimators)

    def fit(self, X, y):
        self.reg.fit(X, y)
        return self

    def predict(self, X, output_type="median"):
        return self.reg.predict(X, output_type=output_type)


class SeedEnsembleTabPFN:
    """Idée 5 : médiane de 2 TabPFNRegressor à graines différentes (motif
    déjà codé pour V14 dans tabpfn_v14_ablation.SeedEnsembleTabPFN mais
    jamais testé sur ce pipeline column-individual)."""
    def __init__(self, seed, offsets=(0, 10_000)):
        from tabpfn_client import TabPFNRegressor
        self.regs = [TabPFNRegressor(random_state=seed + off) for off in offsets]

    def fit(self, X, y):
        for r in self.regs:
            r.fit(X, y)
        return self

    def predict(self, X, output_type="median"):
        preds = np.vstack([r.predict(X, output_type=output_type) for r in self.regs])
        return np.median(preds, axis=0)


def fit_predict_with_retries(make_regressor, seed, Xfit, yfit, val_rows, X, y):
    error = None
    for attempt in range(MAX_RETRIES):
        try:
            reg = make_regressor(seed)
            reg.fit(Xfit, yfit)
            return bench.predict_and_score(reg, val_rows, X, y)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 == MAX_RETRIES:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(error)


def run_cluster_restricted(train_rows, val_rows, km, repeat, X, y):
    """Idée 4 : au lieu d'un fit TabPFN unique partagé par les 100 courbes
    d'entraînement pour TOUTES les colonnes de validation, restreint le
    contexte d'entraînement de chaque colonne de validation aux courbes
    d'entraînement du MÊME cluster de forme (repli sur le pool complet si
    moins de 20 courbes d'entraînement partagent ce cluster -- même seuil
    de repli que build_v14_features)."""
    def cluster_label(row):
        return feb.cluster_of(pd.Series(row["interp"]), km)

    train_cl = {id(r): cluster_label(r) for r in train_rows}
    val_cl = {id(r): cluster_label(r) for r in val_rows}
    groups = {}
    for r in val_rows:
        groups.setdefault(val_cl[id(r)], []).append(r)

    total_err_sum = 0.0
    total_points = 0
    for cl, rows in groups.items():
        same = [r for r in train_rows if train_cl[id(r)] == cl]
        pool = same if len(same) >= 20 else train_rows
        Xfit, yfit = bench.balanced_context(pool, bench.MAX_TRAIN_ROWS, bench.SEED + repeat)
        make_regressor = lambda seed: __import__("tabpfn_client").TabPFNRegressor(random_state=seed)
        mae, _, n_points = fit_predict_with_retries(
            make_regressor, bench.SEED + repeat, Xfit, yfit, rows, X, y)
        total_err_sum += mae * n_points
        total_points += n_points
    return total_err_sum / total_points, total_points


def run_variant(name, repeats, build_rows_fn, X, y, context, make_regressor=None,
                cluster_restrict=False):
    print(f"\n=== {name} ===", flush=True)
    km = context[1]
    cache = {}
    scores = []
    for repeat in range(repeats):
        started = time.time()
        train_cols, val_cols = bench.split_columns(
            y.columns, repeat, bench.N_TRAIN_CURVES, bench.N_VAL_COLS)
        needed = train_cols + val_cols
        for col in needed:
            if col not in cache:
                cache[col] = build_rows_fn(col, X, y, context)
        train_rows = [cache[c] for c in train_cols if cache[c] is not None]
        val_rows = [cache[c] for c in val_cols if cache[c] is not None]
        if cluster_restrict:
            mae, n_points = run_cluster_restricted(train_rows, val_rows, km, repeat, X, y)
        else:
            Xfit, yfit = bench.balanced_context(train_rows, bench.MAX_TRAIN_ROWS, bench.SEED + repeat)
            mae, _, n_points = fit_predict_with_retries(
                make_regressor, bench.SEED + repeat, Xfit, yfit, val_rows, X, y)
        dt = time.time() - started
        print(f"[{name}] repeat {repeat+1}/{repeats}: MAE={mae:.2f} W ({dt:.1f}s, n={n_points})",
              flush=True)
        scores.append(mae)
    return scores


def main():
    if not os.environ.get("TABPFN_API_KEY"):
        raise SystemExit("TABPFN_API_KEY manquant")
    import tabpfn_client
    tabpfn_client.set_access_token(os.environ["TABPFN_API_KEY"])

    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]
    context = ft.build_donor_context(X, donors)
    extra_width = measure_extra_width(X, y, context)
    print(f"Mesuré: extra_width={extra_width}, K_NEIGHBORS={K_NEIGHBORS}", flush=True)

    default_regressor = lambda seed: __import__("tabpfn_client").TabPFNRegressor(random_state=seed)

    variants = {
        "1_fix_alignment": dict(
            build_rows_fn=lambda col, X, y, ctx: build_curve_rows_fixed(col, X, y, ctx, extra_width),
            make_regressor=default_regressor),
        "2_mean_median_blend": dict(
            build_rows_fn=bench.build_curve_rows,
            make_regressor=lambda seed: MeanMedianBlendTabPFN(seed)),
        "3_n_estimators_32": dict(
            build_rows_fn=bench.build_curve_rows,
            make_regressor=lambda seed: WideEnsembleTabPFN(seed)),
        "4_cluster_restricted_context": dict(
            build_rows_fn=bench.build_curve_rows,
            cluster_restrict=True),
        "5_seed_ensemble": dict(
            build_rows_fn=bench.build_curve_rows,
            make_regressor=lambda seed: SeedEnsembleTabPFN(seed)),
    }

    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    for name, cfg in variants.items():
        if name in results:
            print(f"skip {name} (déjà fait)", flush=True)
            continue
        scores = run_variant(name, len(BASELINE_SCORES), cfg["build_rows_fn"], X, y, context,
                             cfg.get("make_regressor"), cfg.get("cluster_restrict", False))
        results[name] = {"scores": scores, "mean": float(np.mean(scores)),
                          "std": float(np.std(scores)),
                          "delta_vs_baseline": float(np.mean(scores) - np.mean(BASELINE_SCORES)),
                          "wins": int(np.sum(np.array(scores) < np.array(BASELINE_SCORES)))}
        tmp = str(RESULTS_PATH) + ".tmp"
        Path(tmp).write_text(json.dumps({"baseline_scores": BASELINE_SCORES,
                                          "baseline_mean": float(np.mean(BASELINE_SCORES)),
                                          "variants": results}, indent=2))
        os.replace(tmp, RESULTS_PATH)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
