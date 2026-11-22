"""Prochain levier structurel (cf. benchmark.md §11/§13, point 4 des
prochaines pistes) : remplacer le blend alpha "à la main" (continuous_alpha
+ multiplicateurs periodic/volatility, tous réglés manuellement en V14) par
un blend appris (Ridge positif) sur (interp_pred, model_pred, meta-features)
-> y_true, avec un split meta-train/meta-test FRAIS (seed 999, disjoint des
seeds 42..51 déjà utilisés pour le criblage V12/V14, donc pas de fuite)."""
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tabpfn_client
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from evaluate import load_data, official_mae, fill_nan_with_interpolation
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14

warnings.filterwarnings("ignore")
RANDOM_STATE_OOF = 999
V14_OPTS = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                extra_lags=(24,), neighbor_rolling=True)


def build_rows_for_column(col, s, y_true_col, pool_vals, km, cluster_col_idx, tfeat, index):
    """Calcule interp_pred, model_pred (TabPFN, memes features que V14) et
    meta-features pour chaque point manquant de la colonne -- une seule fois,
    reutilise ensuite a la fois pour le blend appris ET pour recalculer le
    blend V14 sur exactement les memes points (comparaison directe)."""
    o = {**v14.DEFAULT_OPTS, **V14_OPTS}
    observed = s.notna().to_numpy(); missing = ~observed
    if not missing.any():
        return None
    obs_idx, miss_idx = np.flatnonzero(observed), np.flatnonzero(missing)
    target_obs = s.to_numpy()[obs_idx]
    interp_full = fill_nan_with_interpolation(s)
    cluster = feb.cluster_of(interp_full, km)
    same_idx = cluster_col_idx.get(cluster)
    sub_vals = (pool_vals[:, same_idx]
                if same_idx is not None and len(same_idx) >= max(2 * o["k"], 30)
                else pool_vals)
    knn_pred, _, scaled = v14.weighted_knn_pred_tunable(
        sub_vals, obs_idx, miss_idx, target_obs, o["k"], k_min=o["k_min"],
        corr_threshold=o["corr_threshold"], euclid_weight=o["euclid_weight"],
        ratio_clip=o["ratio_clip"])
    lags = (48, 96, 144, 336) + tuple(o["extra_lags"])
    periodic = feb.periodic_copy_features(s.to_numpy(), lags=lags, include_summary=True,
                                           normalize=True, include_availability=True)
    seasonal = feb.same_curve_seasonality_feature(s.to_numpy(), index)
    extra = np.hstack([tfeat, seasonal, periodic])
    if o["neighbor_rolling"]:
        extra = np.hstack([extra, feb.neighbor_rolling_features(scaled)])

    model_pred = None
    error = None
    if len(obs_idx) >= 15:
        for attempt in range(v14.MAX_RETRIES):
            try:
                model_pred = feb.local_regression_pred(v14.MedianTabPFN, scaled, obs_idx,
                                                         miss_idx, target_obs, extra)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(2 * (attempt + 1))
    if model_pred is None:
        model_pred = knn_pred

    runs = feb.run_lengths_for(missing)[miss_idx]
    edges = feb.edge_distance_for(missing)[miss_idx]
    period_agree = feb.periodic_agreement(s.to_numpy())[miss_idx]
    vol = feb.curve_volatility(s.to_numpy())
    interp_pred = interp_full.to_numpy()[miss_idx]
    y_true = y_true_col.to_numpy()[miss_idx]
    valid = ~np.isnan(y_true)

    meta = np.column_stack([
        np.minimum(runs, 50) / 50.0,
        np.minimum(edges, 20) / 20.0,
        period_agree,
        np.full(len(miss_idx), min(vol, 1.0)),
    ])
    return dict(col=col, interp=interp_pred[valid], model=model_pred[valid],
                meta=meta[valid], y=y_true[valid], error=error)


def build_context(X, y, donor_cols, eval_cols, helper_cols):
    """helper_cols : colonnes holed_* dont la vérité sert de donneur
    supplémentaire (comme dev_true_df dans V12/V14) -- DOIT être disjoint de
    eval_cols (les colonnes qu'on va prédire), sinon fuite (une colonne ne
    doit jamais pouvoir se servir elle-même de donneur)."""
    assert not (set(eval_cols) & set(helper_cols)), "fuite : eval_cols recoupe helper_cols"
    pool = pd.concat([X[donor_cols], y[helper_cols]], axis=1)
    pool_vals = pool.ffill().bfill().to_numpy(dtype=np.float64)
    km, labels = feb.fit_shape_clusters(X[donor_cols], n_clusters=15, random_state=42)
    dev_filled = y[helper_cols].ffill().bfill()
    Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
    dev_labels = pd.Series(km.predict(Zdev.T.values), index=helper_cols)
    all_labels = pd.concat([labels, dev_labels])
    pos = pd.Series(np.arange(len(all_labels)), index=all_labels.index)
    cluster_col_idx = {cl: pos[idx].values for cl, idx in all_labels.groupby(all_labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(X.index, ("hour", "dow", "dom", "year"))
    return pool_vals, km, cluster_col_idx, tfeat


def collect_rows(cols, X, y, donor_cols, pool_vals, km, cluster_col_idx, tfeat, label):
    rows = []
    fallbacks = 0
    with ThreadPoolExecutor(max_workers=v14.MAX_WORKERS) as executor:
        futures = [executor.submit(build_rows_for_column, c, X[c], y[c], pool_vals, km,
                                    cluster_col_idx, tfeat, X.index) for c in cols]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r is not None:
                rows.append(r)
                if r["error"]:
                    fallbacks += 1
            if i % 20 == 0:
                print(f"  [{label}] {i}/{len(cols)} colonnes traitées", flush=True)
    print(f"  [{label}] terminé : {len(rows)} colonnes, {fallbacks} replis", flush=True)
    return rows


if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant")
    tabpfn_client.set_access_token(token)
    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]
    holed = list(y.columns)

    N_TRAIN, N_TEST, N_HELPER = 300, 100, 300
    meta_train_cols, rest = train_test_split(holed, test_size=1 - N_TRAIN/len(holed), random_state=RANDOM_STATE_OOF)
    meta_test_cols, rest2 = train_test_split(rest, test_size=1 - N_TEST/len(rest), random_state=RANDOM_STATE_OOF + 1)
    # pool de donneurs "vérité" additionnel, disjoint de meta_train ET meta_test
    # (comme dev_true_df dans V12/V14) -- sans ça une colonne peut se servir
    # elle-même de donneur quasi parfait (fuite, cf. incident précédent).
    helper_cols, _ = train_test_split(rest2, test_size=1 - N_HELPER/len(rest2), random_state=RANDOM_STATE_OOF + 2)
    assert not (set(meta_train_cols) & set(meta_test_cols) & set(helper_cols))
    print(f"meta_train={len(meta_train_cols)} meta_test={len(meta_test_cols)} helper={len(helper_cols)}", flush=True)

    pool_vals, km, cluster_col_idx, tfeat = build_context(X, y, donors, meta_train_cols + meta_test_cols, helper_cols)

    train_rows = collect_rows(meta_train_cols, X, y, donors, pool_vals, km, cluster_col_idx, tfeat, "meta_train")
    test_rows = collect_rows(meta_test_cols, X, y, donors, pool_vals, km, cluster_col_idx, tfeat, "meta_test")

    def stack(rows):
        interp = np.concatenate([r["interp"] for r in rows])
        model = np.concatenate([r["model"] for r in rows])
        meta = np.concatenate([r["meta"] for r in rows])
        yv = np.concatenate([r["y"] for r in rows])
        return interp, model, meta, yv

    interp_tr, model_tr, meta_tr, y_tr = stack(train_rows)
    interp_te, model_te, meta_te, y_te = stack(test_rows)

    # meta-modele : Ridge positif sur [interp, model, meta-features] -> y_true
    X_meta_tr = np.column_stack([interp_tr, model_tr, meta_tr])
    X_meta_te = np.column_stack([interp_te, model_te, meta_te])
    meta_model = Ridge(alpha=1.0, positive=True)
    meta_model.fit(X_meta_tr, y_tr)
    pred_learned = meta_model.predict(X_meta_te)
    mae_learned = float(np.abs(pred_learned - y_te).mean())

    # 2e meta-modele : non lineaire (HistGradientBoosting), pour capter la
    # logique par paliers de continuous_alpha qu'un Ridge linéaire ne peut
    # pas représenter -- avec 300 colonnes d'entraînement (5x plus qu'avant)
    # au lieu de 60, pour donner sa chance au blend appris.
    from sklearn.ensemble import HistGradientBoostingRegressor
    meta_model_gbm = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=300, learning_rate=.05,
        max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
        random_state=42)
    meta_model_gbm.fit(X_meta_tr, y_tr)
    pred_learned_gbm = meta_model_gbm.predict(X_meta_te)
    mae_learned_gbm = float(np.abs(pred_learned_gbm - y_te).mean())

    mae_interp_only = float(np.abs(interp_te - y_te).mean())
    mae_model_only = float(np.abs(model_te - y_te).mean())

    # blend V14 (formule a la main) recalcule sur les MEMES points de test
    v14_alpha_rows = []
    for r, col in zip(test_rows, meta_test_cols):
        pass  # l'alpha V14 depend de features par courbe entiere (runs/edges deja dans meta)
    # reconstruction directe du blend V14 a partir des memes composantes stockees
    runs_norm, edges_norm, period_agree, vol = meta_te[:, 0], meta_te[:, 1], meta_te[:, 2], meta_te[:, 3]
    runs = runs_norm * 50.0
    edges = edges_norm * 20.0
    base_alpha = np.array([feb.continuous_alpha(r, e, True) for r, e in zip(runs, edges)])
    alpha_v14 = base_alpha * (1.0 - 1.0 * period_agree)
    alpha_v14 = alpha_v14 * np.array([feb.volatility_alpha_mult(v, vol_weight=0.8, cap=0.6) for v in vol])
    alpha_v14 = np.clip(alpha_v14, 0.0, 0.95)
    pred_v14 = alpha_v14 * interp_te + (1 - alpha_v14) * model_te
    mae_v14 = float(np.abs(pred_v14 - y_te).mean())

    print(f"\n=== Comparaison sur meta_test ({len(meta_test_cols)} colonnes, split frais seed=999/1000/1001) ===")
    print(f"interpolation seule            : {mae_interp_only:8.2f} W")
    print(f"TabPFN seul (modele)           : {mae_model_only:8.2f} W")
    print(f"blend V14 (formule main)       : {mae_v14:8.2f} W")
    print(f"blend appris (Ridge OOF)       : {mae_learned:8.2f} W")
    print(f"blend appris (GBM OOF)         : {mae_learned_gbm:8.2f} W")
    print(f"\ncoeffs meta-modele Ridge: interp={meta_model.coef_[0]:.3f} model={meta_model.coef_[1]:.3f} "
          f"meta_feats={meta_model.coef_[2:]} intercept={meta_model.intercept_:.2f}")

    json.dump({
        "mae_interp_only": mae_interp_only, "mae_model_only": mae_model_only,
        "mae_v14_formula": mae_v14, "mae_learned_ridge": mae_learned,
        "mae_learned_gbm": mae_learned_gbm,
        "coef": meta_model.coef_.tolist(), "intercept": float(meta_model.intercept_),
        "n_train_cols": len(meta_train_cols), "n_test_cols": len(meta_test_cols),
    }, open("tabpfn_oof_alpha_results.json", "w"), indent=2)
    print("\nTerminé.", flush=True)
