"""Génère la soumission avec le pipeline V14 + 72 features zero-start
concaténées (tabpfn_v14_concat72.py), validé à 62,25 +/- 29,54 W sur le
protocole apparié 10x10 (5/10 victoires vs V14 seul, IC bootstrap
[-1,99 ; -0,03], cf. benchmark.md §21 et tabpfn_v14_concat72_results.json).

Garde le pipeline V14 exact (build_v14_features, MedianTabPFN, blend
alpha adaptatif) inchangé et ajoute les 72 features indépendantes de
tabpfn_zero_start.py à la matrice passée à TabPFN, comme dans le
protocole d'évaluation. Pool donneur maximal (donneurs X_test + donneurs
X_train + y_train), même convention que generate_submission_tabpfn_v14.py."""
import getpass
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tabpfn_client

from evaluate import load_data
import feature_engineering_bench as feb
import tabpfn_v14_ablation as v14
import tabpfn_zero_start as zero

warnings.filterwarnings("ignore")
OUT_PATH = "submission_tabpfn_v14_concat72.csv"
CHECKPOINT_PATH = "submission_tabpfn_v14_concat72_partial.csv"
N_CLUSTERS, RANDOM_STATE = 15, 42
BATCH_SIZE = 20
MAX_RETRIES = 3

BEST_V14 = dict(
    volatility_alpha=True,
    vol_weight=0.8,
    periodic_alpha_strength=1.0,
    extra_lags=(24,),
    neighbor_rolling=True,
)


def predict_column_concat72(
    column: str,
    series: pd.Series,
    pool_vals: np.ndarray,
    km: object,
    cluster_idx: dict,
    tfeat: np.ndarray,
    raw_index,
    dt_index: pd.DatetimeIndex,
    independent_pool: zero.DonorPool,
) -> tuple[str, pd.Series, str | None]:
    built = v14.build_v14_features(column, series, pool_vals, km, cluster_idx, tfeat, raw_index, BEST_V14)
    if built is None:
        return column, series.copy(), None
    obs_idx, miss_idx, target_obs, interp_full, knn_pred, scaled, extra, missing, lags = built

    zero_frame, _, _, _ = zero.build_features(
        dt_index, series.to_numpy(dtype=float), independent_pool, zero.CONFIGS["full"]
    )
    zero_values = zero_frame.to_numpy(dtype=np.float32)
    design = np.hstack([scaled, extra, zero_values])

    prediction = None
    error = None
    if len(obs_idx) >= 15:
        for attempt in range(MAX_RETRIES):
            try:
                regressor = v14.MedianTabPFN(random_state=RANDOM_STATE)
                regressor.fit(design[obs_idx], target_obs)
                prediction = regressor.predict(design[miss_idx])
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(2 * (attempt + 1))
    if prediction is None:
        prediction = knn_pred

    # Blend final V14 inchangé (alpha adaptatif durée trou / bord / accord
    # périodique / volatilité).
    runs = feb.run_lengths_for(missing)
    edges = feb.edge_distance_for(missing)
    alpha = np.array([feb.continuous_alpha(runs[i], edges[i], True) for i in miss_idx])
    periodic_agreement = feb.periodic_agreement(series.to_numpy(dtype=float), lags=lags)[miss_idx]
    alpha *= 1.0 - BEST_V14["periodic_alpha_strength"] * periodic_agreement
    volatility = feb.curve_volatility(series.to_numpy(dtype=float))
    alpha *= feb.volatility_alpha_mult(volatility, vol_weight=BEST_V14["vol_weight"], cap=0.6)
    alpha = np.clip(alpha, 0.0, 0.95)
    blended = alpha * interp_full.to_numpy()[miss_idx] + (1.0 - alpha) * np.asarray(prediction)

    filled = series.copy()
    filled.iloc[miss_idx] = blended
    return column, filled, error


if __name__ == "__main__":
    # os.environ d'abord (usage local classique, `export TABPFN_API_KEY=...`) ;
    # sinon saisie interactive (getpass masque l'entrée, fonctionne aussi bien
    # dans un notebook Colab que dans un terminal). Fixer le token explicitement
    # ici évite que tabpfn_client bascule sur son flux de login navigateur --
    # surtout critique en multithreading, où chaque thread sans token déclenche
    # sa propre invite interactive en parallèle.
    token = os.environ.get("TABPFN_API_KEY") or getpass.getpass(
        "TABPFN_API_KEY introuvable dans l'environnement -- collez votre clé : "
    )
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant")
    tabpfn_client.set_access_token(token)
    print("Chargement des données...", flush=True)
    X_train, y_train, X_test = load_data()
    test_donors = [c for c in X_test if not str(c).startswith("holed_")]
    train_donors = [c for c in X_train if not str(c).startswith("holed_")]
    holes = [c for c in X_test if str(c).startswith("holed_")]
    pool = pd.concat([X_test[test_donors], X_train[train_donors], y_train], axis=1)
    print(f"Pool maximal : {pool.shape[1]} donneurs", flush=True)

    if os.path.exists(CHECKPOINT_PATH):
        done = pd.read_csv(CHECKPOINT_PATH, index_col=0)
        done.index = X_test.index
    else:
        done = pd.DataFrame(index=X_test.index)
    remaining = [c for c in holes if c not in done]
    print(f"Reprise : {len(done.columns)} faites, {len(remaining)} restantes", flush=True)

    km, labels = feb.fit_shape_clusters(pool, N_CLUSTERS, RANDOM_STATE)
    positions = pd.Series(np.arange(len(labels)), index=labels.index)
    cluster_idx = {cl: positions[idx].to_numpy() for cl, idx in labels.groupby(labels).groups.items()}
    tfeat = feb.time_features_full_seasonality(X_test.index, groups=("hour", "dow", "dom", "year"))

    # Un seul pool robuste-scalé (float32) partagé par les deux blocs de
    # features, comme dans tabpfn_v14_concat72.prepare_split.
    independent_pool = zero.DonorPool(pool.ffill().bfill())
    pool_vals = independent_pool.raw
    dt_index = pd.DatetimeIndex(pd.to_datetime(X_test.index))

    fallbacks = 0
    started = time.time()
    for start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[start:start + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = [
                executor.submit(
                    predict_column_concat72,
                    c, X_test[c], pool_vals, km, cluster_idx, tfeat, X_test.index, dt_index, independent_pool,
                )
                for c in batch
            ]
            for future in as_completed(futures):
                col, filled, error = future.result()
                done[col] = filled
                if error:
                    fallbacks += 1
                    print(f"Repli KNN pour {col}: {error}", flush=True)
        done.to_csv(CHECKPOINT_PATH)
        completed = start + len(batch)
        elapsed = time.time() - started
        eta = elapsed / completed * (len(remaining) - completed) / 60
        rate = completed / elapsed if elapsed else 0.0
        print(
            f"[{len(done.columns)}/1000] lot={len(batch)} parallèle={BATCH_SIZE} "
            f"temps={elapsed/60:.1f} min vitesse={rate*60:.1f} col/min "
            f"ETA={eta:.1f} min replis={fallbacks}",
            flush=True,
        )

    final = done[holes].clip(lower=0)
    final.index.name = "Horodate"
    sample = pd.read_csv("data/new_output_sample.csv", index_col=0)
    assert final.shape == sample.shape == (1057, 1000)
    assert list(final.columns) == list(sample.columns)
    assert final.isna().sum().sum() == 0
    assert np.isfinite(final.to_numpy()).all()
    final.to_csv(OUT_PATH)
    print(f"Soumission écrite : {OUT_PATH}; replis KNN={fallbacks}", flush=True)
