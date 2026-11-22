"""Génère une première soumission avec le pipeline TabPFN "zero-start"
(tabpfn_zero_start.py, 72 features, réimplémentation autonome sans
dépendance au reste du dépôt), validé à 63,93 +/- 17,20 W sur le protocole
réel 10x10 (cf. tabpfn_zero_start_full_full.json). Réutilise DonorPool /
predict_column tels quels ; le pool donneur est le pool maximal autorisé
(donneurs X_test + donneurs X_train + y_train), comme les autres scripts
generate_submission_tabpfn_v*.py."""
import os
import os
os.environ["TABPFN_API_KEY"] = "tabpfn_sk_rTQzro4xlLZC3RpHHxPxte5r-LMBQ2d5NS7XAib7kC0"


import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tabpfn

from evaluate import load_data
import tabpfn_zero_start as zero

warnings.filterwarnings("ignore")
OUT_PATH = "submission_tabpfn_zero_start.csv"
CHECKPOINT_PATH = "submission_tabpfn_zero_start_partial.csv"
RANDOM_STATE = 42
# Machine de bureau à 7,6 Go de RAM totale, partagée avec PyCharm/Firefox/
# GNOME (~2-3 Go déjà utilisés hors ce script) : 32 puis même 20 threads ont
# fait grimper la RAM à moins de 250 Mo disponibles à deux reprises (quasi
# OOM). Réduit à 10 pour une vraie marge, la charge restant de toute façon
# I/O-bound (attente API), pas CPU-bound.
BATCH_SIZE = 10
MAX_RETRIES = 3
CONFIG = zero.CONFIGS["full"]


def fill_column(
    column: str, series: pd.Series, index: pd.DatetimeIndex, pool: zero.DonorPool
) -> tuple[str, np.ndarray, str | None]:
    target = series.to_numpy(dtype=float)
    missing = ~np.isfinite(target)
    error = None
    prediction = None
    for attempt in range(MAX_RETRIES):
        try:
            prediction, _ = zero.predict_column(
                target, index, pool, CONFIG, RANDOM_STATE, dry_run=False
            )
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * (attempt + 1))
    if prediction is None:
        # Repli déterministe : même chemin que le mode dry-run du module
        # (médiane des voisins corrélés + blend), sans appel API, pour ne
        # jamais laisser une colonne sans prédiction.
        prediction, _ = zero.predict_column(
            target, index, pool, CONFIG, RANDOM_STATE, dry_run=True
        )
    filled = target.copy()
    filled[missing] = prediction
    return column, filled, error


if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant dans l'environnement")

    print("Chargement des données...", flush=True)
    X_train, y_train, X_test = load_data()
    test_donors = [c for c in X_test if not str(c).startswith("holed_")]
    train_donors = [c for c in X_train if not str(c).startswith("holed_")]
    holes = [c for c in X_test if str(c).startswith("holed_")]
    donor_pool_df = pd.concat([X_test[test_donors], X_train[train_donors], y_train], axis=1)
    print(f"Pool maximal : {donor_pool_df.shape[1]} donneurs", flush=True)
    pool = zero.DonorPool(donor_pool_df)
    # evaluate.load_data() ne parse pas les dates (contrairement à
    # zero.load_train) : calendar_features() exige un vrai DatetimeIndex.
    test_index = pd.DatetimeIndex(pd.to_datetime(X_test.index))

    if os.path.exists(CHECKPOINT_PATH):
        done = pd.read_csv(CHECKPOINT_PATH, index_col=0)
        done.index = X_test.index
    else:
        done = pd.DataFrame(index=X_test.index)
    remaining = [c for c in holes if c not in done]
    print(f"Reprise : {len(done.columns)} faites, {len(remaining)} restantes", flush=True)

    fallbacks = 0
    started = time.time()
    for start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[start:start + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = [
                executor.submit(fill_column, c, X_test[c], test_index, pool)
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
