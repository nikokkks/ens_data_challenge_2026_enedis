"""
SoftImpute / factorisation matricielle bas-rang (idée #14 de benchmark.md,
jusqu'ici non implémentée) -- approche différente des modèles "par courbe"
(Ridge/TabPFN/XGBoost) testés jusqu'à présent : au lieu de régresser CHAQUE
courbe individuellement sur ses voisins les plus corrélés, on complète
TOUTES les courbes d'un même cluster de forme EN UNE SEULE FOIS via une
factorisation bas-rang itérative (HardImpute -- variante simplifiée de
SoftImpute, troncature dure du rang plutôt que seuillage doux des valeurs
singulières, cf. Mazumder, Hastie & Tibshirani 2010 "Spectral Regularization
Algorithms for Learning Large Incomplete Matrices").

Recherche : la littérature (SoftImpute, TRMF, GRIN) confirme que les
approches bas-rang / factorisation matricielle sont l'approche standard pour
compléter des séries temporelles corrélées et manquantes de façon groupée --
pertinent ici vu la structure du problème (69k courbes synthétiques
partageant des profils de consommation similaires).

Algorithme, par cluster (les ~25 clusters de KMeans déjà utilisés par le
pipeline V9) :
  1. Rassembler toutes les courbes du cluster (donneurs + val_holed
     appartenant à ce cluster), avec leurs vrais trous.
  2. Initialiser les valeurs manquantes par la moyenne de chaque courbe.
  3. Itérer : SVD tronquée à `rank` composantes -> reconstruction -> on
     RÉINJECTE les valeurs connues (elles ne changent jamais), seules les
     valeurs manquantes sont mises à jour par la reconstruction -> répéter
     jusqu'à convergence.
  4. Blend avec l'interpolation linéaire via le même alpha continu que les
     autres modèles du projet (position dans le trou), pour cohérence.

Même harnais que le reste du projet (feature_engineering_bench.evaluate_repeated_splits,
n_repeats=5, test_size=0.10, random_state=42, MAE officielle NaN-safe).
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

warnings.filterwarnings("ignore")

from evaluate import load_data, fill_nan_with_interpolation
import feature_engineering_bench as feb

RANDOM_STATE = 42
N_REPEATS = 5
TEST_SIZE = 0.10
N_CLUSTERS = 25


def hard_impute(X, mask_known, rank=15, n_iter=25, tol=1e-4, random_state=0):
    """X: (T, n_curves) avec NaN aux positions manquantes. mask_known: bool,
    True = valeur connue (à ne jamais modifier). Retourne la matrice complétée."""
    col_mean = np.nanmean(X, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)  # colonne 100% manquante (rare) -> 0
    X_filled = np.where(mask_known, X, np.broadcast_to(col_mean, X.shape))
    prev_missing_vals = X_filled[~mask_known].copy()

    for _ in range(n_iter):
        rk = min(rank, min(X_filled.shape) - 1)
        U, S, Vt = randomized_svd(X_filled, n_components=rk, random_state=random_state)
        recon = (U * S) @ Vt
        X_filled = np.where(mask_known, X, recon)
        cur_missing_vals = X_filled[~mask_known]
        delta = np.linalg.norm(cur_missing_vals - prev_missing_vals) / (np.linalg.norm(prev_missing_vals) + 1e-9)
        prev_missing_vals = cur_missing_vals
        if delta < tol:
            break
    return X_filled


def make_softimpute_model(rank=15, n_iter=25, n_clusters=N_CLUSTERS, blend_interp=True):
    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool_complete = pd.concat([donor_df, dev_true_df], axis=1)
        km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=n_clusters, random_state=RANDOM_STATE)
        if dev_true_df.shape[1] > 0:
            dev_filled = dev_true_df.ffill().bfill()
            Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
            dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
            pool_labels = pd.concat([donor_labels, dev_labels])
        else:
            pool_labels = donor_labels

        # cluster de chaque courbe val_holed (via son interpolation, comme le reste du pipeline)
        val_cluster = {}
        for col in val_holed_df.columns:
            interp_full = fill_nan_with_interpolation(val_holed_df[col])
            val_cluster[col] = feb.cluster_of(interp_full, km)

        preds = {c: val_holed_df[c].copy() for c in val_holed_df.columns}

        for cl in sorted(set(val_cluster.values())):
            val_cols_cl = [c for c, l in val_cluster.items() if l == cl]
            donor_cols_cl = pool_labels[pool_labels == cl].index.tolist()
            if len(donor_cols_cl) < 20:  # cluster trop petit -> repli sur tout le pool
                donor_cols_cl = list(pool_labels.index)

            block_cols = donor_cols_cl + val_cols_cl
            block = pd.concat([pool_complete[donor_cols_cl], val_holed_df[val_cols_cl]], axis=1)
            X = block.values.astype(np.float64)
            mask_known = ~np.isnan(X)

            X_completed = hard_impute(X, mask_known, rank=rank, n_iter=n_iter, random_state=RANDOM_STATE)

            for j, col in enumerate(val_cols_cl):
                j_full = len(donor_cols_cl) + j
                s = val_holed_df[col]
                missing = s.isna().values
                if not missing.any():
                    continue
                pred_softimpute = X_completed[:, j_full]

                if blend_interp:
                    interp_full = fill_nan_with_interpolation(s)
                    run_len_full = feb.run_lengths_for(missing)
                    edge_dist_full = feb.edge_distance_for(missing)
                    miss_idx = np.where(missing)[0]
                    alphas = np.array([
                        feb.continuous_alpha(run_len_full[m], edge_dist_full[m], adaptive=True) for m in miss_idx
                    ])
                    pred = alphas * interp_full.values[miss_idx] + (1 - alphas) * pred_softimpute[miss_idx]
                else:
                    pred = pred_softimpute[missing]

                filled = s.copy()
                filled.iloc[np.where(missing)[0]] = pred
                preds[col] = filled

        return pd.DataFrame(preds, index=val_holed_df.index)
    return model


if __name__ == "__main__":
    print("Chargement des données (X_train, y_train)...")
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    for rank in [10, 15, 25]:
        model = make_softimpute_model(rank=rank, n_iter=25)
        print(f"=== SoftImpute rank={rank} (+ blend interpolation) ===")
        t0 = time.time()
        scores = feb.evaluate_repeated_splits(
            X_train, y_train, donor_cols, model,
            n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )
        print(f"  total: {time.time()-t0:.1f}s\n")

        import json, os
        results = {}
        if os.path.exists("softimpute_bench_results.json"):
            results = json.load(open("softimpute_bench_results.json"))
        results[f"softimpute_rank{rank}"] = scores.tolist()
        with open("softimpute_bench_results.json", "w") as f:
            json.dump(results, f, indent=2)
