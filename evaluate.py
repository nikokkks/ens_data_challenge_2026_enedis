"""
Harness d'évaluation pour le challenge Enedis "Challenge Data" (imputation de courbes).

Principe
--------
- X_train contient 20 000 courbes "donneuses" (~complètes) + 1 000 courbes "holed_*"
  avec de vrais trous (NaN).
- y_train contient les valeurs vraies des 1 000 courbes holed_* (y compris aux positions
  manquantes dans X_train).
- La métrique officielle est une MAE calculée UNIQUEMENT sur les positions réellement
  manquantes (mask = X_holed.isna()), jamais sur les points déjà connus.

Comme on connaît déjà, pour les 1000 colonnes holed_* de X_train, à la fois le trou réel
(le masque) et la vraie valeur (y_train), on peut faire de la validation croisée fiable
SANS créer de trous artificiels : on découpe simplement l'ENSEMBLE DES COLONNES (les 1000
courbes trouées) en deux groupes à chaque répétition :
  - dev_cols  (90%) : peut servir de "donneurs" additionnels / à du tuning de modèle
  - val_cols  (10%) : sert uniquement à l'évaluation, sur son masque de trous réel

On répète ça n_repeats fois (Monte-Carlo / repeated hold-out, cf. sklearn.ShuffleSplit)
pour avoir une estimation plus robuste (moyenne +/- écart-type) qu'un simple split unique,
et donc une meilleure idée de ce que donnerait une soumission sur le leaderboard.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

DATA_DIR = "data"

# ---------------------------------------------------------------------------
# 1. Chargement des données
# ---------------------------------------------------------------------------

def load_data():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train_78VdSWL.csv", index_col=0)
    y_train = pd.read_csv(f"{DATA_DIR}/y_train_u0UkKEh.csv", index_col=0)
    X_test = pd.read_csv(f"{DATA_DIR}/X_test_XKVc4no.csv", index_col=0)
    return X_train, y_train, X_test


# ---------------------------------------------------------------------------
# 2. Métrique officielle du challenge : MAE calculée uniquement sur les trous
# ---------------------------------------------------------------------------

def official_mae(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, mask_df: pd.DataFrame) -> float:
    """MAE = moyenne des erreurs absolues, uniquement là où mask_df est True
    (c'est-à-dire là où la valeur était réellement manquante dans X).

    y_train contient lui-même une poignée de NaN (~1056 sur l'ensemble du
    dataset) : quelques positions manquantes de X_train n'ont pas de vérité
    terrain connue. Ces positions ne sont pas notables -> exclues du calcul
    plutôt que de propager un NaN et corrompre la moyenne de tout le repeat.
    """
    y_true_df = y_true_df[mask_df.columns]
    y_pred_df = y_pred_df[mask_df.columns]
    diff = (y_true_df.values - y_pred_df.values)
    m = mask_df.values & ~np.isnan(y_true_df.values)
    return float(np.abs(diff[m]).mean())


# ---------------------------------------------------------------------------
# 3. Baseline officielle du challenge : interpolation linéaire
# ---------------------------------------------------------------------------

def fill_nan_with_interpolation(column: pd.Series) -> pd.Series:
    return column.interpolate(method="linear", limit_direction="both")


def baseline_model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
    """Signature commune à tous les modèles du harness : ne regarde même pas
    les donneurs ni le dev set, exactement comme le benchmark officiel."""
    return val_holed_df.apply(fill_nan_with_interpolation, axis=0)


# ---------------------------------------------------------------------------
# 4. Modèle "KNN inter-courbes" : utilise les courbes donneuses corrélées
# ---------------------------------------------------------------------------

def make_knn_model(k=20, use_dev_as_donors=True, blend_interp=0.3):
    """
    Pour chaque courbe trouée à compléter :
      1. on calcule la corrélation de Pearson entre sa partie connue et les
         courbes du pool (donneurs + éventuellement dev_true_df), calculée
         uniquement sur les timestamps connus de la cible ;
      2. on prend les k courbes les plus corrélées ;
      3. on remet chaque courbe voisine à l'échelle (moyenne/écart-type) de la
         cible, puis on fait une moyenne pondérée par la corrélation aux
         positions manquantes ;
      4. on mélange (blend) avec l'interpolation linéaire pour lisser les
         variations locales que les voisins ne captent pas forcément.
    """

    def knn_model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = donor_df
        if use_dev_as_donors:
            pool = pd.concat([pool, dev_true_df], axis=1)
        pool_vals = pool.ffill().bfill().values.astype(np.float64)  # (T, n_pool)

        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            observed = s.notna().values
            missing = ~observed
            if not missing.any():
                preds[col] = s
                continue

            obs_idx = np.where(observed)[0]
            miss_idx = np.where(missing)[0]
            target_obs = s.values[obs_idx]

            donors_obs = pool_vals[obs_idx, :]
            t_c = target_obs - target_obs.mean()
            d_c = donors_obs - donors_obs.mean(axis=0, keepdims=True)
            num = d_c.T @ t_c
            denom = np.sqrt((d_c ** 2).sum(axis=0) * (t_c ** 2).sum()) + 1e-9
            corr = num / denom

            kk = min(k, corr.shape[0])
            top_idx = np.argpartition(-corr, kk - 1)[:kk]
            weights = np.clip(corr[top_idx], 1e-3, None)
            weights = weights / weights.sum()

            donor_sel = pool_vals[:, top_idx]
            donor_sel_obs = donor_sel[obs_idx, :]
            d_mean = donor_sel_obs.mean(axis=0)
            d_std = donor_sel_obs.std(axis=0) + 1e-9
            t_mean, t_std = target_obs.mean(), target_obs.std() + 1e-9

            donor_scaled = (donor_sel - d_mean) / d_std * t_std + t_mean
            pred_missing_knn = (donor_scaled[miss_idx, :] * weights).sum(axis=1)

            filled = s.copy()
            if blend_interp > 0:
                interp = fill_nan_with_interpolation(s).values[miss_idx]
                pred_missing = blend_interp * interp + (1 - blend_interp) * pred_missing_knn
            else:
                pred_missing = pred_missing_knn
            filled.iloc[miss_idx] = pred_missing
            preds[col] = filled

        return pd.DataFrame(preds, index=val_holed_df.index)

    return knn_model


# ---------------------------------------------------------------------------
# 5. Harness : hold-out répété 90/10, n_repeats fois
# ---------------------------------------------------------------------------

def repeated_holdout_eval(X_train, y_train, donor_cols, model_fit_predict,
                           n_repeats=5, test_size=0.10, random_state=42, verbose=True):
    holed_cols = list(y_train.columns)
    ss = ShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=random_state)
    scores = []
    for i, (dev_idx, val_idx) in enumerate(ss.split(holed_cols)):
        dev_cols = [holed_cols[j] for j in dev_idx]
        val_cols = [holed_cols[j] for j in val_idx]

        t0 = time.time()
        y_pred_val = model_fit_predict(
            donor_df=X_train[donor_cols],
            dev_holed_df=X_train[dev_cols],
            dev_true_df=y_train[dev_cols],
            val_holed_df=X_train[val_cols],
        )
        mask = X_train[val_cols].isna()
        mae = official_mae(y_train[val_cols], y_pred_val, mask)
        dt = time.time() - t0
        scores.append(mae)
        if verbose:
            print(f"  repeat {i + 1}/{n_repeats}: MAE = {mae:8.2f} W "
                  f"(n_val_cols={len(val_cols)}, {dt:.1f}s)")

    scores = np.array(scores)
    if verbose:
        print(f"  -> Mean MAE over {n_repeats} repeats: "
              f"{scores.mean():.2f} +/- {scores.std():.2f} W\n")
    return scores


# ---------------------------------------------------------------------------
# 6. Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Chargement des données...")
    X_train, y_train, X_test = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, "
          f"donors: {len(donor_cols)}, holed: {len(y_train.columns)}\n")

    N_REPEATS = 5
    TEST_SIZE = 0.10

    print(f"=== Baseline officielle (interpolation linéaire) — "
          f"{N_REPEATS} x hold-out {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} ===")
    baseline_scores = repeated_holdout_eval(
        X_train, y_train, donor_cols, baseline_model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE,
    )

    print(f"=== Modèle KNN inter-courbes (k=20, blend interp=0.3) — "
          f"{N_REPEATS} x hold-out {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)} ===")
    knn_model = make_knn_model(k=20, use_dev_as_donors=True, blend_interp=0.3)
    knn_scores = repeated_holdout_eval(
        X_train, y_train, donor_cols, knn_model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE,
    )

    print("=== Résumé ===")
    print(f"Baseline (interp. linéaire): {baseline_scores.mean():.2f} +/- {baseline_scores.std():.2f} W "
          f"(par repeat: {np.round(baseline_scores, 1).tolist()})")
    print(f"KNN inter-courbes:           {knn_scores.mean():.2f} +/- {knn_scores.std():.2f} W "
          f"(par repeat: {np.round(knn_scores, 1).tolist()})")
    gain = 100 * (1 - knn_scores.mean() / baseline_scores.mean())
    print(f"Gain relatif vs baseline: {gain:.1f}%")
