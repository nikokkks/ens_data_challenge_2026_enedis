"""
Round 3 : pistes "drastiques" issues de recherche (D1-D10 proposées dans la
conversation), construites sur T1 (V9 + XGBoost, objectif MAE natif = le
meilleur du round 1). Même harnais officiel (n=5 hold-out 90/10 indépendants,
graines différentes, MAE officielle NaN-safe).

Implémentées ici : D8 (log1p), D2 (bagging), D4 (boosting sur résidus
d'interpolation), D6 (lissage post-hoc), D1 (NMF au lieu de SVD pour
SoftImpute), D3 (stacking global interp+XGBoost+SoftImpute).
Reportées (documentées dans xgboost_bench_round2.py) : D5, D7, D9.
"""

import time
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

from evaluate import load_data, fill_nan_with_interpolation, official_mae
import feature_engineering_bench as feb
import softimpute_bench as sib

RANDOM_STATE = 42
N_REPEATS = 5
TEST_SIZE = 0.10
K = 20
N_CLUSTERS = 25
X1_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, objective="reg:absoluteerror")


# --- D8 : transformation log1p (cible non-negative apres remise a l'echelle,
# shift de securite si jamais negative) ---
class Log1pXGB:
    def __init__(self, **xgb_kwargs):
        self.xgb_kwargs = xgb_kwargs

    def fit(self, X, y):
        self._offset = -y.min() + 1.0 if y.min() < 0 else 0.0
        self.reg = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, **self.xgb_kwargs)
        self.reg.fit(X, np.log1p(y + self._offset))
        return self

    def predict(self, X):
        return np.expm1(self.reg.predict(X)) - self._offset


# --- D2 : bagging (moyenne de N XGBoost sur ré-échantillonnages bootstrap) ---
class BaggingXGB:
    def __init__(self, n_bags=5, frac=0.8, **xgb_kwargs):
        self.n_bags = n_bags
        self.frac = frac
        self.xgb_kwargs = xgb_kwargs

    def fit(self, X, y):
        rng = np.random.RandomState(RANDOM_STATE)
        n = len(y)
        self.models = []
        for b in range(self.n_bags):
            idx = rng.choice(n, size=max(20, int(n * self.frac)), replace=True)
            m = XGBRegressor(random_state=RANDOM_STATE + b, n_jobs=1, **self.xgb_kwargs)
            m.fit(X[idx], y[idx])
            self.models.append(m)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


# --- D4 : boosting sur les residus d'interpolation (au lieu de la valeur brute) ---
def make_residual_boost_model(k=K, n_clusters=N_CLUSTERS, xgb_kwargs=None):
    xgb_kwargs = xgb_kwargs or X1_PARAMS

    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        pool = pd.concat([donor_df, dev_true_df], axis=1)
        pool_vals = pool.ffill().bfill().values.astype(np.float64)
        km, donor_labels = feb.fit_shape_clusters(donor_df, n_clusters=n_clusters, random_state=RANDOM_STATE)
        if dev_true_df.shape[1] > 0:
            dev_filled = dev_true_df.ffill().bfill()
            Zdev = (dev_filled - dev_filled.mean()) / (dev_filled.std() + 1e-9)
            dev_labels = pd.Series(km.predict(Zdev.T.values), index=dev_true_df.columns)
            pool_labels = pd.concat([donor_labels, dev_labels])
        else:
            pool_labels = donor_labels
        pos = pd.Series(np.arange(len(pool_labels)), index=pool_labels.index)
        cluster_col_idx = {cl: pos[idx].values for cl, idx in pool_labels.groupby(pool_labels).groups.items()}
        tfeat = feb.time_features(val_holed_df.index)

        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            observed = s.notna().values
            missing = ~observed
            if not missing.any():
                preds[col] = s
                continue
            obs_idx, miss_idx = np.where(observed)[0], np.where(missing)[0]
            target_obs = s.values[obs_idx]
            interp_full = fill_nan_with_interpolation(s)
            cl = feb.cluster_of(interp_full, km)
            same_idx = cluster_col_idx.get(cl)
            sub_vals = pool_vals[:, same_idx] if (same_idx is not None and len(same_idx) >= max(2 * k, 30)) else pool_vals
            knn_pred, top_idx, scaled = feb.weighted_knn_pred(
                sub_vals, obs_idx, miss_idx, target_obs, k, robust=True, adaptive_k=True, use_euclid=True)

            if len(obs_idx) >= 15:
                X_obs = np.hstack([scaled[obs_idx, :], tfeat[obs_idx, :]])
                X_miss = np.hstack([scaled[miss_idx, :], tfeat[miss_idx, :]])
                resid_obs = target_obs - interp_full.values[obs_idx]
                reg = XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, **xgb_kwargs)
                reg.fit(X_obs, resid_obs)
                pred_neighbors = interp_full.values[miss_idx] + reg.predict(X_miss)
            else:
                pred_neighbors = knn_pred

            interp = interp_full.values[miss_idx]
            run_len_full = feb.run_lengths_for(missing)
            edge_dist_full = feb.edge_distance_for(missing)
            alphas = np.array([feb.continuous_alpha(run_len_full[m], edge_dist_full[m], adaptive=True) for m in miss_idx])
            pred = alphas * interp + (1 - alphas) * pred_neighbors
            filled = s.copy()
            filled.iloc[miss_idx] = pred
            preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model


# --- D6 : lissage post-hoc, uniquement au milieu des trous longs (edge_dist
# >= window), pour eviter le jitter irrealiste point-a-point d'un modele
# qui predit chaque point independamment ---
def make_smoothed_model(base_model_fn, window=3):
    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        raw = base_model_fn(donor_df, dev_holed_df, dev_true_df, val_holed_df)
        smoothed = raw.copy()
        for col in val_holed_df.columns:
            missing = val_holed_df[col].isna().values
            if not missing.any():
                continue
            edge_dist = feb.edge_distance_for(missing)
            vals = raw[col].values.copy()
            roll = pd.Series(vals).rolling(window=window, center=True, min_periods=1).mean().values
            deep_mask = missing & (edge_dist >= window)
            vals[deep_mask] = roll[deep_mask]
            smoothed[col] = vals
        return smoothed
    return model


# --- D1 : NMF au lieu de SVD pour SoftImpute (respecte la non-negativite nativement) ---
def make_nmf_softimpute_model(rank=15, n_iter=8, n_clusters=N_CLUSTERS, blend_interp=True):
    from sklearn.decomposition import NMF

    def nmf_impute(X, mask_known, rank, n_iter):
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        X_filled = np.clip(np.where(mask_known, X, np.broadcast_to(col_mean, X.shape)), 0, None)
        prev = X_filled[~mask_known].copy()
        for _ in range(n_iter):
            rk = min(rank, min(X_filled.shape) - 1)
            nmf = NMF(n_components=rk, init="nndsvda", max_iter=100, random_state=RANDOM_STATE)
            W = nmf.fit_transform(X_filled)
            recon = W @ nmf.components_
            X_filled = np.clip(np.where(mask_known, X, recon), 0, None)
            cur = X_filled[~mask_known]
            delta = np.linalg.norm(cur - prev) / (np.linalg.norm(prev) + 1e-9)
            prev = cur
            if delta < 1e-3:
                break
        return X_filled

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

        val_cluster = {c: feb.cluster_of(fill_nan_with_interpolation(val_holed_df[c]), km) for c in val_holed_df.columns}
        preds = {c: val_holed_df[c].copy() for c in val_holed_df.columns}

        for cl in sorted(set(val_cluster.values())):
            val_cols_cl = [c for c, l in val_cluster.items() if l == cl]
            donor_cols_cl = pool_labels[pool_labels == cl].index.tolist()
            if len(donor_cols_cl) < 20:
                donor_cols_cl = list(pool_labels.index)
            block = pd.concat([pool_complete[donor_cols_cl], val_holed_df[val_cols_cl]], axis=1)
            X = block.values.astype(np.float64)
            mask_known = ~np.isnan(X)
            X_completed = nmf_impute(X, mask_known, rank, n_iter)

            for j, col in enumerate(val_cols_cl):
                j_full = len(donor_cols_cl) + j
                s = val_holed_df[col]
                missing = s.isna().values
                if not missing.any():
                    continue
                pred_nmf = X_completed[:, j_full]
                if blend_interp:
                    interp_full = fill_nan_with_interpolation(s)
                    run_len_full = feb.run_lengths_for(missing)
                    edge_dist_full = feb.edge_distance_for(missing)
                    miss_idx = np.where(missing)[0]
                    alphas = np.array([feb.continuous_alpha(run_len_full[m], edge_dist_full[m], adaptive=True) for m in miss_idx])
                    pred = alphas * interp_full.values[miss_idx] + (1 - alphas) * pred_nmf[miss_idx]
                else:
                    pred = pred_nmf[missing]
                filled = s.copy()
                filled.iloc[np.where(missing)[0]] = pred
                preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model


# --- D3 : stacking global (interpolation + XGBoost-T1 + SoftImpute), poids
# appris par Ridge sur un sous-echantillon de dev_cols. Simplifie : les
# modeles de base utilisent UNIQUEMENT donor_cols (pas dev_true_df en donneur
# supplementaire) pour eviter toute fuite lors du calcul des features de
# stacking sur dev_cols. ---
def make_stacking_model(dev_subsample=100, random_state=RANDOM_STATE):
    rng = np.random.RandomState(random_state)
    xgb_model_fn = feb.make_v9_model(
        k=K, n_clusters=N_CLUSTERS, adaptive_k=True, use_euclid=True,
        make_regressor=lambda: XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, **X1_PARAMS),
        random_state=RANDOM_STATE)
    soft_model_fn = sib.make_softimpute_model(rank=15, n_iter=25, blend_interp=False)

    def base_preds(donor_df, cols_df):
        empty_dev = pd.DataFrame(index=donor_df.index)
        interp = cols_df.apply(fill_nan_with_interpolation, axis=0)
        xgb_pred = xgb_model_fn(donor_df, empty_dev, empty_dev, cols_df)
        soft_pred = soft_model_fn(donor_df, empty_dev, empty_dev, cols_df)
        return interp, xgb_pred, soft_pred

    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        dev_cols = list(dev_holed_df.columns)
        sub_cols = rng.choice(dev_cols, size=min(dev_subsample, len(dev_cols)), replace=False)
        dev_sub_df = dev_holed_df[sub_cols]
        interp_d, xgb_d, soft_d = base_preds(donor_df, dev_sub_df)

        rows_X, rows_y = [], []
        for col in sub_cols:
            missing = dev_holed_df[col].isna().values
            if not missing.any():
                continue
            y_true = dev_true_df[col].values[missing]
            valid = ~np.isnan(y_true)  # y_train est lui-meme NaN sur qqs positions (cf. evaluate.official_mae)
            if not valid.any():
                continue
            rows_X.append(np.column_stack([
                interp_d[col].values[missing][valid], xgb_d[col].values[missing][valid], soft_d[col].values[missing][valid]]))
            rows_y.append(y_true[valid])
        X_meta = np.vstack(rows_X)
        y_meta = np.concatenate(rows_y)
        meta = Ridge(alpha=1.0, positive=True)
        meta.fit(X_meta, y_meta)

        interp_v, xgb_v, soft_v = base_preds(donor_df, val_holed_df)
        preds = {}
        for col in val_holed_df.columns:
            s = val_holed_df[col]
            missing = s.isna().values
            if not missing.any():
                preds[col] = s
                continue
            X_stack = np.column_stack([
                interp_v[col].values[missing], xgb_v[col].values[missing], soft_v[col].values[missing]])
            pred = meta.predict(X_stack)
            filled = s.copy()
            filled.iloc[np.where(missing)[0]] = pred
            preds[col] = filled
        return pd.DataFrame(preds, index=val_holed_df.index)
    return model


def run(name, model):
    print(f"=== {name} ===")
    t0 = time.time()
    scores = feb.evaluate_repeated_splits(
        X_train, y_train, donor_cols, model,
        n_repeats=N_REPEATS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"  total: {time.time()-t0:.1f}s\n")
    return scores


def build_model(name):
    if name == "D8_log1p":
        return feb.make_v9_model(k=K, n_clusters=N_CLUSTERS, adaptive_k=True, use_euclid=True,
                                  make_regressor=lambda: Log1pXGB(**X1_PARAMS), random_state=RANDOM_STATE)
    if name == "D2_bagging":
        return feb.make_v9_model(k=K, n_clusters=N_CLUSTERS, adaptive_k=True, use_euclid=True,
                                  make_regressor=lambda: BaggingXGB(n_bags=5, frac=0.8, **X1_PARAMS),
                                  random_state=RANDOM_STATE)
    if name == "D4_residual_boost":
        return make_residual_boost_model(xgb_kwargs=X1_PARAMS)
    if name == "D6_smoothing":
        base = feb.make_v9_model(k=K, n_clusters=N_CLUSTERS, adaptive_k=True, use_euclid=True,
                                  make_regressor=lambda: XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, **X1_PARAMS),
                                  random_state=RANDOM_STATE)
        return make_smoothed_model(base, window=3)
    if name == "D1_nmf_softimpute":
        return make_nmf_softimpute_model(rank=15, n_iter=8)
    if name == "D3_stacking":
        return make_stacking_model(dev_subsample=100)
    raise ValueError(name)


RESULTS_PATH = "xgboost_bench_round3_results.json"


def load_results():
    import json, os
    if os.path.exists(RESULTS_PATH):
        return json.load(open(RESULTS_PATH))
    return {}


def save_result(name, scores):
    import json
    results = load_results()
    results[name] = scores.tolist()
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import sys
    ALL = ["D8_log1p", "D2_bagging", "D4_residual_boost", "D6_smoothing", "D1_nmf_softimpute", "D3_stacking"]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    names = [only] if only else ALL

    print("Chargement des données (X_train, y_train)...")
    X_train, y_train, _ = load_data()
    donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]

    for name in names:
        model = build_model(name)
        scores = run(name, model)
        save_result(name, scores)
        print(f"[{name}] sauvegardé dans {RESULTS_PATH}")
