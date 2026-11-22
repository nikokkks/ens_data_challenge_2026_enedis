import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from evaluate import repeated_holdout_eval

DATA_DIR = os.path.join(REPO_ROOT, "data")
HERE = os.path.dirname(os.path.abspath(__file__))

t0 = time.time()
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_78VdSWL.csv"), index_col=0)
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train_u0UkKEh.csv"), index_col=0)
donor_cols = [c for c in X_train.columns if not c.startswith("holed_")]
holed_cols = list(y_train.columns)
print(f"loaded in {time.time()-t0:.1f}s, donors={len(donor_cols)} holed={len(holed_cols)}", flush=True)

rng = np.random.RandomState(42)
pca_cols = rng.choice(donor_cols, size=3000, replace=False)
M = pd.concat([X_train[pca_cols].ffill().bfill(), y_train.ffill().bfill()], axis=1).to_numpy(dtype=np.float64)
Mc = M - M.mean(axis=0, keepdims=True)
total_var = float((Mc ** 2).sum())
t0 = time.time()
U, s, Vt = randomized_svd(Mc, n_components=250, random_state=42)
var_ratio = (s ** 2) / total_var
cum = np.cumsum(var_ratio)
k90 = int(np.searchsorted(cum, 0.90) + 1)
k95 = int(np.searchsorted(cum, 0.95) + 1)
print(f"svd({M.shape}) in {time.time()-t0:.1f}s", flush=True)
print("cumvar@[1,5,10,20,50,100,200]:", [round(float(cum[k-1]), 3) for k in (1,5,10,20,50,100,200) if k <= len(cum)])
print("k90:", k90, "k95:", k95, flush=True)

t0 = time.time()
nn_cols = rng.choice(holed_cols, size=200, replace=False)
donor_vals = X_train[donor_cols].ffill().bfill().to_numpy(dtype=np.float64)
best_corrs = []
for c in nn_cols:
    sc = X_train[c]
    obs = sc.notna().to_numpy()
    if obs.sum() < 10:
        continue
    tgt = sc.to_numpy()[obs]
    d = donor_vals[obs, :]
    tc = tgt - tgt.mean()
    dc = d - d.mean(axis=0, keepdims=True)
    num = dc.T @ tc
    den = np.sqrt((dc ** 2).sum(axis=0) * (tc ** 2).sum()) + 1e-9
    corr = num / den
    best_corrs.append(float(np.max(np.abs(corr))))
best_corrs = np.array(best_corrs)
print(f"nn test in {time.time()-t0:.1f}s", flush=True)
print("nn best|corr| mean:", best_corrs.mean(), "median:", np.median(best_corrs),
      "frac>0.8:", (best_corrs > 0.8).mean(), "frac>0.9:", (best_corrs > 0.9).mean(), flush=True)


def make_svd_impute_model(rank, donor_sample=600, iters=15, seed=42):
    rng2 = np.random.RandomState(seed)

    def model(donor_df, dev_holed_df, dev_true_df, val_holed_df):
        cols = rng2.choice(donor_df.columns, size=min(donor_sample, donor_df.shape[1]), replace=False)
        Mdf = pd.concat([donor_df[cols], dev_true_df, val_holed_df], axis=1)
        Mv = Mdf.to_numpy(dtype=np.float64)
        mask = ~np.isnan(Mv)
        col_mean = np.nanmean(Mv, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        X = np.where(mask, Mv, col_mean[None, :])
        for _ in range(iters):
            Uc, sc2, Vtc = randomized_svd(X, n_components=rank, random_state=0)
            Xr = (Uc * sc2) @ Vtc
            X = np.where(mask, Mv, Xr)
        val_start = len(cols) + dev_true_df.shape[1]
        val_arr = np.clip(X[:, val_start:], 0.0, None)
        return pd.DataFrame(val_arr, index=val_holed_df.index, columns=val_holed_df.columns)

    return model


rank = int(min(60, max(5, k90)))
print("chosen rank:", rank, flush=True)
t0 = time.time()
model = make_svd_impute_model(rank=rank)
scores = repeated_holdout_eval(X_train, y_train, donor_cols, model, n_repeats=5, test_size=0.10, random_state=42)
print(f"svd matrix completion 5x100 done in {time.time()-t0:.1f}s", flush=True)
print("MAE mean:", float(scores.mean()), "std:", float(scores.std()), "per_repeat:", scores.tolist())

result = {
    "pca_k90": k90,
    "pca_k95": k95,
    "pca_cumvar": {str(k): round(float(cum[k-1]), 4) for k in (1,5,10,20,50,100,200) if k <= len(cum)},
    "nn_best_corr_mean": float(best_corrs.mean()),
    "nn_best_corr_median": float(np.median(best_corrs)),
    "nn_frac_above_0.8": float((best_corrs > 0.8).mean()),
    "nn_frac_above_0.9": float((best_corrs > 0.9).mean()),
    "svd_matrix_completion_rank": rank,
    "svd_matrix_completion_5x100": {
        "mean": float(scores.mean()), "std": float(scores.std()), "per_repeat": scores.tolist()
    },
    "reference_lightgbm_5x100_k100": {"mean": 74.50, "std": 7.11},
    "reference_tabpfn_v14_10x10": {"mean": 63.24, "std": 30.46, "note": "protocole different, non comparable directement"},
}
out_path = os.path.join(HERE, "svd_matrix_completion_results.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print("written:", out_path, flush=True)
