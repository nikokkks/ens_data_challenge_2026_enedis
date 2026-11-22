"""
EDA rapide : les features de X_train les plus liées à la difficulté
d'imputation de y_train, en 4 graphiques (cf. figures/).

1. Distribution des longueurs de trous (bimodale : isolés vs pleine-journée)
2. Corrélation inter-courbes : intra-cluster de forme vs inter-cluster
3. Profil journalier moyen (heure de la journée), semaine vs weekend
4. Erreur d'imputation (baseline vs meilleur modèle V5) par longueur de trou
   -> le lien direct X_train (pattern des trous) -> y_train (qualité de reconstruction)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from evaluate import load_data, fill_nan_with_interpolation
import feature_engineering_bench as feb

# --- palette (cf. dataviz skill / references/palette.md) ---
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # series-1
ORANGE = "#eb6834"   # series-2
AQUA = "#1baf7a"     # series-3

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE_AXIS, "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    "grid.color": GRID, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif",
})

print("Chargement des données...")
X_train, y_train, X_test = load_data()
donor_cols = [c for c in X_train.columns if not str(c).startswith("holed_")]
holed_cols = [c for c in X_train.columns if str(c).startswith("holed_")]


# ============================================================
# 1. Distribution des longueurs de trous
# ============================================================
def run_lengths_all(cols):
    lengths = []
    for c in cols:
        isna = X_train[c].isna().values
        n = len(isna)
        i = 0
        while i < n:
            if isna[i]:
                j = i
                while j < n and isna[j]:
                    j += 1
                lengths.append(j - i)
                i = j
            else:
                i += 1
    return np.array(lengths)

gap_lengths = run_lengths_all(holed_cols)
total_pts = gap_lengths.sum()
buckets = [("1", 1, 1), ("2-3", 2, 3), ("4-20", 4, 20), ("21-47", 21, 47),
           ("48 (1 jour)", 48, 48), ("49-100", 49, 100), (">100", 101, 100000)]
bucket_pct = []
for label, lo, hi in buckets:
    m = (gap_lengths >= lo) & (gap_lengths <= hi)
    bucket_pct.append(100 * gap_lengths[m].sum() / total_pts)

fig, ax = plt.subplots(figsize=(8, 4.5))
bar_colors = [BLUE] * len(buckets)
bars = ax.bar([b[0] for b in buckets], bucket_pct, color=bar_colors, width=0.62)
for bar, pct in zip(bars, bucket_pct):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{pct:.0f}%",
            ha="center", va="bottom", fontsize=10, color=INK_PRIMARY, fontweight="bold")
ax.set_title("Distribution bimodale des trous : isolés vs pleine-journée")
ax.set_ylabel("% des points manquants")
ax.set_xlabel("Longueur du trou (nb de points, pas 30 min)")
ax.grid(axis="y", linewidth=0.6)
ax.set_axisbelow(True)
ax.set_ylim(0, max(bucket_pct) * 1.2)
fig.tight_layout()
fig.savefig("figures/01_gap_length_distribution.png", dpi=150)
plt.close(fig)
print("figures/01_gap_length_distribution.png")


# ============================================================
# 2. Corrélation intra-cluster vs inter-cluster
# ============================================================
sample_cols = donor_cols[:600]
km, labels = feb.fit_shape_clusters(X_train[sample_cols], n_clusters=8, random_state=42)
corr = X_train[sample_cols].ffill().bfill().corr().values.copy()
np.fill_diagonal(corr, np.nan)
lab = labels.values
n = len(sample_cols)
within, across = [], []
for i in range(n):
    same = lab == lab[i]
    row = corr[i]
    within.extend(row[same][~np.isnan(row[same])].tolist())
    across.extend(row[~same][~np.isnan(row[~same])].tolist())
within, across = np.array(within), np.array(across)

fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(-0.3, 1.0, 60)
ax.hist(across, bins=bins, color=ORANGE, alpha=0.55, density=True, label=f"Inter-cluster (moy={across.mean():.3f})")
ax.hist(within, bins=bins, color=BLUE, alpha=0.55, density=True, label=f"Intra-cluster (moy={within.mean():.3f})")
ax.axvline(across.mean(), color=ORANGE, linewidth=2, linestyle="--")
ax.axvline(within.mean(), color=BLUE, linewidth=2, linestyle="--")
ax.set_title("Corrélation entre courbes : intra-cluster ×8 plus forte")
ax.set_xlabel("Corrélation de Pearson entre deux courbes donneuses")
ax.set_ylabel("Densité")
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/02_correlation_intra_inter_cluster.png", dpi=150)
plt.close(fig)
print("figures/02_correlation_intra_inter_cluster.png")


# ============================================================
# 3. Profil journalier moyen : semaine vs weekend
# ============================================================
sample = X_train[donor_cols[:3000]]
dt_index = pd.to_datetime(X_train.index)
hour = dt_index.hour + dt_index.minute / 60
is_weekend = dt_index.dayofweek >= 5

df = sample.copy()
df["hour"] = hour
df["weekend"] = is_weekend
prof_week = df[~df["weekend"]].groupby("hour")[donor_cols[:3000]].mean().mean(axis=1)
prof_weekend = df[df["weekend"]].groupby("hour")[donor_cols[:3000]].mean().mean(axis=1)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(prof_week.index, prof_week.values, color=BLUE, linewidth=2, label="Semaine")
ax.plot(prof_weekend.index, prof_weekend.values, color=ORANGE, linewidth=2, label="Weekend")
ax.set_title("Profil journalier moyen : semaine vs weekend")
ax.set_xlabel("Heure de la journée")
ax.set_ylabel("Consommation moyenne (W)")
ax.set_xlim(0, 24)
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/03_daily_profile_weekday_weekend.png", dpi=150)
plt.close(fig)
print("figures/03_daily_profile_weekday_weekend.png")


# ============================================================
# 4. Erreur d'imputation par longueur de trou : baseline vs V5
# ============================================================
from sklearn.model_selection import train_test_split

dev_cols, val_cols = train_test_split(holed_cols, test_size=0.10, random_state=42)
donor_df = X_train[donor_cols]
dev_true_df = y_train[dev_cols]
val_holed_df = X_train[val_cols]

baseline_pred = val_holed_df.apply(fill_nan_with_interpolation, axis=0)
v5_model = feb.make_stepped_model(k=20, use_cluster=True, adaptive_alpha=True,
                                   use_ridge=True, robust_scale=True, n_clusters=25)
v5_pred = v5_model(donor_df, X_train[dev_cols], dev_true_df, val_holed_df)

rows = []
for col in val_cols:
    s = val_holed_df[col]
    missing = s.isna().values
    if not missing.any():
        continue
    run_len = feb.run_lengths_for(missing)
    y_true = y_train[col].values
    valid = missing & ~np.isnan(y_true)  # exclut les positions où y_train est lui-même NaN
    err_base = np.abs(baseline_pred[col].values[valid] - y_true[valid])
    err_v5 = np.abs(v5_pred[col].values[valid] - y_true[valid])
    gl = run_len[valid]
    for g, eb, e5 in zip(gl, err_base, err_v5):
        rows.append((g, eb, e5))

err_df = pd.DataFrame(rows, columns=["gap_len", "err_baseline", "err_v5"])
edges = [(1, 1, "1"), (2, 3, "2-3"), (4, 20, "4-20"), (21, 47, "21-47"),
         (48, 48, "48"), (49, 100, "49-100"), (101, 100000, ">100")]
labels_x, mae_base, mae_v5, ns = [], [], [], []
for lo, hi, lab in edges:
    sub = err_df[(err_df.gap_len >= lo) & (err_df.gap_len <= hi)]
    if len(sub) == 0:
        continue
    labels_x.append(lab)
    mae_base.append(sub.err_baseline.mean())
    mae_v5.append(sub.err_v5.mean())
    ns.append(len(sub))

x = np.arange(len(labels_x))
width = 0.36
fig, ax = plt.subplots(figsize=(9, 4.8))
b1 = ax.bar(x - width / 2, mae_base, width, color=ORANGE, label="Baseline (interpolation linéaire)")
b2 = ax.bar(x + width / 2, mae_v5, width, color=BLUE, label="V5 (meilleur modèle)")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(mae_base + mae_v5) * 0.015,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels(labels_x)
ax.set_title("Erreur d'imputation par longueur de trou (1 split, val=100 courbes)")
ax.set_xlabel("Longueur du trou (nb de points)")
ax.set_ylabel("MAE (W)")
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/04_error_by_gap_length.png", dpi=150)
plt.close(fig)
print("figures/04_error_by_gap_length.png")

print("\nTerminé.")
