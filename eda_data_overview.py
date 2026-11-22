"""
EDA d'introduction : à quoi ressemblent les données du challenge Enedis.

Complète eda_plots.py (qui se concentre sur la difficulté d'imputation) avec
une vue plus "description du jeu de données" :

5. Une courbe holed_* : points connus vs trou réel vs vérité terrain (y_train)
   -> rend concret ce qu'est un "trou" dans ce challenge.
6. Distribution des valeurs de consommation (W) sur les courbes donneuses
   -> unité, échelle, asymétrie typique d'une courbe de charge résidentielle.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluate import load_data

# --- palette (cf. dataviz skill / references/palette.md), identique à eda_plots.py ---
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # série connue
ORANGE = "#eb6834"   # vérité terrain dans le trou
AQUA = "#1baf7a"     # accent secondaire

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
# 5. Une courbe holed_* : connu / trou / vérité terrain
# ============================================================
col = "holed_1"
s = X_train[col]
truth = y_train[col]
t = pd.to_datetime(X_train.index)
missing = s.isna().values

fig, ax = plt.subplots(figsize=(10, 4.5))
# vérité terrain dans le trou, en dessous, tracée en continu pour montrer le "vrai" signal
ax.plot(t, truth.values, color=ORANGE, linewidth=1.6, label="Vérité terrain (y_train, visible seulement dans le trou)")
# points connus par-dessus
known = s.values.copy()
ax.plot(t, known, color=BLUE, linewidth=1.8, label="Valeurs connues (X_train)")
# ombrer les zones de trou
in_gap = False
for i in range(len(missing)):
    if missing[i] and not in_gap:
        start = t[i]
        in_gap = True
    if in_gap and (i == len(missing) - 1 or not missing[i + 1]):
        end = t[i]
        ax.axvspan(start, end, color=INK_MUTED, alpha=0.15, linewidth=0)
        in_gap = False

ax.set_title(f"Anatomie d'une courbe trouée ({col}) : {missing.sum()} points manquants sur {len(missing)}")
ax.set_ylabel("Consommation (W)")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(axis="y", linewidth=0.6)
ax.set_axisbelow(True)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig("figures/05_example_holed_curve.png", dpi=150)
plt.close(fig)
print("figures/05_example_holed_curve.png")


# ============================================================
# 6. Distribution des valeurs de consommation (W)
# ============================================================
rng = np.random.default_rng(42)
sample_cols = rng.choice(donor_cols, size=2000, replace=False)
vals = X_train[sample_cols].values.ravel()
vals = vals[~np.isnan(vals)]

fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.linspace(0, np.percentile(vals, 99.5), 80)
ax.hist(vals, bins=bins, color=BLUE, alpha=0.85)
ax.axvline(np.median(vals), color=ORANGE, linewidth=2, linestyle="--",
           label=f"Médiane = {np.median(vals):.0f} W")
ax.axvline(np.mean(vals), color=INK_PRIMARY, linewidth=1.5, linestyle=":",
           label=f"Moyenne = {np.mean(vals):.0f} W")
ax.set_title("Distribution des puissances (2000 courbes donneuses, points à 30 min)")
ax.set_xlabel("Puissance (W)")
ax.set_ylabel("Nombre de points (sur échantillon)")
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/06_value_distribution.png", dpi=150)
plt.close(fig)
print("figures/06_value_distribution.png")

print("\nTerminé.")
