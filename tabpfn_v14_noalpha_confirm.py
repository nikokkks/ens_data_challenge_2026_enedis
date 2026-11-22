"""Confirmation 10x10 de "TabPFN seul, sans blend interpolation" -- copie de
tabpfn_v14_confirm.py (config BEST/v14_best, protocole officiel 10x10
apparié, mêmes graines, même baseline 63.24 +/- 30.46 W) avec pour SEUL
changement `alpha_scale=0` (au lieu de 1.0).

`alpha_scale` est un knob déjà existant dans tabpfn_v14_ablation.DEFAULT_OPTS
-- aucune modification de code nécessaire. Dans predict_column_v14 :

    alpha = np.clip(alpha * o["alpha_scale"], 0.0, 0.95)
    pred = alpha * interp_full[miss_idx] + (1 - alpha) * pred_neighbors

`alpha_scale=0` force alpha=0 sur toute la courbe -> pred = pred_neighbors
(TabPFN pur, ou repli KNN pondéré si TabPFN échoue), l'interpolation
linéaire n'intervient plus du tout dans la prédiction finale. Vérifié
bit-à-bit en local (régresseur LightGBM en proxy, sans API) avant de
lancer ce script : alpha_scale=0 reproduit exactement la sortie brute du
régresseur, np.allclose == True.

Contexte déjà connu (mais PAS sur le protocole 10x10 exact -- deux tests
antérieurs sur colonnes fraîches, cf. benchmark.md §14 et §15ter) :

    | Test              | interpolation | TabPFN seul | blend V14 |
    |-------------------|--------------:|------------:|----------:|
    | §14 (30 colonnes) | 111,46 W      | 75,93 W     | 75,57 W   |
    | §15ter (100 col.) | 92,05 W       | 64,91 W     | 64,55 W   |

Dans les deux cas TabPFN seul est ~0.36 W derrière le blend -- ce script
vérifie si ce petit écart tient sur le protocole 10x10 apparié officiel du
dépôt (mêmes graines que tabpfn_v14_confirm_results.json).

Résultats directement comparables PAR PAIRE (même protocole, mêmes graines,
même config BEST sauf alpha_scale) à tabpfn_v14_confirm_results.json.
"""
import json
import os

import numpy as np

import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_noalpha_confirm_results.json"
BASELINE_PATH = "tabpfn_v14_confirm_results.json"

# Config BEST identique a tabpfn_v14_confirm.py, + alpha_scale=0 (pas de blend).
BEST = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
            extra_lags=(24,), neighbor_rolling=True)

if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant")
    tabpfn_client.set_access_token(token)
    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]

    opts = {**BEST, "alpha_scale": 0}
    print(f"=== CONFIRM v14_noalpha (TabPFN pur, sans blend): {opts} ===", flush=True)
    scores = v14.run_config("v14_noalpha", opts, X, y, donors, n_repeats=10, n_val_sub=10)

    result = {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}
    json.dump(result, open(RESULTS_PATH, "w"), indent=2)
    print("Terminé.", flush=True)

    # Comparaison appariée immédiate avec le baseline confirm (mêmes graines,
    # même protocole 10x10) -- cf. CLAUDE.md: comparer repeat par repeat, pas
    # juste sur la moyenne.
    if os.path.exists(BASELINE_PATH):
        baseline = json.load(open(BASELINE_PATH))["scores"]
        print("\nrepeat | baseline (v14_best, blend) | TabPFN seul (no blend) |   delta")
        wins = losses = ties = 0
        for i, (b, p) in enumerate(zip(baseline, scores)):
            delta = p - b
            tag = "gagné" if delta < -1e-9 else ("perdu" if delta > 1e-9 else "égalité")
            wins += tag == "gagné"; losses += tag == "perdu"; ties += tag == "égalité"
            print(f"{i + 1:6d} | {b:26.2f} | {p:22.2f} | {delta:+7.2f}  ({tag})")
        print(f"\n{wins} gagnés / {losses} perdus / {ties} égalités sur {len(baseline)} repeats")
        print(f"Moyenne baseline (blend)    : {np.mean(baseline):.2f} W")
        print(f"Moyenne TabPFN seul         : {scores.mean():.2f} W")
