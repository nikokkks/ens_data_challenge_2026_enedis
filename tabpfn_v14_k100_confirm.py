"""Confirmation 10x10 de k=100 (au lieu de k=40) -- copie de
tabpfn_v14_confirm.py (config BEST/v14_best, protocole officiel 10x10
apparié, mêmes graines, même baseline 63.24 +/- 30.46 W) avec pour SEUL
changement `k=100` (nombre max de voisins sélectionnés par
adaptive_top_k_corr_euclid_tunable, au lieu de K_DEFAULT=40).

⚠️ Déjà testé, deux fois, sur l'architecture V12 (pas V14/v14_best) -- et
rejeté les deux fois :

    | Test                              | Résultat            | Verdict  |
    |-----------------------------------|---------------------|----------|
    | k=100 sur V12, écran 3 repeats    | 79,68 vs 79,24 W    | rejeté   |
    | k=300 sur V12, confirmation 10x10 | 67,00 vs 66,24 W    | rejeté (6/10 défaites) |

(cf. benchmark.md §7 et §12 : "60 voisins bruyants" / "le signal utile est
déjà capté par les ~40 meilleurs voisins"). Ni l'un ni l'autre n'a été
testé sur la config BEST actuelle (v14_best) précisément -- ce script comble
ce trou plutôt que de supposer que le rejet sur V12 tient identiquement sur
V14, mais le signal historique est cohérent sur 2 valeurs de k différentes :
attends-toi probablement à une dégradation, pas une amélioration.

Vérifié en local avant de lancer (sans API) : k=100 sélectionne bien 100
voisins (vs 40 par défaut), le reste des features (35 colonnes `extra`)
est inchangé -- k n'affecte que le bloc de voisins `scaled`, rien d'autre.

Résultats directement comparables PAR PAIRE (même protocole, mêmes graines,
même config BEST sauf k) à tabpfn_v14_confirm_results.json.
"""
import json
import os

import numpy as np

import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_k100_confirm_results.json"
BASELINE_PATH = "tabpfn_v14_confirm_results.json"
K = 100

# Config BEST identique a tabpfn_v14_confirm.py, + k=100 au lieu de 40.
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

    opts = {**BEST, "k": K}
    print(f"=== CONFIRM v14_k100 (k={K} au lieu de 40): {opts} ===", flush=True)
    scores = v14.run_config("v14_k100", opts, X, y, donors, n_repeats=10, n_val_sub=10)

    result = {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std()), "k": K}
    json.dump(result, open(RESULTS_PATH, "w"), indent=2)
    print("Terminé.", flush=True)

    # Comparaison appariée immédiate avec le baseline confirm (mêmes graines,
    # même protocole 10x10) -- cf. CLAUDE.md: comparer repeat par repeat, pas
    # juste sur la moyenne.
    if os.path.exists(BASELINE_PATH):
        baseline = json.load(open(BASELINE_PATH))["scores"]
        print("\nrepeat | baseline (k=40, v14_best) |   k=100   |   delta")
        wins = losses = ties = 0
        for i, (b, p) in enumerate(zip(baseline, scores)):
            delta = p - b
            tag = "gagné" if delta < -1e-9 else ("perdu" if delta > 1e-9 else "égalité")
            wins += tag == "gagné"; losses += tag == "perdu"; ties += tag == "égalité"
            print(f"{i + 1:6d} | {b:26.2f} | {p:9.2f} | {delta:+7.2f}  ({tag})")
        print(f"\n{wins} gagnés / {losses} perdus / {ties} égalités sur {len(baseline)} repeats")
        print(f"Moyenne baseline (k=40)  : {np.mean(baseline):.2f} W")
        print(f"Moyenne k=100            : {scores.mean():.2f} W")
