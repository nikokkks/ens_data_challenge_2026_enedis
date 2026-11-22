"""Confirmation 10x10 de l'idée "3 méthodes d'interpolation intra-courbe
comme features" -- copie de tabpfn_v14_confirm.py (config BEST/v14_best,
protocole officiel 10x10 apparié, mêmes graines, même baseline
63.24 +/- 30.46 W) avec pour SEUL changement l'ajout de
`interp_features=("linear", "pchip", "quadratic")`.

Nouveau bloc de features (cf. tabpfn_v14_ablation.interpolation_method_
features) : pour chaque colonne, la courbe complète reconstruite par
interpolation linéaire, PCHIP (monotone, sans oscillation) et quadratique
(courbure locale) -- 3 colonnes de plus données au régresseur, en
COMPLEMENT du blend alpha existant (qui continue d'utiliser uniquement
l'interpolation linéaire, comportement inchangé). Normalisées médiane/MAD
et clippées aux bornes observées (même garde-fou que le reste du pipeline,
motivé par l'échec documenté "PCHIP en remplacement du blend" à 1808 W sur
10x10, cf. benchmark.md §7 -- ici on ajoute juste des features, on ne
remplace rien, mais le clip reste une précaution appliquée par prudence).

Contexte déjà connu, mais PAS sur le pipeline actuel : `pchip_interp_
feature` seule avait été testée sur l'ancien pipeline V10 (régresseur
Ridge, pas TabPFN) -- 108,49 W sur 5x100, rejetée (benchmark.md §7). Jamais
revalidée sur le pipeline V14/TabPFN actuel, ni combinée aux méthodes
linéaire/quadratique. Ce script comble ce trou plutôt que de supposer que
le rejet d'alors tient toujours sur l'architecture actuelle (cf. le
principe déjà établi dans ce dépôt : "une feature gagnante/perdante sur une
architecture ne se transfère pas automatiquement à une autre").

Résultats directement comparables PAR PAIRE (même protocole, mêmes graines,
même config BEST sauf interp_features) à tabpfn_v14_confirm_results.json.
"""
import json
import os

import numpy as np

import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_interp_confirm_results.json"
BASELINE_PATH = "tabpfn_v14_confirm_results.json"
INTERP_METHODS = ("linear", "pchip", "quadratic")

# Config BEST identique a tabpfn_v14_confirm.py, + interp_features ajouté.
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

    opts = {**BEST, "interp_features": INTERP_METHODS}
    print(f"=== CONFIRM v14_interp (methodes={INTERP_METHODS}): {opts} ===", flush=True)
    scores = v14.run_config("v14_interp", opts, X, y, donors, n_repeats=10, n_val_sub=10)

    result = {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std()),
               "interp_features": list(INTERP_METHODS)}
    json.dump(result, open(RESULTS_PATH, "w"), indent=2)
    print("Terminé.", flush=True)

    # Comparaison appariée immédiate avec le baseline confirm (mêmes graines,
    # même protocole 10x10) -- cf. CLAUDE.md: comparer repeat par repeat, pas
    # juste sur la moyenne.
    if os.path.exists(BASELINE_PATH):
        baseline = json.load(open(BASELINE_PATH))["scores"]
        print("\nrepeat | baseline (v14_best) | v14 + interp feats |   delta")
        wins = losses = ties = 0
        for i, (b, p) in enumerate(zip(baseline, scores)):
            delta = p - b
            tag = "gagné" if delta < -1e-9 else ("perdu" if delta > 1e-9 else "égalité")
            wins += tag == "gagné"; losses += tag == "perdu"; ties += tag == "égalité"
            print(f"{i + 1:6d} | {b:20.2f} | {p:18.2f} | {delta:+7.2f}  ({tag})")
        print(f"\n{wins} gagnés / {losses} perdus / {ties} égalités sur {len(baseline)} repeats")
        print(f"Moyenne baseline (v14_best)  : {np.mean(baseline):.2f} W")
        print(f"Moyenne v14 + interp feats   : {scores.mean():.2f} W")
