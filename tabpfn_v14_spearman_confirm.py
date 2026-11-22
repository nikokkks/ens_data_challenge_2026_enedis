"""Confirmation 10x10 du second bloc de voisins "Spearman + PowerTransform"
-- copie de tabpfn_v14_confirm.py (config BEST/v14_best, protocole officiel
10x10 apparié, mêmes graines, même baseline 63.24 +/- 30.46 W) avec pour
SEUL changement l'ajout de `n_spearman=20`.

Nouveau bloc de features (cf. tabpfn_v14_ablation.spearman_powertransform_
neighbor_features), COMPLEMENTAIRE au bloc de voisins existant (Pearson +
euclidien, `scaled`, inchangé) :
  1. Sélection d'un second ensemble de voisins par corrélation de SPEARMAN
     (sur les rangs, robuste aux pics/valeurs aberrantes de consommation qui
     peuvent fausser Pearson -- jamais testé dans ce dépôt) ;
  2. Les courbes retenues sont PowerTransformées (Yeo-Johnson) avant d'être
     données en feature au régresseur -- PAS pour influencer la sélection
     (PowerTransform est monotone, donc Spearman(PT(x),PT(y)) == Spearman(x,y)
     exactement, la transformation avant sélection serait mathématiquement
     sans effet), mais pour donner au modèle une représentation différente
     (distribution reshapée) des mêmes courbes.

Vérifié en local avant de lancer (régresseur LightGBM en proxy, sans API) :
20 colonnes ajoutées, aucun NaN/inf, le reste du pipeline bit-identique à la
config sans n_spearman, pipeline complet (eval_split) sans erreur.

Résultats directement comparables PAR PAIRE (même protocole, mêmes graines,
même config BEST sauf n_spearman) à tabpfn_v14_confirm_results.json.
"""
import json
import os

import numpy as np

import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_spearman_confirm_results.json"
BASELINE_PATH = "tabpfn_v14_confirm_results.json"
N_SPEARMAN = 20

# Config BEST identique a tabpfn_v14_confirm.py, + n_spearman ajouté.
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

    opts = {**BEST, "n_spearman": N_SPEARMAN}
    print(f"=== CONFIRM v14_spearman (n_spearman={N_SPEARMAN}): {opts} ===", flush=True)
    scores = v14.run_config("v14_spearman", opts, X, y, donors, n_repeats=10, n_val_sub=10)

    result = {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std()),
               "n_spearman": N_SPEARMAN}
    json.dump(result, open(RESULTS_PATH, "w"), indent=2)
    print("Terminé.", flush=True)

    # Comparaison appariée immédiate avec le baseline confirm (mêmes graines,
    # même protocole 10x10) -- cf. CLAUDE.md: comparer repeat par repeat, pas
    # juste sur la moyenne.
    if os.path.exists(BASELINE_PATH):
        baseline = json.load(open(BASELINE_PATH))["scores"]
        print("\nrepeat | baseline (v14_best) | v14 + spearman+PT |   delta")
        wins = losses = ties = 0
        for i, (b, p) in enumerate(zip(baseline, scores)):
            delta = p - b
            tag = "gagné" if delta < -1e-9 else ("perdu" if delta > 1e-9 else "égalité")
            wins += tag == "gagné"; losses += tag == "perdu"; ties += tag == "égalité"
            print(f"{i + 1:6d} | {b:20.2f} | {p:18.2f} | {delta:+7.2f}  ({tag})")
        print(f"\n{wins} gagnés / {losses} perdus / {ties} égalités sur {len(baseline)} repeats")
        print(f"Moyenne baseline (v14_best)   : {np.mean(baseline):.2f} W")
        print(f"Moyenne v14 + spearman+PT     : {scores.mean():.2f} W")
