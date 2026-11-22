"""Confirmation 10x10 de l'idée "ST-PCA / EOF via matrice de décalage
temporel" -- copie de tabpfn_v14_confirm.py (config BEST/v14_best,
protocole officiel 10x10 apparié, mêmes graines, même baseline
63.24 +/- 30.46 W) avec pour SEUL changement l'ajout de
`lagged_eof_lags=(0, 48, 96, 144)`.

Contexte (demande explicite : comparaison de 4 transformateurs de features
classiques, sans deep learning) :
  1. ST-PCA/EOF (matrice de décalage temporel + PCA)      -- APPLICABLE
  2. Tucker/HOSVD (tenseur 3D stations x temps x variables) -- inapplicable
     (une seule variable par courbe, pas de tenseur 3D naturel)
  3. Spatial PCA / Moran (coordonnées géographiques)        -- inapplicable
     (identifiants de courbe aléatoires, aucune notion d'espace)
  4. Kernel PCA spatio-temporel (noyau spatial x temporel)  -- inapplicable
     (même absence de coordonnées spatiales que #3)

Seule l'option #1 s'applique à ce jeu de données. Une version SIMPLE existe
déjà dans le pipeline (compute_svd_factors, option n_svd -- SVD unique sur
le pool brut, jamais confirmée en 10x10 malgré son usage en soumission).
Ce script teste la version "matrice de décalage temporel" plus complète :
SVD tronquée SEPAREE à chaque décalage (0/48/96/144h) du pool, facteurs
concaténés -- capture la covariance inter-courbes DECALEE dans le temps
(deux courbes peuvent se ressembler avec un déphasage), pas seulement
instantanée comme n_svd seul (cf. tabpfn_v14_ablation.lagged_eof_features).

Vérifié en local avant de lancer (sans API) : 32 colonnes ajoutées (4 lags
x 8 composantes), aucun NaN/inf, reste du pipeline bit-identique, calcul
~6.5s par split (négligeable face au temps des appels TabPFN), pipeline
complet (eval_split) sans erreur.

Résultats directement comparables PAR PAIRE (même protocole, mêmes graines,
même config BEST sauf lagged_eof_lags) à tabpfn_v14_confirm_results.json.
"""
import json
import os

import numpy as np

import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_eof_confirm_results.json"
BASELINE_PATH = "tabpfn_v14_confirm_results.json"
LAGS = (0, 48, 96, 144)

# Config BEST identique a tabpfn_v14_confirm.py, + lagged_eof_lags ajouté.
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

    opts = {**BEST, "lagged_eof_lags": LAGS}
    print(f"=== CONFIRM v14_eof (lags={LAGS}): {opts} ===", flush=True)
    scores = v14.run_config("v14_eof", opts, X, y, donors, n_repeats=10, n_val_sub=10)

    result = {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std()),
               "lagged_eof_lags": list(LAGS)}
    json.dump(result, open(RESULTS_PATH, "w"), indent=2)
    print("Terminé.", flush=True)

    # Comparaison appariée immédiate avec le baseline confirm (mêmes graines,
    # même protocole 10x10) -- cf. CLAUDE.md: comparer repeat par repeat, pas
    # juste sur la moyenne.
    if os.path.exists(BASELINE_PATH):
        baseline = json.load(open(BASELINE_PATH))["scores"]
        print("\nrepeat | baseline (v14_best) |   v14 + EOF decale  |   delta")
        wins = losses = ties = 0
        for i, (b, p) in enumerate(zip(baseline, scores)):
            delta = p - b
            tag = "gagné" if delta < -1e-9 else ("perdu" if delta > 1e-9 else "égalité")
            wins += tag == "gagné"; losses += tag == "perdu"; ties += tag == "égalité"
            print(f"{i + 1:6d} | {b:20.2f} | {p:20.2f} | {delta:+7.2f}  ({tag})")
        print(f"\n{wins} gagnés / {losses} perdus / {ties} égalités sur {len(baseline)} repeats")
        print(f"Moyenne baseline (v14_best)  : {np.mean(baseline):.2f} W")
        print(f"Moyenne v14 + EOF décalée    : {scores.mean():.2f} W")
