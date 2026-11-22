"""Confirmation 10x10 de l'idée "facteurs SVD du pool comme features" --
copie de tabpfn_v14_confirm.py (config BEST/v14_best, protocole officiel
10x10 apparié, mêmes graines, même baseline 63.24 +/- 30.46 W) avec pour
SEUL changement l'ajout de `n_svd` facteurs temporels latents du pool
(cf. tabpfn_v14_ablation.compute_svd_factors, nouvellement branché dans le
pipeline via l'option `n_svd`, comportement par défaut inchangé si absent).

Nombre de composantes retenu : 10, PAS le N_SVD=50 du script colab
(claude/colab_generate_submission_tabpfn_v14_full.py) qui n'était validé
que sur un seul repeat (100 colonnes), jamais confirmé au niveau 10x10 de
ce dépôt. Le choix de 10 vient du notebook externe dempxl/
ChallengeData-ENEDIS (section 3.4, sweep rank=[5,10,20,50,100] sur la SVD
utilisée comme méthode d'imputation DIRECTE, pas comme feature) :

    rank= 5  ->  MAE = 111.68 W
    rank=10  ->  MAE = 111.58 W  <- meilleur
    rank=20  ->  MAE = 116.08 W
    rank=50  ->  MAE = 121.14 W
    rank=100 ->  MAE = 127.78 W

Au-delà d'une dizaine de composantes, la SVD sur-apprend/reconstruit du
bruit sur ce jeu de données plutôt que du signal utile -- signal indirect
mais concret que peu de composantes suffisent à capturer l'essentiel de la
structure du pool. Usage ici DIFFERENT du notebook : pas une méthode
d'imputation directe, mais des facteurs concaténés à la matrice de
features passée à TabPFN (comme les autres blocs de `extra` dans
predict_column_v14/build_v14_features).

Résultats directement comparables PAR PAIRE (même protocole, mêmes graines,
même config BEST sauf n_svd) à tabpfn_v14_confirm_results.json.
"""
import json
import os

import numpy as np

import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_svd_confirm_results.json"
BASELINE_PATH = "tabpfn_v14_confirm_results.json"
N_SVD = 10  # rang optimal empirique trouvé dans le notebook externe (cf. docstring)

# Config BEST identique a tabpfn_v14_confirm.py, + n_svd ajouté.
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

    opts = {**BEST, "n_svd": N_SVD}
    print(f"=== CONFIRM v14_svd (n_svd={N_SVD}): {opts} ===", flush=True)
    scores = v14.run_config("v14_svd", opts, X, y, donors, n_repeats=10, n_val_sub=10)

    result = {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std()),
               "n_svd": N_SVD}
    json.dump(result, open(RESULTS_PATH, "w"), indent=2)
    print("Terminé.", flush=True)

    # Comparaison appariée immédiate avec le baseline confirm (mêmes graines,
    # même protocole 10x10) -- cf. CLAUDE.md: comparer repeat par repeat, pas
    # juste sur la moyenne.
    if os.path.exists(BASELINE_PATH):
        baseline = json.load(open(BASELINE_PATH))["scores"]
        print("\nrepeat | baseline (v14_best) |  v14 + svd(10)  |   delta")
        wins = losses = ties = 0
        for i, (b, p) in enumerate(zip(baseline, scores)):
            delta = p - b
            tag = "gagné" if delta < -1e-9 else ("perdu" if delta > 1e-9 else "égalité")
            wins += tag == "gagné"; losses += tag == "perdu"; ties += tag == "égalité"
            print(f"{i + 1:6d} | {b:20.2f} | {p:16.2f} | {delta:+7.2f}  ({tag})")
        print(f"\n{wins} gagnés / {losses} perdus / {ties} égalités sur {len(baseline)} repeats")
        print(f"Moyenne baseline (v14_best) : {np.mean(baseline):.2f} W")
        print(f"Moyenne v14 + svd({N_SVD})     : {scores.mean():.2f} W")
