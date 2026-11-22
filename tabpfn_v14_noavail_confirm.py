"""Confirmation 10x10 du retrait des 10 flags de disponibilité des copies
périodiques -- copie de tabpfn_v14_confirm.py (config BEST/v14_best,
protocole officiel 10x10 apparié, mêmes graines, même baseline
63.24 +/- 30.46 W) avec pour SEUL changement `periodic_availability_feat=False`.

Motivation (screening local gratuit, LightGBM en proxy, 150 vraies colonnes
holed_* de X_train, avant de dépenser du budget TabPFN) :

    feature         pearson   spearman   importance perm. (moyenne)
    avail_back96     0,015     0,015     -0,00109  (négatif)
    avail_fwd48     -0,012    -0,008     -0,000047 (négatif)
    avail_back24     0,007     0,010     -0,000046 (négatif)
    avail_fwd144    -0,022    -0,018     -0,000019 (négatif)
    avail_back144    0,038     0,033     -0,0000084 (négatif)
    avail_fwd96/24/back48        ~0        exactement 0
    avail_fwd336/back336        ~0,05-0,07  ~0,0001-0,0002 (quasi nul)

Comparé aux features qui comptent vraiment : roll_mean (corr 0,87,
importance 9,0), self_seasonality (corr 0,88, importance 5,3),
periodic_median (importance 6,6) -- écart de 3 à 4 ordres de grandeur. Ces
10 flags 0/1 indiquent juste si une copie périodique à un lag donné est
disponible (non-NaN) -- information déjà accessible au modèle via le
pattern de NaN des colonnes brutes elles-mêmes (back48/fwd48/etc.),
probablement redondante plutôt qu'informative.

⚠️ Screening LightGBM, PAS une confirmation TabPFN -- ce dépôt a déjà
documenté qu'une feature inutile sur une architecture ne l'est pas
forcément sur une autre (benchmark.md §17). Ce script tranche la question
sur le vrai pipeline TabPFN, protocole 10x10 identique à la référence.

Vérifié en local avant de lancer : le retrait supprime exactement les 10
colonnes visées (indices 23-32 du bloc `extra`), aucune autre colonne
n'est affectée (bit-identique), pipeline complet (eval_split) sans erreur.

Résultats directement comparables PAR PAIRE (même protocole, mêmes graines,
même config BEST sauf periodic_availability_feat) à
tabpfn_v14_confirm_results.json.
"""
import json
import os

import numpy as np

import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_noavail_confirm_results.json"
BASELINE_PATH = "tabpfn_v14_confirm_results.json"

# Config BEST identique a tabpfn_v14_confirm.py, + periodic_availability_feat=False.
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

    opts = {**BEST, "periodic_availability_feat": False}
    print(f"=== CONFIRM v14_noavail (retrait 10 flags dispo periodiques): {opts} ===", flush=True)
    scores = v14.run_config("v14_noavail", opts, X, y, donors, n_repeats=10, n_val_sub=10)

    result = {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}
    json.dump(result, open(RESULTS_PATH, "w"), indent=2)
    print("Terminé.", flush=True)

    # Comparaison appariée immédiate avec le baseline confirm (mêmes graines,
    # même protocole 10x10) -- cf. CLAUDE.md: comparer repeat par repeat, pas
    # juste sur la moyenne.
    if os.path.exists(BASELINE_PATH):
        baseline = json.load(open(BASELINE_PATH))["scores"]
        print("\nrepeat | baseline (v14_best) | sans 10 flags dispo |   delta")
        wins = losses = ties = 0
        for i, (b, p) in enumerate(zip(baseline, scores)):
            delta = p - b
            tag = "gagné" if delta < -1e-9 else ("perdu" if delta > 1e-9 else "égalité")
            wins += tag == "gagné"; losses += tag == "perdu"; ties += tag == "égalité"
            print(f"{i + 1:6d} | {b:20.2f} | {p:19.2f} | {delta:+7.2f}  ({tag})")
        print(f"\n{wins} gagnés / {losses} perdus / {ties} égalités sur {len(baseline)} repeats")
        print(f"Moyenne baseline (v14_best)     : {np.mean(baseline):.2f} W")
        print(f"Moyenne sans 10 flags dispo      : {scores.mean():.2f} W")
