"""Confirmation 10x10 du clip de sécurité "valeurs non négatives" -- copie de
tabpfn_v14_confirm.py (config BEST/v14_best, protocole officiel 10x10
apparié, mêmes graines, même baseline 63.24 +/- 30.46 W) avec pour SEUL
changement `clip_nonneg=True`.

Motivation
----------
Tous les scripts generate_submission_tabpfn_*.py (v12, v13, v14,
zero_start, concat72) font `final = done[holes].clip(lower=0)` avant
d'écrire la soumission -- mais le harnais d'évaluation (tabpfn_v14_ablation
.predict_column_v14, utilisé par ce script et par tabpfn_v14_confirm.py)
n'a jamais appliqué ce clip : la MAE documentée (63.24 +/- 30.46 W) est
calculée sur des prédictions parfois négatives.

Vérifié empiriquement sur 150 vraies colonnes holed_* de X_train (pipeline
V14 réel, régresseur LightGBM en local comme proxy gratuit de TabPFN,
mêmes features/blend) : 5/150 colonnes (3.3%) avaient au moins une
prédiction négative, minimum global -9.02 W -- ni un cas isolé, ni une
explosion, mais un biais systématique réel et non documenté du harnais
d'évaluation.

Comme la vraie consommation est toujours >= 0 (confirmé dans l'EDA du
notebook externe, min observé = 0W), clipper une prédiction négative à 0 ne
peut mathématiquement que RÉDUIRE ou laisser inchangée l'erreur absolue
(|clip(pred,0) - y| <= |pred - y| dès que pred < 0 <= y), jamais la
dégrader. C'est donc un correctif sans risque, purement une question de
cohérence entre harnais d'évaluation et pipeline de soumission réel.

Résultats directement comparables PAR PAIRE (même protocole, mêmes graines,
même config BEST sauf clip_nonneg) à tabpfn_v14_confirm_results.json.
"""
import json
import os

import numpy as np

import tabpfn_client
from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_clip_confirm_results.json"
BASELINE_PATH = "tabpfn_v14_confirm_results.json"

# Config BEST identique a tabpfn_v14_confirm.py, + clip_nonneg=True ajouté.
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

    opts = {**BEST, "clip_nonneg": True}
    print(f"=== CONFIRM v14_clip: {opts} ===", flush=True)
    scores = v14.run_config("v14_clip", opts, X, y, donors, n_repeats=10, n_val_sub=10)

    result = {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}
    json.dump(result, open(RESULTS_PATH, "w"), indent=2)
    print("Terminé.", flush=True)

    # Comparaison appariée immédiate avec le baseline confirm (mêmes graines,
    # même protocole 10x10) -- cf. CLAUDE.md: comparer repeat par repeat, pas
    # juste sur la moyenne.
    if os.path.exists(BASELINE_PATH):
        baseline = json.load(open(BASELINE_PATH))["scores"]
        print("\nrepeat | baseline (v14_best) |  v14 + clip>=0  |   delta")
        wins = losses = ties = 0
        for i, (b, p) in enumerate(zip(baseline, scores)):
            delta = p - b
            tag = "gagné" if delta < -1e-9 else ("perdu" if delta > 1e-9 else "égalité")
            wins += tag == "gagné"; losses += tag == "perdu"; ties += tag == "égalité"
            print(f"{i + 1:6d} | {b:20.2f} | {p:16.2f} | {delta:+7.2f}  ({tag})")
        print(f"\n{wins} gagnés / {losses} perdus / {ties} égalités sur {len(baseline)} repeats")
        print(f"Moyenne baseline (v14_best) : {np.mean(baseline):.2f} W")
        print(f"Moyenne v14 + clip>=0       : {scores.mean():.2f} W")
