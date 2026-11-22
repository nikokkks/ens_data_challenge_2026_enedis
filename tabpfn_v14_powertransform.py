"""Test de l'idée "PowerTransformer sur la cible avant fit TabPFN" -- copie
de tabpfn_v14_confirm.py (config BEST/v14_best, protocole officiel 10x10
apparie, memes graines, meme baseline 63.24 +/- 30.46 W) avec pour SEUL
changement le regressor_cls utilise : PowerTransformedMedianTabPFN au lieu
de v14.MedianTabPFN.

Motivation
----------
La consommation est tres asymetrique (mediane 181W << moyenne 361W, cf. EDA
du notebook / benchmark.md). TabPFN discretise deja la cible en interne via
une "bar distribution" dont les bornes s'adaptent aux quantiles du y observe
passe a .fit() -- l'effet d'un PowerTransformer (Yeo-Johnson) EXPLICITE sur
la cible avant le fit est donc incertain (possiblement redondant avec ce que
fait deja TabPFN, possiblement complementaire sur des echantillons de
voisins tres petits ~15-40 points ou l'auto-adaptation interne a peu de
marge) -- d'ou ce test empirique isole plutot qu'une hypothese non testee.

Box-Cox est exclu (exige y > 0 strictement ; la conso vaut parfois 0W,
verifie dans l'EDA). Yeo-Johnson gere le zero.

Point cle : Yeo-Johnson est une transformation MONOTONE, donc la mediane
commute avec elle -- predire la mediane dans l'espace transforme puis
inverse_transform revient exactement a predire la mediane dans l'espace
d'origine. Compatible avec MedianTabPFN (output_type="median") deja utilise
en V14 : pas besoin de toucher au blend alpha en aval (continuous_alpha,
volatility_alpha_mult, etc. restent inchanges).

Resultats directement comparables PAR PAIRE (meme protocole, memes graines,
meme config BEST sauf regressor_cls) a tabpfn_v14_confirm_results.json.
"""
import json
import os

import numpy as np
from sklearn.preprocessing import PowerTransformer

import tabpfn_client
from tabpfn_client import TabPFNRegressor

from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v14_powertransform_results.json"
BASELINE_PATH = "tabpfn_v14_confirm_results.json"

# Config BEST identique a tabpfn_v14_confirm.py (cf. docstring pour le seul
# changement : regressor_cls, ajoute au dict plus bas).
BEST = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
            extra_lags=(24,), neighbor_rolling=True)


class PowerTransformedMedianTabPFN:
    """Meme regressor que v14.MedianTabPFN (TabPFN, output_type="median"),
    mais fit/predict sur la cible transformee par PowerTransformer
    (Yeo-Johnson, standardize=True) plutot que sur la cible brute.
    inverse_transform est applique sur la prediction avant de la renvoyer :
    feb.local_regression_pred (qui attend juste .fit(X, y) / .predict(X))
    ne voit donc aucune difference d'interface avec MedianTabPFN."""

    def __init__(self, random_state=v14.RANDOM_STATE):
        self.reg = TabPFNRegressor(random_state=random_state)
        self.pt = PowerTransformer(method="yeo-johnson", standardize=True)

    def fit(self, X, y):
        y_t = self.pt.fit_transform(np.asarray(y, dtype=np.float64).reshape(-1, 1)).ravel()
        self.reg.fit(X, y_t)
        return self

    def predict(self, X):
        pred_t = self.reg.predict(X, output_type="median")
        return self.pt.inverse_transform(np.asarray(pred_t).reshape(-1, 1)).ravel()


if __name__ == "__main__":
    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant")
    tabpfn_client.set_access_token(token)
    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]

    opts = {**BEST, "regressor_cls": PowerTransformedMedianTabPFN}
    print(f"=== CONFIRM v14_powertransform: {opts} ===", flush=True)
    scores = v14.run_config("v14_powertransform", opts, X, y, donors, n_repeats=10, n_val_sub=10)

    result = {"scores": scores.tolist(), "mean": float(scores.mean()), "std": float(scores.std())}
    json.dump(result, open(RESULTS_PATH, "w"), indent=2)
    print("Terminé.", flush=True)

    # Comparaison appariée immédiate avec le baseline confirm (mêmes graines,
    # même protocole 10x10) -- cf. CLAUDE.md: comparer repeat par repeat, pas
    # juste sur la moyenne.
    if os.path.exists(BASELINE_PATH):
        baseline = json.load(open(BASELINE_PATH))["scores"]
        print("\nrepeat | baseline (v14_best) | power-transform |   delta")
        wins = losses = ties = 0
        for i, (b, p) in enumerate(zip(baseline, scores)):
            delta = p - b
            tag = "gagné" if delta < -1e-9 else ("perdu" if delta > 1e-9 else "égalité")
            wins += tag == "gagné"; losses += tag == "perdu"; ties += tag == "égalité"
            print(f"{i + 1:6d} | {b:20.2f} | {p:16.2f} | {delta:+7.2f}  ({tag})")
        print(f"\n{wins} gagnés / {losses} perdus / {ties} égalités sur {len(baseline)} repeats")
        print(f"Moyenne baseline (v14_best) : {np.mean(baseline):.2f} W")
        print(f"Moyenne power-transform     : {scores.mean():.2f} W")
