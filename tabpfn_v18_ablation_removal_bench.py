"""Round 2026-08-17 : ablation par SUPPRESSION de composants du pipeline V14
(le meilleur validé, 63.24 +/- 30.46 W sur 10x10), plutôt que par ajout.

Motivation : les rounds précédents (benchmark.md §16-17, `liste.md`) ont
testé 22 idées d'AJOUT de features sur V14, 0 gain confirmé -- et plusieurs
échouent justement en DILUANT le signal utile d'une feature par ailleurs
positive (`kitchen_sink`, `best_of_all`). Hypothèse jamais testée : le même
phénomène pourrait s'appliquer à des composants déjà présents dans V14
(ajoutés progressivement depuis V12 : `extra_lags`, `neighbor_rolling`,
renforcement de l'alpha périodique/volatilité) sur les ~900 lignes
d'entraînement d'un fit par-courbe.

Deux phases, toutes deux checkpointées (résultats déjà présents sautés au
relancement) :

  1. Recherche (`--search`, défaut) : chaque variante retire/atténue UN
     composant de V14 (+ quelques combinaisons), évaluée sur repeat 1 seul
     (seed=42, RANDOM_STATE), nval=10 -- identique au protocole de
     `tabpfn_v16_hypotheses_bench.py` (baseline déjà mesurée : 110.803 W,
     bit-à-bit identique à `tabpfn_v14_confirm_results.json`).
  2. Confirmation (`--confirm NAME`) : rejoue le(s) candidat(s) retenu(s) sur
     le protocole robuste 10x10 complet (mêmes seeds que
     `tabpfn_v14_confirm.py`), pour comparaison directe et appariée avec les
     63.24 +/- 30.46 W de référence déjà enregistrés.
"""
import argparse
import json
import os
import time

import numpy as np
import tabpfn_client

from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v18_ablation_removal_results.json"
V14_CONFIRM_PATH = "tabpfn_v14_confirm_results.json"

# Identique à V14_OPTS de generate_submission_tabpfn_v14.py.
V14_BASE = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                extra_lags=(24,), neighbor_rolling=True)

# Chaque entrée : overrides appliqués par-dessus V14_BASE. () = suppression
# totale d'un composant optionnel (retour à sa valeur DEFAULT_OPTS).
ABLATIONS = {
    "v14_baseline": {},
    # --- suppression d'un seul composant ajouté depuis V12 ---
    "no_extra_lag24": dict(extra_lags=()),
    "no_neighbor_rolling": dict(neighbor_rolling=False),
    "no_volatility_alpha": dict(volatility_alpha=False),
    "vol_weight_default": dict(vol_weight=0.5),           # valeur DEFAULT_OPTS
    "periodic_alpha_default": dict(periodic_alpha_strength=0.5),  # valeur DEFAULT_OPTS
    "periodic_alpha_off": dict(periodic_alpha_strength=0.0),
    # --- retour complet à la recette V12 sur le harnais V14 (sanity check :
    # doit reproduire approximativement le score V12 documenté) ---
    "revert_to_v12_recipe": dict(volatility_alpha=False, extra_lags=(),
                                  neighbor_rolling=False, periodic_alpha_strength=0.5),
    # --- combinaisons des retraits individuels les plus prometteurs (remplies
    # après coup si besoin ; laissées ici en réserve à 2 retraits) ---
    "no_lag24_no_rolling": dict(extra_lags=(), neighbor_rolling=False),
    "no_lag24_periodic_default": dict(extra_lags=(), periodic_alpha_strength=0.5),
    "no_rolling_vol_default": dict(neighbor_rolling=False, vol_weight=0.5),
    # --- alpha_scale global (jamais retouché depuis son introduction) ---
    "alpha_scale_down": dict(alpha_scale=0.8),
    "alpha_scale_up": dict(alpha_scale=1.2),
    # --- vol_cap (jamais retesté) ---
    "vol_cap_low": dict(vol_cap=0.4),
    "vol_cap_high": dict(vol_cap=0.8),
    # --- contrôles négatifs : déjà montrés insensibles ailleurs (§7/§12),
    # sert à valider que le protocole repeat-1 ne bruite pas tout ---
    "corr_threshold_low": dict(corr_threshold=0.10),
    "corr_threshold_high": dict(corr_threshold=0.25),
    "k_20": dict(k=20),
    "k_60": dict(k=60),
    "euclid_weight_low": dict(euclid_weight=0.15),
    "euclid_weight_high": dict(euclid_weight=0.5),
    "n_clusters_10": dict(n_clusters=10),
    "n_clusters_25": dict(n_clusters=25),
    # --- round 2 : les 2 seuls signaux positifs du round 1 combinés / poussés
    # plus loin dans la même direction (faire plus confiance au modèle) ---
    "alpha_scale_vol_cap_combo": dict(alpha_scale=0.8, vol_cap=0.4),
    "vol_cap_lower": dict(vol_cap=0.3),
    "alpha_scale_lower": dict(alpha_scale=0.7),
    "combo_aggressive": dict(alpha_scale=0.7, vol_cap=0.3),
}


def load_results():
    try:
        return json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        return {}


def save_results(results):
    json.dump(results, open(RESULTS_PATH, "w"), indent=2)


def run_search(X, y, donors, names, nval):
    results = load_results()
    key_suffix = f"1x{nval}"
    for name in names:
        key = f"{name}_{key_suffix}"
        if key in results:
            print(f"[{key}] déjà fait, MAE={results[key]['mean']:.3f} W -- skip", flush=True)
            continue
        opts = {**V14_BASE, **ABLATIONS[name]}
        print(f"=== {key}: overrides={ABLATIONS[name]} ===", flush=True)
        t0 = time.time()
        scores = v14.run_config(key, opts, X, y, donors, n_repeats=1, n_val_sub=nval)
        results[key] = {"overrides": ABLATIONS[name], "scores": scores.tolist(),
                         "mean": float(scores.mean()), "std": float(scores.std()),
                         "time_s": time.time() - t0}
        save_results(results)

    base_key = f"v14_baseline_{key_suffix}"
    if base_key in results:
        base_mean = results[base_key]["mean"]
        print(f"\n=== Résumé recherche (delta vs {base_key} = {base_mean:.3f} W) ===", flush=True)
        rows = []
        for name in names:
            key = f"{name}_{key_suffix}"
            if key not in results:
                continue
            r = results[key]
            delta = r["mean"] - base_mean
            rows.append((delta, name, r["mean"]))
            print(f"  {name:28s} MAE={r['mean']:8.3f} W  delta={delta:+7.3f} W", flush=True)
        rows.sort()
        print("\nMeilleurs candidats (delta le plus négatif) :", flush=True)
        for delta, name, mean in rows[:5]:
            print(f"  {name}: {mean:.3f} W ({delta:+.3f} W)", flush=True)
    return results


def run_confirm(X, y, donors, names):
    results = load_results()
    v12_ref = None
    try:
        v14_confirm = json.load(open(V14_CONFIRM_PATH))
    except FileNotFoundError:
        v14_confirm = None

    for name in names:
        key = f"{name}_10x10"
        if key in results:
            print(f"[{key}] déjà fait, MAE={results[key]['mean']:.3f} +/- {results[key]['std']:.3f} W -- skip", flush=True)
            continue
        opts = {**V14_BASE, **ABLATIONS.get(name, {})}
        print(f"=== CONFIRM {key}: overrides={ABLATIONS.get(name, {})} ===", flush=True)
        t0 = time.time()
        scores = v14.run_config(key, opts, X, y, donors, n_repeats=10, n_val_sub=10)
        results[key] = {"overrides": ABLATIONS.get(name, {}), "scores": scores.tolist(),
                         "mean": float(scores.mean()), "std": float(scores.std()),
                         "time_s": time.time() - t0}
        save_results(results)

    if v14_confirm:
        print(f"\n=== Comparaison finale (protocole 10x10 identique, mêmes seeds) ===", flush=True)
        print(f"  v14_baseline (référence enregistrée) : {v14_confirm['mean']:.3f} +/- {v14_confirm['std']:.3f} W", flush=True)
        for name in names:
            key = f"{name}_10x10"
            if key not in results:
                continue
            r = results[key]
            delta = np.array(r["scores"]) - np.array(v14_confirm["scores"])
            print(f"  {name:24s} {r['mean']:8.3f} +/- {r['std']:6.3f} W  "
                  f"delta_mean={r['mean']-v14_confirm['mean']:+.3f} W  "
                  f"wins={int((delta<0).sum())}/10", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nval", type=int, default=10)
    p.add_argument("--confirm", nargs="*", default=None,
                    help="passe en mode confirmation 10x10 pour ces noms (sinon: recherche repeat-1)")
    p.add_argument("names", nargs="*", help="sous-ensemble de configs (défaut: toutes)")
    a = p.parse_args()

    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant dans l'environnement")
    tabpfn_client.set_access_token(token)

    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]

    if a.confirm is not None:
        names = a.confirm or list(ABLATIONS)
        run_confirm(X, y, donors, names)
    else:
        names = a.names or list(ABLATIONS)
        unknown = [n for n in names if n not in ABLATIONS]
        if unknown:
            raise SystemExit(f"config(s) inconnue(s): {unknown}")
        run_search(X, y, donors, names, a.nval)
