"""Compare 3 nouvelles hypothèses de features (colonnes x lignes / lignes
seules / colonnes seules, cf. discussion du 2026-08-15) contre le pipeline
V14 réel (63.24 +/- 30.46 W sur 10x10, meilleur modèle validé, cf.
`generate_submission_tabpfn_v14.py` / benchmark.md §13) -- pas un proxy
LightGBM ni un harness séparé : on réutilise `tabpfn_v14_ablation.run_config`
tel quel, avec les mêmes opts que V14 (`V14_OPTS`) plus `v15_groups` pour
activer chaque hypothèse individuellement puis toutes combinées.

Hypothèses "round 1" (implémentées dans `tabpfn_v15_features.py`, jamais
testées avant cette session) :
  - local_reselect          : voisins resélectionnés localement autour de
                              chaque trou plutôt que globalement sur toute la
                              courbe (cf. benchmark.md §10 pt.2)
  - diverse_donor_consensus : consensus sur un sous-ensemble de voisins
                              diversifiés plutôt que sur les ~40 voisins
                              bruts (variante non testée de §10 pt.5)
  - shape_match_block       : copie d'un bloc historique de la courbe par
                              alignement libre sur les deux bords -- BUG
                              identifié (2026-08-15) : constante à 0 sur 100%
                              des points observés (jamais entraînable), donc
                              score identique au baseline au bit près ; gardé
                              pour mémoire, exclu des combos "round 2".

Hypothèses "round 2" : features glanées dans les criblages faits en parallèle
sur une autre architecture (`tabpfn_column_feature_screen.py`, "colonnes
empilées", cf. `tabpfn_column_feature_screen_results.md`), rejouées ici sur
le pipeline V14 réel pour une comparaison apples-to-apples (protocole
`run_config` standard, pas le pool 8000-lignes/greedy de ce script-là) :
  - new::ewma_context (meilleure individuelle de ce criblage, -0.885 W)
  - boundary_values, gap_geometry, neighbor_dynamics (déjà dans v15_groups)
  - new::load_period_ratios
  - screen_best_combo : ewma_context+boundary_values+neighbor_dynamics, le
    combo retenu par la recherche greedy de ce criblage (-1.28 W, 3/3 seeds,
    mais sur leur architecture/sous-ensemble de seeds -- à reconfirmer ici)
  - kitchen_sink : toutes les hypothèses round 1 (sauf shape_match_block,
    cassée) + round 2 combinées

Par défaut : un seul repeat (seed=42, RANDOM_STATE de tabpfn_v14_ablation),
donc EXACTEMENT le repeat 1 déjà documenté pour V12/V13/V14 dans
benchmark.md -- comparable point par point avec ces tables, mais un seul
repeat reste très bruité (cf. variance inter-repeats 31-134 W observée dans
la table V12 10x10, §9) : à confirmer sur davantage de repeats avant de
trancher. Utiliser --repeats/--nval pour élargir si le budget API le permet.
"""
import argparse
import json
import os
import time

import tabpfn_client

from evaluate import load_data
import tabpfn_v14_ablation as v14

RESULTS_PATH = "tabpfn_v16_hypotheses_results.json"

# Identique à V14_OPTS de generate_submission_tabpfn_v14.py / BEST de
# tabpfn_v14_confirm.py : le pipeline réellement en production aujourd'hui.
V14_BASE = dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
                extra_lags=(24,), neighbor_rolling=True)

HYPOTHESES = {
    # --- round 1 (mes 3 hypothèses, cf. discussion du 2026-08-15) ---
    "v14_baseline": (),
    "v14_local_reselect": ("local_reselect",),
    "v14_diverse_donor_consensus": ("diverse_donor_consensus",),
    "v14_shape_match_block": ("shape_match_block",),  # bug connu, cf. docstring
    "v14_all3_combined": ("local_reselect", "diverse_donor_consensus", "shape_match_block"),
    # --- round 2 : meilleures individuelles trouvées dans les criblages
    # parallèles (tabpfn_column_feature_screen_results.md), rejouées ici ---
    "v14_ewma_context": ("new::ewma_context",),
    "v14_boundary_values": ("boundary_values",),
    "v14_gap_geometry": ("gap_geometry",),
    "v14_load_period_ratios": ("new::load_period_ratios",),
    "v14_neighbor_dynamics": ("neighbor_dynamics",),
    # combo greedy retenu par ce criblage-là (autre architecture) -> reconfirmé ici
    "v14_screen_best_combo": ("new::ewma_context", "boundary_values", "neighbor_dynamics"),
    # tout combiné (round 1 utile + round 2), shape_match_block exclue (bug)
    "v14_kitchen_sink": ("local_reselect", "diverse_donor_consensus", "new::ewma_context",
                          "boundary_values", "gap_geometry", "new::load_period_ratios",
                          "neighbor_dynamics"),
    # --- round 3 (2026-08-16) : 2 nouvelles idées lignes x colonnes,
    # génuinement pas encore testées (cf. liste.md) ---
    "v14_donor_residual_synchrony": ("donor_residual_synchrony",),
    "v14_micro_shift_consensus": ("micro_shift_consensus",),
    "v14_round3_combo": ("donor_residual_synchrony", "micro_shift_consensus"),
    # meilleur candidat round1 (diverse_donor_consensus) + les 2 nouvelles
    "v14_best_of_all": ("diverse_donor_consensus", "donor_residual_synchrony", "micro_shift_consensus"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeats", type=int, default=1, help="nombre de repeats (défaut: 1, le premier repeat protocolaire)")
    p.add_argument("--nval", type=int, default=10, help="colonnes de validation par repeat (défaut: 10, protocole standard)")
    p.add_argument("names", nargs="*", help="sous-ensemble d'hypothèses à lancer (défaut: toutes)")
    a = p.parse_args()

    token = os.environ.get("TABPFN_API_KEY")
    if not token:
        raise SystemExit("TABPFN_API_KEY manquant dans l'environnement")
    tabpfn_client.set_access_token(token)

    print("Chargement des données...", flush=True)
    X, y, _ = load_data()
    donors = [c for c in X if not str(c).startswith("holed_")]

    try:
        results = json.load(open(RESULTS_PATH))
    except FileNotFoundError:
        results = {}

    names = a.names or list(HYPOTHESES)
    unknown = [n for n in names if n not in HYPOTHESES]
    if unknown:
        raise SystemExit(f"hypothèse(s) inconnue(s): {unknown} (choix: {list(HYPOTHESES)})")

    key_suffix = f"{a.repeats}x{a.nval}"
    for name in names:
        key = f"{name}_{key_suffix}"
        if key in results:
            print(f"[{key}] déjà fait, MAE={results[key]['mean']:.3f} W -- skip", flush=True)
            continue
        opts = {**V14_BASE, "v15_groups": HYPOTHESES[name]}
        print(f"=== {key}: v15_groups={HYPOTHESES[name]} ===", flush=True)
        t0 = time.time()
        scores = v14.run_config(key, opts, X, y, donors, n_repeats=a.repeats, n_val_sub=a.nval)
        results[key] = {
            "groups": list(HYPOTHESES[name]),
            "scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "time_s": time.time() - t0,
        }
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    baseline_key = f"v14_baseline_{key_suffix}"
    if baseline_key in results:
        base_scores = results[baseline_key]["scores"]
        base_mean = results[baseline_key]["mean"]
        print(f"\n=== Résumé (delta vs {baseline_key} = {base_mean:.3f} W) ===", flush=True)
        for name in HYPOTHESES:
            key = f"{name}_{key_suffix}"
            if key not in results:
                continue
            r = results[key]
            delta = r["mean"] - base_mean
            per_repeat = ", ".join(f"{s:.2f}" for s in r["scores"])
            print(f"  {name:32s} MAE={r['mean']:8.3f} W  delta={delta:+7.3f} W  "
                  f"scores=[{per_repeat}]", flush=True)


if __name__ == "__main__":
    main()
