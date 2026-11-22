"""Résumé apparié de la campagne T01--T50 (lightgbm_10x10_*) vs baseline.

Les scores par repeat sont directement comparables (mêmes graines de split
`random_state=42+i` pour dev/val, mêmes 10 colonnes de validation par
repeat entre `baseline` et chaque groupe T -- cf. `run()` dans
tabpfn_v15_bench.py) : on peut donc calculer un delta apparié repeat par
repeat, pas seulement un delta de moyennes.
"""
import json

RESULTS = "tabpfn_v15_feature_results.json"
MODEL = "lightgbm"
REPEATS_NVAL = "10x10"

d = json.load(open(RESULTS))
base_key = f"{MODEL}_{REPEATS_NVAL}_baseline"
if base_key not in d:
    raise SystemExit(f"{base_key} absent -- lance d'abord la baseline 10x10")
base = d[base_key]
base_scores = base["scores"]

rows = []
for i in range(1, 51):
    name = f"T{i:02d}"
    key = f"{MODEL}_{REPEATS_NVAL}_{name}"
    if key not in d:
        continue
    v = d[key]
    scores = v["scores"]
    n = min(len(scores), len(base_scores))
    wins = sum(1 for a, b in zip(scores[:n], base_scores[:n]) if a < b)
    deltas = [a - b for a, b in zip(scores[:n], base_scores[:n])]
    mean_delta_paired = sum(deltas) / len(deltas) if deltas else float("nan")
    rows.append(dict(name=name, mean=v["mean"], std=v["std"],
                      delta_mean=v["mean"] - base["mean"],
                      delta_paired=mean_delta_paired, wins=wins, n=n))

rows.sort(key=lambda r: r["delta_paired"])

print(f"Baseline {REPEATS_NVAL}: {base['mean']:.3f} +/- {base['std']:.3f} W "
      f"({len(base_scores)} repeats)\n")
print(f"{'Groupe':6} {'MAE':>10} {'std':>9} {'Δ moyennes':>11} {'Δ apparié':>11} {'gagnés':>8}")
for r in rows:
    print(f"{r['name']:6} {r['mean']:10.3f} {r['std']:9.3f} {r['delta_mean']:11.3f} "
          f"{r['delta_paired']:11.3f} {r['wins']:>4}/{r['n']}")

print(f"\n{len(rows)}/50 groupes évalués.")
improved = [r for r in rows if r["delta_paired"] < 0 and r["wins"] > r["n"] / 2]
print(f"Groupes avec delta apparié négatif ET majorité de repeats gagnés: "
      f"{[r['name'] for r in improved]}")
