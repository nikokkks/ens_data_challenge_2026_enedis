# V15 — EDA et criblage de 15 nouveaux groupes de features

## EDA

- période : 9 au 31 janvier 2023, cadence régulière de 30 minutes ;
- 129 909 valeurs absentes, dont 128 853 évaluables ;
- 55 966 séquences manquantes : médiane 1 pas, P95 3 pas, P99 48 pas ;
- interpolation linéaire : 101,95 W ;
- corrélation de l'erreur par courbe avec la volatilité : 0,819 ;
- corrélation avec l'amplitude : 0,827 ;
- erreur plus forte en milieu de journée (environ 111–112 W) que la nuit
  (88 W), et le week-end (106,36 contre 100,24 W).

Le rapport exhaustif et reproductible est dans `eda_v15_report.json` et se
recalcule avec `python eda_v15.py`.

## Écran LightGBM apparié 3×10

Référence recalculée sur les mêmes splits : **85,414 ± 29,091 W**.

| Groupe | MAE (W) | Delta | Repeats gagnés |
|---|---:|---:|---:|
| volatilité locale multi-échelle | 84,463 | -0,951 | 3/3 |
| pentes aux frontières | 84,528 | -0,886 | 2/3 |
| dynamique des voisins | 85,181 | -0,232 | 3/3 |
| résidus périodiques | 85,320 | -0,094 | 2/3 |
| meilleur combo (volatilité + dynamique + résidus) | **83,804** | **-1,609** | **3/3** |

Les onze autres groupes sont neutres ou négatifs. Les valeurs aux frontières
et la fiabilité KNN brute sont notamment rejetées à cause de fortes pertes sur
les courbes difficiles.

## Confirmation LightGBM appariée 10×10

| Configuration | MAE (W) | std | Delta | Repeats gagnés |
|---|---:|---:|---:|---:|
| baseline | 71,911 | 32,435 | — | — |
| meilleur combo V15 | 71,894 | 33,126 | **-0,017** | 5/10 |

Verdict : le gain de l'écran 3×10 ne généralise pas. Cette V15 ne remplace
donc pas TabPFN V14 et aucun générateur de soumission V15 n'est créé avant une
confirmation positive avec le vrai modèle TabPFN.

## Reproduction et confirmation TabPFN

```bash
.venv/bin/python tabpfn_v15_bench.py --model lightgbm
.venv/bin/python tabpfn_v15_bench.py --model lightgbm --confirm \
  baseline local_volatility+neighbor_dynamics+periodic_residuals

# Nécessite TABPFN_API_KEY : écran apparié, puis confirmation éventuelle.
.venv/bin/python tabpfn_v15_bench.py --model tabpfn \
  baseline local_volatility neighbor_dynamics periodic_residuals \
  local_volatility+neighbor_dynamics+periodic_residuals
.venv/bin/python tabpfn_v15_bench.py --model tabpfn --confirm \
  baseline local_volatility+neighbor_dynamics+periodic_residuals
```

Les sorties détaillées, y compris chaque repeat, sont enregistrées dans
`tabpfn_v15_feature_results.json`.
