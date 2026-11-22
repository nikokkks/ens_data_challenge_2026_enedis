# Challenge Enedis — imputation de courbes électriques

Ce projet reconstruit les valeurs manquantes de 1 000 courbes synthétiques de
consommation électrique à partir de dizaines de milliers de courbes donneuses.
La métrique officielle est la MAE calculée uniquement aux positions manquantes.

## État actuel

Le meilleur modèle validé est **TabPFN V14**, avec une MAE de
**63,24 ± 30,46 W** sur le protocole apparié `10 repeats × 10 courbes`
(-4,53 % vs V12, 9 repeats sur 10 gagnés — cf. `benchmark.md` §13). V14 =
V12 + poids alpha (interpolation vs modèle) réduit davantage quand la courbe
est localement bruitée ou que les copies périodiques s'accordent, plus une
copie périodique à ±24h et une feature de tendance locale des voisins.

Générateur : `generate_submission_tabpfn_v14.py` (même schéma checkpointé
que V12, écrit `submission_tabpfn_v14.csv`). La soumission V12 précédente
reste disponible :

- [submission_tabpfn_v12.csv](submission_tabpfn_v12.csv) (référence, 66,24 W) ;
- dimensions : `1057 × 1000` ;
- aucune valeur manquante ou infinie ;
- colonnes et index identiques à `data/new_output_sample.csv`.

37 configurations supplémentaires ont été testées pour aller plus loin
(§13-15 de `benchmark.md`) : élargir le pool de voisins (rejeté, aligné
CatBoost et TabPFN), un blend alpha appris par régression OOF (rejeté,
budget d'entraînement trop faible). -4,53 % est le gain confirmé le plus
robuste depuis V12 ; les pistes non épuisées restent documentées en §15.

## Modèles les plus compétitifs

Résultats sur le même protocole réduit `10×10` :

| Rang | Modèle | MAE (W) | Écart-type | Commentaire |
|---:|---|---:|---:|---|
| 1 | **TabPFN V12** | **66,24** | 32,45 | meilleur modèle actuel |
| 2 | TabPFN C2 | 68,93 | 32,92 | calendrier + périodicité |
| 3 | **LightGBM + imputation itérative 2 passes** | **68,89** | 31,82 | meilleur modèle rapide (round 7) |
| 4 | CatBoost + imputation itérative 2 passes | 68,92 | 31,16 | confirme l'idée sur un 2e régresseur |
| 5 | Ensemble médian LightGBM/CatBoost/ExtraTrees | 69,24 | 31,43 | ancien meilleur ensemble, plus lent |
| 6 | CatBoost MAE | 69,43 | 31,50 | meilleur régresseur rapide seul |
| 7 | LightGBM V12-features | 69,75 | 31,80 | référence rapide reproductible |
| 8 | TabPFN médiane V10 | 70,06 | 31,98 | ancienne référence TabPFN |
| 9 | TabPFN moyen V10 | 70,80 | 31,79 | moyenne moins adaptée à la MAE |
| 10 | LSTM/Transformer séquentiel | 74,66 | 36,30 | baseline documentée, non compétitive |
| 11 | V8 Ridge | 77,87 | 33,67 | ancien pipeline linéaire |

Sur le protocole plus robuste `5 repeats × 100 courbes`, le meilleur résultat
rapide enregistré est LightGBM avec `k=100` à **74,50 ± 7,11 W**. Ces chiffres
ne sont pas directement comparables à ceux du tableau `10×10` : les courbes
évaluées et la variance diffèrent.

## Composition de TabPFN V12

- sélection de 40 voisins par corrélation et proximité d’amplitude ;
- 15 clusters de forme ;
- remise à l’échelle robuste médiane/MAD des voisins ;
- TabPFN avec `output_type="median"`, cohérent avec l’optimisation de la MAE ;
- saisonnalité journalière propre à la courbe ;
- sinus/cosinus de l’heure, du jour de semaine, du jour du mois et de la
  position annuelle ;
- copies périodiques `t±48`, `t±96`, `t±144`, `t±336` ;
- normalisation robuste et indicateurs de disponibilité de ces copies ;
- poids interpolation/modèle adapté à la longueur du trou, à la distance au
  bord et à l’accord entre copies périodiques ;
- `k=40` conservé : le test `k=100` a dégradé la moyenne sur 2 des 3 premiers
  repeats (`79,68` contre `79,24 W`).

## Protocoles d’évaluation

Deux protocoles coexistent dans l’historique :

1. **Protocole robuste `5×100`** : cinq splits, 100 courbes notées par split.
   À privilégier pour les modèles locaux rapides.
2. **Protocole TabPFN `10×10`** : dix splits, dix courbes notées par split,
   imposé par le coût des appels API. Toutes les comparaisons récentes sont
   appariées : mêmes graines et mêmes courbes pour chaque modèle.

Dans les deux cas, les splits portent sur les colonnes `holed_*` et la MAE est
calculée uniquement sur leurs vrais trous. Les positions où `y_train` est
lui-même manquant sont exclues.

## Commandes utiles

```bash
# Baselines historiques
.venv/bin/python evaluate.py
.venv/bin/python feature_engineering_bench.py

# Comparaison des régresseurs rapides avec les features récentes
.venv/bin/python advanced_regressors_bench.py

# Benchmark du meilleur TabPFN V12
.venv/bin/python tabpfn_five_improvements_bench.py I4_confidence_alpha

# Round 7 : idées issues de la recherche (A4 = imputation itérative, retenue)
.venv/bin/python research_ideas_bench.py A4_iterative_2pass

# Baseline séquentielle documentée : sweep de 15 configs puis évaluation 10x10
.venv/bin/python lstm_bench.py && .venv/bin/python lstm_official_eval.py

# Génération/reprise de la soumission finale, 20 colonnes en parallèle
.venv/bin/python generate_submission_tabpfn_v12.py

# Ensemble séquentiel V3 : 5 BiLSTM, split temporel 90/10, loss MAE
.venv/bin/python generate_submission_lstm_v3.py

# Entraîner/reprendre les cinq checkpoints sans générer le CSV
.venv/bin/python generate_submission_lstm_v3.py --train-only
```

L’accès TabPFN doit idéalement être fourni par `TABPFN_API_KEY`. Les anciens
scripts contiennent encore un jeton historique en clair : il doit être révoqué
et retiré du dépôt.

## Fichiers principaux

| Fichier | Rôle |
|---|---|
| `feature_engineering_bench.py` | pipeline partagé, features et évaluation |
| `tabpfn_five_improvements_bench.py` | benchmark TabPFN V12 |
| `advanced_regressors_bench.py` | LightGBM, CatBoost, ExtraTrees et ensemble |
| `generate_submission_tabpfn_v12.py` | générateur final checkpointé |
| `tabpfn_five_improvements_results.json` | scores du meilleur TabPFN |
| `advanced_regressors_results.json` | scores des régresseurs concurrents |
| `research_ideas.py` | round 7 : agrégats, bas-rang ACP, sélection locale, itératif |
| `research_ideas_bench.py` | screening round 7 sur le protocole `10×10` |
| `lstm_bench.py` | sweep 15 configs LSTM/GRU/Transformer, split temporel |
| `lstm_official_eval.py` | évaluation `10×10` du meilleur checkpoint séquentiel |
| `tabpfn_iterative_bench.py` | test (négatif) de l'itératif sur TabPFN |
| `benchmark.md` | historique détaillé, gains et échecs |
| `data/` | fichiers fournis par le challenge |
