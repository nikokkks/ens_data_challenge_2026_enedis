# Rapport de benchmark — Challenge Enedis

Ce document synthétise les expériences utiles du dépôt. Les anciens comptes
rendus redondants ont été fusionnés ; les valeurs ci-dessous sont recalculées
depuis les fichiers JSON de résultats.

## 1. Données et métrique

- `X_train` : 1 057 timestamps espacés de 30 minutes, 20 000 donneurs et
  1 000 courbes `holed_*` ;
- `y_train` : vérité des 1 000 courbes trouées ;
- `X_test` : 37 140 donneurs et 1 000 courbes à compléter ;
- période : du 9 au 31 janvier 2023 ;
- métrique : MAE uniquement sur les valeurs réellement supprimées.

Observations structurantes :

- 32 % des valeurs manquantes appartiennent à des trous isolés ;
- 41 % appartiennent à des trous d’au moins une journée (48 points) ;
- l’interpolation est forte près des bords mais mauvaise au centre des longs
  trous ;
- les corrélations sont faibles globalement mais nettement plus fortes entre
  courbes de même famille ;
- les périodicités journalières et hebdomadaires sont directement accessibles
  grâce aux timestamps.

## 2. Attention aux protocoles

| Protocole | Usage | Avantage | Limite |
|---|---|---|---|
| `5×100` | modèles rapides, historique V0–LightGBM | faible variance | coûteux avec TabPFN cloud |
| `10×10` | comparaisons TabPFN récentes | comparaisons appariées abordables | écart-type élevé |

Il ne faut pas classer un score `5×100` et un score `10×10` uniquement sur
leur moyenne. Les tableaux sont donc séparés.

## 3. Progression historique sur le protocole robuste `5×100`

| Étape | Méthode | MAE (W) | std | Décision |
|---|---|---:|---:|---|
| V0 | interpolation linéaire | 98,11 | 8,82 | baseline officielle |
| V1 | KNN inter-courbes, top 20 | 88,32 | 7,65 | gain majeur |
| V2 | clustering seul | 90,07 | 7,93 | rejeté seul |
| V3 | alpha selon longueur du trou | 82,17 | 7,94 | gain majeur |
| V4 | Ridge local + calendrier | 80,81 | 6,64 | retenu |
| V5 | scaling robuste | 80,81 | 6,64 | neutre mais stabilisant |
| V6 | stacking Ridge | 86,96 | 7,34 | rejeté |
| XGBoost MAE | arbres locaux | 76,37 | 7,55 | retenu |
| LightGBM R1 | boosting MAE | 75,95 | 7,31 | meilleure base rapide |
| E6 | saisonnalité + k40/c15 | 74,91 | 7,30 | amélioration confirmée |
| H1 | LightGBM E6, k=100 | **74,50** | **7,11** | meilleur `5×100` enregistré |

La baseline passe ainsi de 98,11 à 74,50 W, soit environ **24,1 % de gain**
sur ce protocole robuste.

## 4. Progression TabPFN sur le protocole apparié `10×10`

| Version | Évolution | MAE (W) | std | Gain vs précédente |
|---|---|---:|---:|---:|
| V8 Ridge | régression linéaire locale | 77,87 | 33,67 | — |
| V8 TabPFN | remplacement de Ridge | 72,25 | 32,38 | −5,61 |
| V10 TabPFN | euclidien + saisonnalité + k40/c15 | 70,80 | 31,79 | −1,45 |
| G6 | sortie médiane explicite | 70,06 | 31,98 | −0,74 |
| C1 | calendrier cyclique complet | 69,06 | 33,11 | −1,00 |
| C2 | copies périodiques + consensus | 68,93 | 32,92 | −0,13 |
| **V12** | normalisation, disponibilité, alpha de confiance | **66,24** | **32,45** | **−2,69** |
| V13 | V12 + pente/courbure du consensus des voisins | 66,38 | 32,25 | +0,14 vs V12 |

V12 bat C2 sur **10 repeats sur 10**. V13 ne la remplace pas : sa MAE est
supérieure de 0,14 W malgré 6 repeats gagnés sur 10. Le gain total depuis V8 Ridge est de
11,63 W, soit environ **14,9 %**.

### Détail apparié C2 → V12

| Repeat | C2 | V12 | Δ V12−C2 |
|---:|---:|---:|---:|
| 1 | 127,84 | 122,39 | −5,46 |
| 2 | 68,19 | 65,00 | −3,19 |
| 3 | 52,17 | 50,35 | −1,82 |
| 4 | 32,56 | 31,66 | −0,90 |
| 5 | 41,39 | 40,60 | −0,79 |
| 6 | 52,14 | 49,67 | −2,47 |
| 7 | 60,94 | 59,25 | −1,69 |
| 8 | 52,34 | 49,78 | −2,57 |
| 9 | 135,10 | 134,26 | −0,84 |
| 10 | 66,65 | 59,46 | −7,19 |

## 5. Meilleurs modèles rapides avec les features V12 (`10×10`)

| Rang | Régresseur local | MAE (W) | std | Temps relatif |
|---:|---|---:|---:|---|
| 1 | **LightGBM MAE + 2 passes (A4)** | **68,89** | 31,82 | rapide |
| 2 | CatBoost MAE + 2 passes (A4c) | 68,92 | 31,16 | moyen |
| 3 | médiane LightGBM/CatBoost/ExtraTrees | 69,24 | 31,43 | lent |
| 4 | CatBoost MAE | 69,43 | 31,50 | moyen |
| 5 | LightGBM MAE | 69,75 | 31,80 | rapide |
| 6 | HistGradientBoosting MAE | 70,48 | 33,36 | moyen |
| 7 | ExtraTrees rapide | 70,78 | 31,73 | moyen |

L’imputation itérative à deux passes (round 7, § 7bis) est désormais le meilleur
modèle rapide : elle bat l’ensemble médian tout en restant la variante la plus
rapide. Elle reste 2,65 W derrière TabPFN V12.

## 6. Features retenues dans V12

1. **Voisins inter-courbes** : corrélation de Pearson complétée par une
   proximité euclidienne sensible au niveau ; 40 voisins maximum.
2. **Clustering** : 15 familles de forme, avec repli sur le pool complet si
   le cluster est trop petit.
3. **Scaling robuste** : médiane/MAD avec planchers et ratios bornés.
4. **Temps** : sinus/cosinus heure, jour de semaine, jour du mois et position
   annuelle ; indicateur week-end.
5. **Saisonnalité propre** : médiane observée au même créneau de 30 minutes.
6. **Copies périodiques** : valeurs à `±48`, `±96`, `±144`, `±336` points.
7. **Confiance périodique** : médiane, MAD, nombre de copies et huit drapeaux
   de disponibilité.
8. **Alpha adaptatif** : interpolation privilégiée pour les petits trous et
   près des bords ; modèle privilégié au centre des longs trous lorsque les
   copies périodiques sont cohérentes.
9. **Objectif MAE** : prédiction médiane explicite de TabPFN.

## 7bis. Round 7 — idées issues de la recherche bibliographique

Cinq pistes tirées de la littérature récente (TSI-Bench et le survey IJCAI 2025
sur l’imputation profonde, le benchmark smart-meter arXiv 2501.07276, la
complétion matricielle par plus proches voisins arXiv 2506.04166, MICE) ont été
testées sur le protocole `10x10` avec le MEME régresseur et les MEMES features
que la baseline appariée `R0_lightgbm_reference` (69,75 W) — seule l’idée
change. Scripts : `research_ideas.py`, `research_ideas_bench.py`.

| Idée | MAE (W) | Δ vs R0 | Repeats gagnés | Verdict |
|---|---:|---:|---:|---|
| A0 contrôle (options désactivées) | 69,75 | +0,00 | 0/10 | reproduit R0 au bit près |
| A1 agrégats de 500 voisins | 69,80 | +0,05 | 6/10 | rejeté |
| A2 complétion bas-rang ACP (r=20) | 70,04 | +0,29 | 4/10 | rejeté |
| A3 sélection locale, fenêtre ±96 | 69,75 | +0,00 | 0/10 | **no-op** (voir ci-dessous) |
| A3b fenêtre ±12 | 69,60 | −0,15 | 4/10 | non concluant |
| A3c fenêtre ±6 | 69,58 | −0,17 | 6/10 | marginal |
| **A4 imputation itérative, 2 passes** | **68,89** | **−0,86** | **7/10** | **retenu** |
| A4b itératif, 3 passes | 69,26 | −0,49 | 6/10 | 2 passes suffit |
| A4c itératif + CatBoost | 68,92 | −0,51 vs R1 | 7/10 | confirme A4 |
| A4d itératif + agrégats | 69,65 | −0,10 | 6/10 | les agrégats n’ajoutent rien |

Le contrôle A0 reproduisant R0 **exactement sur les 10 repeats**, tout écart
observé est bien imputable à l’idée testée et non au harnais.

**A3 était un piège de mesure.** Avec `gap_window=96`, le voisinage des trous
couvre 100 % des points observés (les trous sont partout : ~56 trous par
colonne), donc la sélection est inchangée et le score identique au bit près.
Seules des fenêtres ≤ 12 modifient réellement le classement — et le gain
devient alors marginal.

### A4 — imputation itérative (la seule idée retenue)

Principe (MICE / self-training) : après une première passe, les trous sont
comblés provisoirement ; les copies périodiques `t±48/±96/±144/±336` — qui
étaient indisponibles là où le jour ou la semaine voisine était elle-même
trouée — redeviennent renseignées pour la seconde passe, qui recalcule
saisonnalité, copies périodiques et alpha de confiance sur la courbe complétée.
Aucune fuite : les valeurs de remplissage dérivent uniquement des donneuses et
des points observés, jamais de `y_train`.

| Repeat | R0 LightGBM | A4 2 passes | Δ |
|---:|---:|---:|---:|
| 1 | 124,67 | 124,73 | +0,06 |
| 2 | 67,61 | 68,11 | +0,50 |
| 3 | 57,10 | 54,74 | −2,36 |
| 4 | 34,22 | 33,98 | −0,24 |
| 5 | 40,77 | 40,67 | −0,10 |
| 6 | 55,20 | 54,31 | −0,89 |
| 7 | 60,30 | 59,29 | −1,01 |
| 8 | 52,02 | 52,32 | +0,30 |
| 9 | 134,67 | 133,74 | −0,93 |
| 10 | 70,93 | 66,96 | −3,97 |

**A4 ne transfère pas à TabPFN** (`tabpfn_iterative_bench.py`) : 66,32 contre
66,24 W pour I4, +0,08 W, meilleur sur seulement 3/10 repeats. Le gain est
concentré sur les deux repeats les plus difficiles (repeat 1 : −2,47 ;
repeat 9 : −2,07) et négatif ailleurs. Interprétation : TabPFN exploite déjà
nativement les copies périodiques manquantes comme information (valeurs NaN
en entrée), donc les combler artificiellement n’apporte rien et ajoute du
bruit. L’idée n’aide que les régresseurs en arbres, qui traitent les NaN de
façon plus fruste.

## 7ter. LSTM / GRU / Transformer séquentiel

Un générateur expérimental plus complet est disponible dans
`generate_submission_lstm_v3.py`. Il utilise cinq BiLSTM-attention de graines
42 à 46, une validation chronologique 90/10, une loss MAE et 93 canaux
(calendrier, périodicités, géométrie des trous, profils propres, clusters et
consensus de 40 voisins). La soumission est la moyenne des cinq modèles.

Cette V3 ne doit pas être ajoutée au classement ci-dessous avant un véritable
entraînement complet et une évaluation sans fuite : ses modèles de soumission
peuvent exploiter `y_train` comme courbes auto-supervisées, ce qui interdit de
les réutiliser tels quels pour noter les trous de `y_train`.

Avis a priori (confirmé) : peu de chances. Le repo montrait déjà qu’un modèle
GLOBAL partagé entre courbes plafonne (91,52 W seul, 77,44 W avec les
périodicités), alors que les gains viennent de l’adaptation SPECIFIQUE à
chaque colonne cible. TabPFN gagne précisément parce qu’il ré-apprend une
relation propre à chaque colonne par in-context learning ; un réseau
pré-entraîné fige au contraire une dynamique moyenne, et les courbes de test
étant inconnues à l’entraînement, aucun embedding par courbe n’est possible.

Protocole (`lstm_bench.py`, `lstm_official_eval.py`) — deux niveaux de split,
volontairement distincts :

1. **Split temporel** (interne, early stopping uniquement) : entraînement
   auto-supervisé sur les 20 000 courbes donneuses, fenêtres d’entraînement
   dans `[0, 792)` (9 → 25 janvier), fenêtres de validation dans `[792, 1057)`
   (25 → 31 janvier). Trous artificiels tirés de la distribution empirique
   réelle (75 % de trous d’un pas, 90 % des points manquants dans des trous
   ≤ 48). Meilleur checkpoint retenu sur cette courbe de validation.
2. **Split par colonnes** (externe) : la mesure officielle `10x10`,
   `random_state=42`, identique à TabPFN I4 et R0.

C’est cohérent avec la métrique officielle car les vrais trous sont répartis
sur toute la période : le split temporel est donc plus strict que nécessaire
et ne sert qu’à arrêter l’entraînement honnêtement. Aucune fuite possible —
les courbes d’entraînement sont disjointes des `holed_*` et `y_train` n’entre
jamais dans l’entraînement.

**Piège corrigé en cours de route** : la convention médiane/MAD du dépôt fait
exploser le z-score jusqu’à 5,6e5 sur les 2,3 % de courbes quasi plates
(paliers nocturnes, MAD au plancher), ce qui faisait diverger l’entraînement
(loss 0,84 → 278 en 100 pas). L’échelle retenue est le maximum de trois
mesures d’étalement, avec bornage à ±20.

15 configurations, classées sur la validation temporelle :

| Config | val MAE (norm.) | Config | val MAE (norm.) |
|---|---:|---|---:|
| L13 Transformer | **0,4779** | L15 cible résiduelle | 0,5272 |
| L08 lr 3e-3 | 0,5219 | L04 3 couches | 0,5279 |
| L11 hidden 256 | 0,5253 | L03 1 couche | 0,5289 |
| L02 hidden 128 | 0,5264 | L09 lr 3e-4 | 0,5311 |
| L10 dropout 0,3 | 0,5268 | L06 fenêtre 144 | 0,5358 |
| L12 GRU | 0,5269 | L05 fenêtre 96 | 0,5489 |
| L01 BiLSTM base | 0,5270 | L07 fenêtre 240 | 0,5498 |
| | | L14 large régularisé | 0,5531 |

Les 15 configurations tiennent dans une fourchette de 0,053 : l’architecture
ne change quasiment rien, ce qui est en soi le résultat — la limite n’est pas
la capacité du réseau mais l’information à laquelle il a accès.

Meilleur checkpoint (L13 Transformer) sur le protocole officiel `10x10` :

| Modèle | MAE (W) | std |
|---|---:|---:|
| V0 interpolation linéaire | 97,17 | 46,62 |
| LSTM/Transformer seul | 76,10 | 36,51 |
| LSTM/Transformer + mélange alpha-interpolation | 74,66 | 36,30 |
| R0 LightGBM (features V12, voisins) | 69,75 | 31,80 |
| **TabPFN I4** | **66,24** | 32,45 |

Le réseau bat nettement l’interpolation (−23 %) — il a donc bien appris
quelque chose de la forme des courbes — mais reste 8,4 W derrière le pipeline
à voisins et 10,4 W derrière TabPFN. Conclusion conforme à l’avis a priori :
la dynamique intra-courbe est déjà saturée par l’interpolation et les copies
périodiques ; ce qui fait la performance ici, c’est l’information
inter-courbes, qu’un modèle séquentiel global ne voit pas.

## 7. Ablations et échecs utiles

| Idée | Résultat | Conclusion |
|---|---:|---|
| PCHIP en remplacement | 1 808 W sur `10×10` | explosions d’extrapolation |
| PCHIP comme feature | 108,49 W sur `5×100` | rejeté |
| seuil corrélation 0,85 | 73,71 W | trop strict |
| interpolation forcée trous ≤3 | 71,36 W | pire que V10 |
| plafond supérieur | 70,79 W | neutre |
| lags directs avant/après | 107,36 W sur `5×100` | décalage train/trous longs |
| SoftImpute/NMF | 84,04 W | structure locale mieux exploitée par voisins |
| modèle global vrais trous | 91,52 W | insuffisant seul |
| + périodicités au modèle global | 77,44 W | gros gain, toujours non compétitif |
| + 1 000 donneurs pseudo-masqués | 77,20 W | gain marginal |
| stacking classique | 79,39 à 86,96 W | surapprentissage/manque d’OOF |
| `k=100` dans TabPFN V12, 3 repeats | 79,68 vs 79,24 W | 60 voisins bruyants, rejeté |
| `k=10 000` dans CatBoost V12 | 68,78 vs 69,43 W | −0,65 W mais porté par un seul repeat (−5,87), meilleur sur 4/10 seulement, ~60x plus lent : non retenu |
| agrégats de 500 voisins (A1) | 69,80 W | résumer un large pool n'aide pas plus que les 40 bruts |
| complétion bas-rang ACP (A2) | 70,04 W | la base globale ignore la spécificité de chaque courbe |
| itératif 3 passes (A4b) | 69,26 W | la 2e passe capte tout le gain, la 3e ajoute du bruit |
| itératif sur TabPFN | 66,32 vs 66,24 W | TabPFN exploite déjà les NaN périodiques |
| LSTM/GRU/Transformer séquentiel | 74,66 W | modèle global : ne voit pas l'information inter-courbes |

À noter : `k=100` aide LightGBM sur `5×100` mais pas TabPFN sur les trois
premiers repeats. Le réglage du nombre de voisins dépend donc du régresseur.

## 8. Soumission finale

La soumission actuelle est `submission_tabpfn_v12.csv` :

- TabPFN V12 ;
- pool de 58 140 donneurs (`X_test` + `X_train` + `y_train`) ;
- 20 colonnes calculées en parallèle ;
- 1 000 prédictions TabPFN réussies, aucun repli KNN ;
- fichier validé contre `data/new_output_sample.csv`.

Commande de génération ou de reprise :

```bash
.venv/bin/python generate_submission_tabpfn_v12.py
```

Le générateur écrit régulièrement `submission_tabpfn_v12_partial.csv` et
reprend automatiquement les colonnes déjà présentes.

## 9. Criblage V13 de 15 axes de features

Analyse préalable : les trois jeux couvrent les mêmes 1 057 demi-heures, du
9 au 31 janvier 2023. Les courbes trouées ont environ 12,2 % de valeurs
manquantes et près de 56 segments manquants par courbe. La médiane des
longueurs vaut 1 point, mais les plus grands trous du test atteignent 144
points. Cela favorise un mélange interpolation locale, répétitions
journalières et information inter-courbes.

Criblage apparié LightGBM `3×20` (référence V12 : **80,744 W**) :

| Axe ajouté à V12 | MAE | Δ vs V12 | Décision |
|---|---:|---:|---|
| J−14/J+14 | 81,302 | +0,558 | rejeté |
| J−4/J−5/J−6 et symétriques | 82,725 | +1,981 | rejeté |
| profil heure × jour de semaine | 86,524 | +5,780 | rejeté |
| profil heure × ouvré/week-end | 81,453 | +0,709 | rejeté |
| moyenne et MAD par demi-heure | 81,083 | +0,339 | rejeté |
| consensus donneurs pondéré | 81,303 | +0,559 | rejeté |
| médiane et MAD des donneurs | **80,494** | **−0,250** | candidat |
| quantiles des donneurs | 81,013 | +0,269 | rejeté |
| incertitude/désaccord donneurs | 80,990 | +0,246 | rejeté |
| consensus glissant 3 points | 81,167 | +0,423 | rejeté |
| consensus glissant 7 points | 81,085 | +0,341 | rejeté |
| consensus glissant 13 points | 81,380 | +0,636 | rejeté |
| pente et courbure du consensus | **80,636** | **−0,108** | candidat |
| harmoniques calendrier d’ordre 2–4 | 80,834 | +0,090 | rejeté |
| profil Fourier quotidien propre | 81,294 | +0,550 | rejeté |

Le criblage initial TabPFN médian sur les trois premiers repeats appariés
`3×10` donnait :

| Configuration | Scores | MAE | Δ vs V12 |
|---|---|---:|---:|
| V12 reproduite | 122,386 / 65,000 / 50,337 | 79,241 | — |
| + médiane/MAD donneurs | 122,320 / 65,519 / 50,839 | 79,559 | +0,318 |
| **V13 : + pente/courbure** | **121,074 / 64,751 / 50,758** | **78,861** | **−0,380 (−0,48 %)** |

La confirmation complète avec le même protocole `10×10` que la section 4
invalide finalement le faible gain initial :

| Repeat | V12 | V13 | Δ V13−V12 |
|---:|---:|---:|---:|
| 1 | 122,386 | 121,074 | −1,313 |
| 2 | 65,000 | 64,751 | −0,250 |
| 3 | 50,348 | 50,758 | +0,410 |
| 4 | 31,660 | 31,792 | +0,132 |
| 5 | 40,603 | 40,579 | −0,024 |
| 6 | 49,668 | 48,993 | −0,675 |
| 7 | 59,246 | 58,358 | −0,888 |
| 8 | 49,776 | 49,743 | −0,033 |
| 9 | 134,262 | 134,554 | +0,292 |
| 10 | 59,461 | 63,177 | +3,716 |
| **Moyenne** | **66,241** | **66,378** | **+0,137** |

V13 gagne **6 repeats sur 10**, avec une médiane des deltas de −0,029 W,
mais sa dégradation sur le repeat 10 annule les petits gains. Elle est donc
**rejetée pour la MAE moyenne** (+0,21 %). Son générateur reste disponible
pour expérimentation dans `generate_submission_tabpfn_v13.py`; V12 reste la
soumission de référence robuste à **66,24 ± 32,45 W sur 10×10**.

Les axes proviennent notamment des travaux sur le copy-paste de blocs
énergétiques, les modèles de forme/énergie, la complétion faible-rang et les
ensembles TabPFN. Les résultats locaux priment toutefois sur la plausibilité
théorique : treize blocs ont été rejetés après ablation.

## 10. Prochaines expériences prioritaires

1. confirmer V12 sur davantage de courbes par repeat si le budget API le
   permet ;
2. sélectionner des voisins spécifiques à chaque trou plutôt qu’à la courbe
   entière ;
3. copy-paste de blocs journaliers avec alignement sur les deux bords ;
4. apprendre un mélange OOF TabPFN/CatBoost selon longueur et confiance ;
5. tester des agrégats diversifiés de 100 voisins plutôt que 100 voisins bruts.

## 11. Fine-tuning CatBoost GPU, k élargi (`catboost_gpu_finetune.py`)

TabPFN n'est pas joignable dans cet environnement (pas d'accès réseau) : ce
fine-tuning a donc porté sur CatBoost, le meilleur régresseur rapide
disponible localement (69,43 W en référence, cf. §5), avec GPU (RTX 3060)
et un pool de voisins élargi (`k_max=500`, sélection réelle par seuil de
corrélation 0,15 → ~421 voisins effectifs mesurés sur une colonne type,
contre 40 auparavant).

Écran `3×15` sur 7 configs (depth 4→8, lr 0,03→0,1, l2_leaf_reg 1→10,
bagging_temperature 0,3→2,0) : toutes les configs "profondes" tournent
entre 103,65 et 103,84 W, quasi indiscernables du k=40 de référence
(103,71 W) tout en étant jusqu'à 27× plus lentes. Seule la config légère
(`depth=4, lr=0,1, iterations=200, l2=1`) ressort à 103,44 W.

Confirmation `8×30` (mêmes splits pour les deux configs) :

| Config | MAE | std | Temps |
|---|---:|---:|---:|
| k=500, hyperparamètres légers | 96,61 | 23,07 | 246 s |
| k=40 (référence) | 96,96 | 23,11 | 118 s |

Gain final : **−0,35 W (−0,36 %)**, dans le bruit (std ≈ 23 W), pour un
coût GPU 2× plus élevé (plus de features à transférer/construire par
colonne). **k=500 et le réglage d'hyperparamètres seuls sont donc rejetés**
— confirme et généralise à CatBoost le résultat déjà observé sur TabPFN
(`k=100` rejeté, cf. §7).

Diagnostic complémentaire (`_alpha_ablation.py`, 20 colonnes, un split
fixe) : le poids `alpha` donné à l'interpolation dans le blend final est en
moyenne **0,39** sur un échantillon de ~28 000 points manquants (médiane
0,51 — la moitié des points manquants ont l'interpolation qui pèse plus de
50 % dans la prédiction). Sur cet échantillon précis, alpha=1 (pure
interpolation) bat à la fois le blend actuel et le modèle seul — signal
bruité sur un seul split, mais qui pointe la vraie cause du plafond : le
blend heuristique (§ description du benchmark, `continuous_alpha`) écrase
une grande partie du signal que le régresseur pourrait apporter, quels que
soient son nombre de voisins ou ses hyperparamètres. **Conclusion : la
prochaine amélioration doit porter sur l'apprentissage du poids alpha
(OOF), pas sur plus de features/hyperparamètres pour le régresseur local**
(point 4 de la liste ci-dessus).

## 13. V14 : criblage de 20+16 idées, réduction du poids alpha (`tabpfn_v14_*.py`)

Suite directe du diagnostic §11 (le poids alpha à l'interpolation plafonne
le régresseur) : 36 configurations testées en écran apparié 3×10 (mêmes
graines que la table V12, réf. 79,241 W) sur le pipeline exact de
`generate_submission_tabpfn_v12.py`, en 5 rounds cumulatifs.

Round 1 (20 idées indépendantes) : deux gains nets se détachent très
au-dessus du bruit des 18 autres (qui restent tous sous 1 W) :
`volatility_alpha` (réduit alpha quand la courbe est bruitée localement,
idée déjà codée mais jamais activée depuis V9) à **-2,40 W**, et
`periodic_alpha_strength` porté de 0,5 (valeur V12) à 0,8 à **-1,72 W**.

Rounds 2-5, poussée et combinaison de ces deux leviers :

| Config | Δ vs V12 (écran 3×10) |
|---|---:|
| `volatility_alpha` seul (vol_weight=0,5) | -2,40 W |
| `vol_weight=0,8` | -3,04 W |
| `periodic_alpha_strength=1,0` | -2,62 W |
| combo des deux (vol_weight=0,8 + strength=1,0) | -3,51 W |
| combo + lag periodique ±24h + moyenne/écart-type glissants du consensus voisins | **-4,98 W** |
| + facteur d'échelle global sur alpha (round 5) | aucun gain supplémentaire, plafond atteint |

Empiler brutalement TOUTES les petites idées positives (`k=60`,
`euclid_weight=0,5`, lags ±24h/±672, rolling) dégrade au lieu d'additionner
(-2,10 W au lieu de -3,51 W) : les gains individuels ne sont pas additifs,
il a fallu les ajouter un par un pour trouver la bonne combinaison.

**Configuration retenue (V14)** :
```python
dict(volatility_alpha=True, vol_weight=0.8, periodic_alpha_strength=1.0,
     extra_lags=(24,), neighbor_rolling=True)
```

Confirmation complète `10×10` (mêmes graines que la table V12) :

| Config | MAE | std |
|---|---:|---:|
| V12 (référence) | 66,24 | 32,45 |
| **V14** | **63,24** | 30,46 |

**Gain confirmé : -3,00 W (-4,53 %), 9 repeats sur 10 gagnés** (seul le
repeat 10 est légèrement perdu, +1,17 W) — beaucoup plus robuste que le
rejet de V13 (6/10 seulement, §9). C'est la plus grosse amélioration
validée depuis V12, et elle confirme le diagnostic du §11 : réduire encore
le poids donné à l'interpolation (via la volatilité locale et l'accord des
copies périodiques) rapporte plus que n'importe quel ajout de
features/hyperparamètres testé jusqu'ici.

Code : `tabpfn_v14_ablation.py` (pipeline paramétrable), `tabpfn_v14_screen.py`
et `tabpfn_v14_round{2,3,4,5}.py` (criblages successifs),
`tabpfn_v14_confirm.py` (confirmation 10×10),
`generate_submission_tabpfn_v14.py` (soumission finale).

## 14. Blend alpha appris par OOF (`tabpfn_oof_alpha.py`) — rejeté

Tentative de remplacer la formule alpha à la main de V14 par un blend appris
(Ridge positif sur `[interp_pred, model_pred, run_len, edge_dist,
periodic_agreement, volatility] -> y_true`), entraîné sur 60 colonnes et
testé sur 30 colonnes fraîches (seeds 999-1001, disjointes de tous les
criblages précédents).

**Incident de fuite de données corrigé en cours de route** : la première
version du script réutilisait les colonnes de test elles-mêmes comme
donneurs dans le pool (`y[cols]` avec `cols` = train+test), donnant à TabPFN
un accès quasi direct à la vérité (8,61 W en sortie, un chiffre implausible
qui a immédiatement signalé le problème). Corrigé avec un pool de donneurs
"vérité" (`helper_cols`) strictement disjoint des colonnes évaluées.

Résultat propre sur le split frais :

| Méthode | MAE |
|---|---:|
| interpolation seule | 111,46 W |
| TabPFN seul (sans blend) | 75,93 W |
| blend V14 (formule à la main) | **75,57 W** |
| blend appris (Ridge OOF, 60 colonnes train) | 80,23 W |

**Rejeté** : le blend appris est moins bon que la formule V14. 60 colonnes
(~7 800 points) sont probablement insuffisantes pour qu'un Ridge linéaire
batte une heuristique déjà affinée sur plusieurs rounds ; il faudrait bien
plus de colonnes d'entraînement et/ou un modèle non linéaire pour capter la
logique par paliers de `continuous_alpha`. Piste à reprendre avec un budget
d'entraînement nettement plus large si prioritaire.

## 15bis. EDA ciblée et 6 idées dérivées — toutes rejetées (`tabpfn_v15_daytype_results.json`)

Nouvelle EDA (300 courbes donneuses échantillonnées) pour chercher des axes
non encore exploités avant de continuer le feature engineering :

- **Écart semaine/week-end** : médiane 20% de différence relative de niveau
  entre jours ouvrés et week-end, 64% des courbes avec >15% d'écart, 31%
  avec >30%. La feature de saisonnalité de V12/V14
  (`same_curve_seasonality_feature`) mélange les deux dans un seul médian
  par créneau horaire, sans distinction.
- **Décalage de type de jour dans les copies périodiques** : les lags ±96h
  et ±144h tombent sur le mauvais type de jour (semaine ↔ week-end) 50-55%
  du temps (ex. vendredi+3j = lundi), contre 28,5% pour ±48h et 0% pour
  ±336h (toujours le même jour de semaine).
- **Dérive de niveau** : 52% des courbes échantillonnées montrent >20% de
  variation de niveau entre la 1ère et la 2e moitié des 3 semaines de
  données, une dérive locale qu'aucune feature actuelle ne capture
  directement (les copies périodiques comparent au jour/semaine précédent,
  pas de tendance propre à la courbe).

Six variantes testées sur l'écran apparié 3×10, ajoutées une par une au
meilleur combo V14 (référence : -4,98 W) :

| Idée | Δ vs V12 (écran) | Verdict |
|---|---:|---|
| saisonnalité week-end/semaine séparée | -4,39 W | rejeté |
| copies périodiques masquées si mauvais type de jour | -2,79 W | rejeté |
| les deux combinées | -2,92 W | rejeté |
| flag de correspondance day-type (additif, sans masquer) | -4,78 W | rejeté |
| tendance locale (fenêtre 72 points, ~1,5j) | -3,64 W | rejeté |
| tendance locale (fenêtre 144 points, ~3j) | -4,06 W | rejeté |

**Toutes rejetées** : chacune dégrade par rapport à la référence V14 seule
(-4,98 W). Les patterns trouvés en EDA (écart semaine/week-end, dérive de
niveau) sont réels dans les données brutes, mais n'apportent rien une fois
ajoutés au pipeline — probablement parce que TabPFN a déjà accès à
l'information nécessaire via les features calendaires globales (sin/cos
jour de semaine déjà dans `time_features_full_seasonality`) et les valeurs
brutes des copies périodiques/voisins, et que les résumer explicitement
n'ajoute que du bruit/dimensionnalité plutôt que du signal nouveau. Ce
constat rejoint un motif déjà observé dans ce dépôt (§9, §13) : le levier
le plus productif reste le réglage du blend alpha (poids interpolation vs
modèle), pas l'ajout de features au régresseur — TabPFN, modèle
"foundation" par apprentissage en contexte, semble déjà extraire l'essentiel
du signal brut sans qu'un résumé statistique manuel soit nécessaire.

## 15ter. Autres angles : clustering par profil, OOF alpha à grande échelle

Deux pistes supplémentaires, non liées au feature engineering direct :

**Clustering par profil journalier** (`tabpfn_v14_ablation.fit_shape_clusters_profile`) :
un vrai DTW entre 20 000 donneurs de 1057 points était hors de portée en
temps raisonnable (milliards d'opérations). Proxy testé : KMeans sur le
profil médian 48-dim (une valeur par demi-heure) plutôt que sur la courbe
brute z-scorée de 1057 points. Résultat écran 3×10 : **-4,99 W, identique à
la référence (-4,98 W) au bruit près**. Neutre — ni gain ni perte, pas
retenu (complexité ajoutée pour rien). Confirme que la méthode de
regroupement par forme n'est pas un levier ici (cohérent avec l'insensibilité
déjà observée à `k`, §11-12).

**Blend alpha appris (OOF), reprise à grande échelle** (`tabpfn_oof_alpha.py`) :
le rejet du §14 reposait sur seulement 60 colonnes d'entraînement — repris
avec 300 colonnes (5×) et un second méta-modèle non linéaire
(HistGradientBoosting) en plus du Ridge, testés sur 100 colonnes fraîches
(vs 30) :

| Méthode | MAE (100 colonnes test) |
|---|---:|
| interpolation seule | 92,05 |
| TabPFN seul (sans blend) | 64,91 |
| **blend V14 (formule à la main)** | **64,55** |
| blend appris (Ridge OOF, 300 col.) | 68,00 |
| blend appris (GBM OOF, non linéaire) | 69,96 |

**Rejeté, cette fois de façon robuste** (budget 5× plus large, protocole de
test 3× plus large). Le GBM fait même pire que le Ridge malgré sa capacité
à représenter des paliers, signe de surapprentissage sur les 4
méta-features simples plutôt qu'un vrai gain de représentation. La formule
V14 (longueur de trou, distance au bord, volatilité locale, accord
périodique) encode apparemment déjà une connaissance du domaine qu'un
méta-modèle générique sur un jeu de features aussi restreint ne parvient
pas à redécouvrir. Cette piste est maintenant considérée close sauf
réengineering plus profond des méta-features (ex. contexte de courbe
entière, pas seulement point par point).

## 15quater. Catalogue de 50 features candidates + 4 testées (`tabpfn_v14_ablation.py`)

Recherche externe (tsfresh/catch22, PMM, seasonal-KNN, copy-paste imputation)
pour inspirer un catalogue de 50 features dérivées des courbes donneuses
(statistiques du pool par position, statistiques globales de courbe,
composition/diversité du pool, features métier électricité, incertitude) --
listé en détail dans la conversation, non reproduit ici par souci de
concision. 4 candidats jugés les plus susceptibles d'apporter une info
réellement nouvelle (pas déjà capturée par le calendrier/les voisins
existants) ont été testés sur l'écran 3×10 :

| Idée | Δ vs V12 (écran) | Verdict |
|---|---:|---|
| référence V14 | -4,98 W | — |
| MAD inter-voisins (pendant du std déjà testé) | -4,05 W | rejeté |
| largeur intervalle 10-90e percentile inter-voisins | -4,75 W | rejeté |
| flag heures pleines/creuses (tarif EDF approx.) | -4,35 W | rejeté |
| indice de consommation collective (médiane de 3000 courbes) | -4,43 W | rejeté |

**Tous rejetés.** Avec ce round, ce sont maintenant **10 idées de feature
engineering testées sur 3 rounds complets (§15bis day-type, §15ter profil,
ce round) sans un seul gain** -- signal statistique clair que ce pipeline a
atteint un plafond pour l'ajout de features au régresseur. Cohérent avec le
diagnostic du §11/§13 : le levier qui a fonctionné (V14, -4,53%) est un
réglage du blend alpha, pas une feature supplémentaire pour TabPFN.
**Recommandation : ne plus chercher de nouvelles features pour ce pipeline
sans un changement structurel plus profond** (ex. sélection de voisins par
sous-segment du trou plutôt que par courbe entière, cf. §10 point 2 --
jamais testé, contrairement à toutes les variantes de features globales
listées ci-dessus).

## 15. Bilan de la recherche de performance V14/OOF

37 configurations testées au total (20 idées indépendantes + 16 combinaisons
ciblées + 1 blend appris), toutes sur le pipeline TabPFN réel (API
`api.priorlabs.ai`). Résultat retenu : **V14 à 63,24 ± 30,46 W, soit -4,53 %
vs V12 (66,24 W), confirmé 9 repeats sur 10 sur le protocole robuste 10×10.**
C'est le plus gros gain validé depuis V12 (§4), et le plus robuste
statistiquement de toute la table (comparer aux 6/10 de V13, rejeté).

L'objectif de -15 % n'est pas atteint : à titre de comparaison, tout le
chemin V8→V12 (§4), qui a représenté des dizaines d'itérations documentées,
n'a gagné que 14,9 %. Les leviers explorés ici (réglage fin du poids
alpha, élargissement du pool de voisins, blend appris simple) sont
désormais tous testés et le premier est en rendement décroissant clair
(rounds 4-5 du §13 : gains marginaux voire nuls). Les leviers restants les
plus prometteurs, non testés faute de budget/temps dans cette session,
restent ceux du §10 : sélection de voisins par sous-segment du trou plutôt
que par courbe entière, et un vrai stacking OOF à plus grande échelle
(centaines de colonnes, modèle non linéaire) plutôt que le Ridge simple du
§14.

GPU CatBoost confirmé ~2× plus rapide que CPU par fit (2,6 s vs 5,1 s à
300 features) — utile pour itérer plus vite, sans effet sur la MAE finale.

## 12. Même expérience rejouée avec le vrai TabPFN (`tabpfn_finetune_bench.py`)

Une fois l'accès réseau/API disponible (le domaine correct est
`api.priorlabs.ai`, pas celui testé initialement), l'expérience du §11 a été
rejouée avec TabPFN plutôt que CatBoost, sur le protocole exact
`10×10` de la table V12 (mêmes graines, mêmes colonnes).

Écran `3×10` : k=40 (référence) 79,24 ± 31,09 W, k=100 79,68 ± 29,91 W
(reproduit très exactement le rejet déjà documenté au §7 : 79,68 vs
79,24 W), k=300 79,45 ± 28,68 W — les trois se valent à ce budget.

Confirmation `10×10` complète pour k=300 (voisins sélectionnés par seuil de
corrélation, cap large plutôt que 40 fixes) :

| Config | MAE | std |
|---|---:|---:|
| k=40 (V12, référence) | 66,24 | 32,45 |
| k=300 (toutes features pertinentes) | **67,00** | 31,38 |

**k=300 rejeté** (+0,76 W, +1,15 %) : 6 repeats sur 10 en faveur de k=40, et
surtout un repeat 10 nettement dégradé (+7,02 W) qui tire la moyenne au-dessus
de la référence — même schéma que le rejet de V13 (§9). Élargir le pool de
voisins ne paie ni pour CatBoost (§11) ni pour TabPFN : la conclusion tient
indépendamment du régresseur, ce qui renforce l'hypothèse que le signal utile
est déjà capté par les ~40 meilleurs voisins et que le blend alpha (§11)
plafonne de toute façon ce qu'un régresseur plus riche en features pourrait
apporter. **k=40 reste la configuration retenue.**

## 16. Round 2026-08-15 : 3 nouvelles hypothèses + réplication du criblage "colonnes empilées" sur V14 réel

Nouveau script `tabpfn_v16_hypotheses_bench.py`, qui réutilise
`tabpfn_v14_ablation.run_config` tel quel (le pipeline V14 réel, pas un
proxy), un seul repeat (seed=42). Reproduction vérifiée bit-à-bit avec
`tabpfn_v14_confirm_results.json` (110,80297585167031 des deux côtés) donc
comparaison fiable, mais **un seul repeat reste bruité par construction**
(cf. variance inter-repeats 30-134 W de la table V12/V14 10×10, §9/§13) --
aucune des conclusions ci-dessous n'est confirmée au niveau d'exigence
habituel de ce repo (10×10 apparié) sans un tour supplémentaire.

### 3 hypothèses nouvelles (jamais testées avant, cf. §10 pt.2/3/5), implémentées dans `tabpfn_v15_features.py`

| Hypothèse | Type | MAE | Δ |
|---|---|---:|---:|
| `local_reselect` (voisins resélectionnés localement autour du trou) | colonnes×lignes | 110,753 | -0,050 |
| `diverse_donor_consensus` (consensus sur voisins diversifiés) | colonnes | 110,081 | **-0,722** |
| `shape_match_block` (copie de bloc par alignement libre des 2 bords) | lignes | 110,803 | 0,000 (bug, voir note) |
| les 3 combinées | -- | 110,239 | -0,564 |

Note `shape_match_block` : bug identifié -- la feature ne vaut jamais rien
qu'aux positions manquantes, donc elle est **constante à 0 sur 100 % des
points d'entraînement** (positions observées), structurellement inapprenable
par un régresseur. Le score identique au bit près au baseline en est la
preuve. À corriger (calcul en leave-one-out aux points observés aussi) avant
tout verdict.

### Réplication des "meilleures features" trouvées par un criblage parallèle (architecture différente)

En parallèle, `tabpfn_column_feature_screen.py` (architecture "colonnes
empilées", cf. plus bas) avait identifié par recherche greedy sur 40
familles un combo gagnant (`ewma_context+boundary_values+neighbor_dynamics`,
-1,28 W sur 3 seeds, 3/3 victoires, cf.
`tabpfn_column_feature_screen_results.md`). Rejoué ici sur le pipeline V14
réel pour vérifier si ça transfère :

| Feature/combo | MAE | Δ vs V14 baseline |
|---|---:|---:|
| `new::ewma_context` (seule) | 128,788 | **+17,985** |
| `boundary_values` (seule) | 137,537 | **+26,734** |
| `gap_geometry` (seule) | 115,411 | +4,608 |
| `new::load_period_ratios` (seule) | 110,803 | 0,000 (constante par courbe, voir note) |
| `neighbor_dynamics` (seule) | 114,052 | +3,249 |
| `screen_best_combo` (les 3 gagnantes du criblage) | 139,265 | **+28,462** |
| `kitchen_sink` (round 1 utile + round 2, 7 features) | 141,233 | **+30,430** |

**Aucune ne transfère -- toutes dégradent, certaines fortement.** Deux
explications distinctes :
1. `load_period_ratios` (et par extension `curve_load_shape`,
   `curve_ramp_signature`, `curve_autocorrelation`,
   `curve_spectral_signature` du même catalogue,
   `boosted_columns_feature_search.py`) sont des résumés **constants par
   courbe** (`np.repeat(...,n,0)`) : utiles dans l'architecture "colonnes
   empilées" où plusieurs courbes différentes partagent un même jeu
   d'entraînement (la constante y varie d'une courbe à l'autre -- signal
   réel), mais **structurellement inertes** dans le fit par-courbe de V14
   (la valeur ne varie jamais dans le jeu d'entraînement d'une régression
   mono-courbe) -- pas un échec, une incompatibilité d'architecture.
2. `ewma_context`, `boundary_values`, `gap_geometry`, `neighbor_dynamics`
   varient bien dans le temps et auraient dû pouvoir aider, mais dégradent
   fortement (+3 à +27 W) une fois ajoutées au jeu de features déjà riche de
   V14 -- probablement redondance/colinéarité avec des features déjà
   présentes (`periodic_copy_features`, edge/run-length, `neighbor_rolling`)
   qui dilue le signal pour TabPFN, qui fit sur peu de lignes (~900 points
   observés d'UNE SEULE courbe, pas 8000 lignes poolées comme l'autre
   architecture).

**Conclusion : une feature gagnante sur une architecture ne se transfère pas
automatiquement à une autre -- chaque candidat doit être revalidé sur le
pipeline cible réel, pas seulement sur un proxy ou une autre architecture.**

## 18. Round 2026-08-17 : ablation par SUPPRESSION (`tabpfn_v18_ablation_removal_bench.py`)

Direction inverse des rounds 16-17 (qui ajoutaient des features, 0 gain) :
retirer/atténuer un par un les composants déjà présents dans V14
(`extra_lags`, `neighbor_rolling`, `volatility_alpha`,
`periodic_alpha_strength`), plus quelques contrôles négatifs déjà connus
insensibles (`corr_threshold`, `k`, `n_clusters`, `euclid_weight`). Repeat 1
(seed=42), nval=10.

**Sanity check du harnais** : `revert_to_v12_recipe` (toutes les additions
V12→V14 retirées) donne 122,386 W -- **identique au bit près** au score V12
repeat 1 déjà documenté (§9). Le harnais est fiable.

| Composant retiré | MAE | Δ |
|---|---:|---:|
| `volatility_alpha` | 113,81 | +3,01 |
| `periodic_alpha_strength` (0) | 114,12 | +3,31 |
| `extra_lags=(24,)` | 113,18 | +2,37 |
| `neighbor_rolling` | 112,93 | +2,13 |
| `periodic_alpha_strength` (→0,5 défaut) | 112,11 | +1,31 |
| `vol_weight` (→0,5 défaut) | 111,71 | +0,91 |

**Toutes les suppressions dégradent, sans exception** -- V14 ne contient
aucun composant superflu à ce niveau de granularité ; l'hypothèse "certaines
features déjà en place diluent le signal" (motivant ce round) est
**infirmée**. Les contrôles négatifs (`corr_threshold`, `k=20/60`,
`n_clusters`, `euclid_weight`) confirment aussi les rejets déjà documentés
en §7/§12, cohérent avec un protocole repeat-1 qui ne bruite pas tout.

Deux retouches de poids continus jamais testées (pas des retraits, des
recalibrages) ont montré un petit signal positif en repeat 1, plafonnant
autour de -0,4 W quelle que soit l'agressivité (`vol_cap` 0,6→0,4→0,3,
`alpha_scale` 1,0→0,8→0,7) :

| Config | repeat 1 (nval=10) | Δ |
|---|---:|---:|
| `vol_cap_low` (0,6→0,4) | 110,39 | -0,42 |
| `alpha_scale`+`vol_cap` combinés | 110,38 | -0,43 |

**Confirmation 10×10 (mêmes seeds que la référence V14 enregistrée) :**

| Config | MAE | Δ vs V14 | Victoires |
|---|---:|---:|---:|
| V14 (référence) | 63,235 ± 30,463 | -- | -- |
| `vol_cap_low` | 63,238 ± 30,356 | +0,003 | 3/10 |
| `alpha_scale`+`vol_cap` combinés | 63,323 ± 30,358 | +0,088 | 3/10 |

**Le gain s'évapore complètement à la confirmation** (3/10 victoires, pire
qu'un tirage à pile ou face) -- troisième occurrence documentée de ce même
piège (V13 §9, combo V15 §15quater) : un delta positif sur un seul repeat ne
prédit rien de fiable, même quand il provient d'un recalibrage sobre plutôt
que d'une nouvelle feature.

**Bilan** : avec ce round, **24 idées testées sur le pipeline V14 réel
(suppression + ajout, rounds 16-18), 0 gain confirmé sur 10×10.** V14 reste
la pipeline de référence, inchangée. Les leviers structurels du §"lacunes"
(donnée météo externe, blend alpha OOF à pleine échelle sur les 1000
colonnes, ensembling TabPFN+GBM avec OOF propre) restent les seules pistes
non épuisées identifiées à ce jour.

## 19. Round 2026-08-17bis : ensembling TabPFN+LightGBM (`tabpfn_gbm_ensemble_bench.py`)

Blend pondéré des sorties complètes (déjà mélangées à l'interpolation) de
deux pipelines V14 identiques en features, seul le régresseur change
(TabPFN vs LightGBM MAE + itératif 2 passes, cf. A4c §7bis). Balayage de
poids sur repeat 1 (seed=42, nval=10) :

| Poids TabPFN | MAE |
|---:|---:|
| 0,0 (LightGBM seul) | 129,50 |
| 0,7 | 111,75 |
| 0,8 | 111,02 |
| **0,9** | **110,67** |
| 1,0 (TabPFN seul = V14) | 110,80 |

Seul w=0,9 bat légèrement le TabPFN pur (−0,14 W) ; tous les autres poids
dégradent. **Confirmation 10×10 (mêmes seeds que la référence V14
enregistrée) :**

| Config | MAE | Δ vs V14 | Victoires |
|---|---:|---:|---:|
| V14 seul (référence) | 63,235 ± 30,463 | -- | -- |
| ensemble w=0,9 | 63,311 ± 30,385 | +0,076 | 4/10 |

**Le gain s'évapore de nouveau** (4/10 victoires) -- 4e occurrence du même
piège repeat-1 dans ce repo (V13, combo V15, `vol_cap`/`alpha_scale` §18, et
maintenant l'ensembling). Contrairement à l'hypothèse de départ (un second
régresseur diversifié devrait aider par décorrélation des erreurs), LightGBM
seul (129,50 W) est trop loin derrière TabPFN (110,80 W) sur ce pipeline
pour qu'un blend même minoritaire n'apporte un vrai gain de diversité --
cohérent avec l'écart déjà documenté en §5 (meilleur LightGBM : 68,89 W vs
TabPFN 63,24 W sur `5×100`/`10×10` respectifs, donc TabPFN domine largement,
pas seulement légèrement).

## 20. Round 2026-08-17ter : blend alpha OOF à pleine échelle (`tabpfn_oof_alpha_fullscale.py`)

Reprise du blend alpha appris (§14/§15ter, rejeté à 60 puis 300 colonnes
d'entraînement, "budget jugé insuffisant") à une échelle nettement plus
grande (700 colonnes d'entraînement / 150 test / 150 helper, contre 300/100
précédemment). Résultats en cours -- voir `tabpfn_oof_alpha_fullscale_results.json`
une fois le run terminé.

### Confirmation indépendante : l'architecture "colonnes empilées" perd face à V14

`tabpfn_column_individual_results.json` compare déjà cette architecture
"colonnes empilées" (`tabpfn_column_individual_bench.py`) à la vraie
référence V14, sur les mêmes 5 seeds (42-46) :

| seed | colonnes empilées | V14 (référence réelle) | Δ |
|---:|---:|---:|---:|
| 42 | 135,11 | 110,80 | +24,30 |
| 43 | 69,90 | 63,63 | +6,27 |
| 44 | 55,05 | 48,34 | +6,70 |
| 45 | 34,29 | 30,20 | +4,09 |
| 46 | 46,49 | 38,09 | +8,40 |
| **moyenne** | **68,17** | **58,21** | **+9,96, 0/5** |

Confirme (déjà documenté dans `TABFM_LORA.md`) que l'architecture "colonnes
empilées" perd systématiquement face à V14, sur toutes les échelles testées
jusqu'ici. Le `baseline_mean_w=53,56 W` apparemment meilleur que V14
(63,24 W) vu dans le criblage de features de cette même architecture
(section précédente) est donc un **artefact du sous-ensemble de seeds
testé** (44/43/45, parmi les repeats les plus faciles de la table 10×10),
pas une supériorité réelle de l'architecture.

### Verdict global de ce round

**Aucune nouvelle feature validée.** Seule `diverse_donor_consensus` montre
un signal dans le bon sens (-0,722 W, un seul repeat) qui mérite une
confirmation multi-repeats avant conclusion ferme ; toutes les autres
(locales ou empruntées à l'autre architecture) sont neutres, cassées, ou
nettement négatives. **V14 (63,24 ± 30,46 W) reste la pipeline de
référence, inchangée par ce round.**

## 17. Round 2026-08-16 : 2 nouvelles idées lignes×colonnes (round 3)

Deux idées supplémentaires, choisies pour être structurellement différentes
de tout ce qui précède (cf. `liste.md`) : un résidu périodique collectif
(pas juste un niveau collectif, déjà rejeté §15quater) et un consensus
donneurs avec micro-décalage temporel (±1h) recherché par courbe. Même
protocole (repeat 1, seed=42, pipeline V14 réel).

| Feature/combo | MAE | Δ vs baseline (110,80 W) |
|---|---:|---:|
| `donor_residual_synchrony` (médiane des résidus périodiques des donneurs) | 111,24 | +0,44 |
| `micro_shift_consensus` (consensus donneurs, décalage ±2 pas optimisé) | 113,06 | +2,26 (1 repli KNN) |
| les deux combinées | 112,50 | +1,70 |
| `diverse_donor_consensus` + les deux nouvelles | 110,82 | **+0,02** |

**Aucune des deux n'aide.** Plus révélateur : combinées avec
`diverse_donor_consensus` (seul candidat positif du round précédent, -0,722 W
seule), elles **annulent complètement son bénéfice** (+0,018 W, quasiment
zéro) -- même schéma de dilution que `kitchen_sink` au round 16 : ajouter des
dimensions même individuellement inoffensives (+0,44 W) dégrade le signal
utile d'une feature déjà positive, cohérent avec l'hypothèse d'un TabPFN
per-colonne qui dispose de peu de lignes d'entraînement (~900 points d'une
seule courbe) et se dilue vite avec plus de colonnes de features.

### Bilan cumulé des 3 rounds (16+17) : 12 idées de features testées, 0 gain confirmé

Sur le pipeline V14 réel, en repeat unique : 1 feature avec un signal
positif isolé et non confirmé (`diverse_donor_consensus`, -0,722 W), 11 avec
un effet nul ou négatif -- certaines fortement négatives (jusqu'à +30 W en
combinant les features empruntées à l'autre architecture). Complète le
constat déjà tiré en §15quater (10 idées, 0 gain, sur LightGBM/TabPFN) :
avec ce round, **22 idées de feature engineering testées sur ce pipeline
mature, 0 gain confirmé au niveau d'exigence 10×10 du repo.** Le signal
statistique est maintenant sans ambiguïté : chercher de nouvelles features
pour ce régresseur n'est plus une piste productive (cf. §18 pour les
leviers structurels alternatifs identifiés).

## 21. Round 2026-08-17quater : réimplémentation indépendante (`tabpfn_zero_start.py`) + concaténation dans V14 (`tabpfn_v14_concat72.py`)

Deux scripts distincts, construits pour tester une hypothèse différente du
criblage de features habituel : est-ce qu'une réimplémentation *totalement
indépendante* du pipeline (autre code, autre échelle robuste médiane/MAD,
sélection de voisins par corrélation Pearson globale sans clustering, pas de
dépendance à `feature_engineering_bench.py`) produit des features assez
décorrélées de celles de V14 pour l'améliorer une fois concaténées ?

### 21.1 `tabpfn_zero_start.py` seul : validation indépendante de la recette

Pipeline volontairement minimal et autonome (72 features : 8 calendaire +
3×8 copies périodiques ±48/96/144/336 + résumé + 40 voisins + résumé),
sortie TabPFN médiane directe, blend interpolation/modèle basé sur la seule
distance au bord + accord périodique (pas de volatilité, pas de clustering
de forme). Évalué en 10×10 réel sur son propre tirage de colonnes (graines
42-51, mais logique de sous-échantillonnage différente de V14 -- **non
apparié** avec la référence V14) :

`tabpfn_zero_start_full_full.json` → **63,93 ± 17,20 W** (10 repeats,
scores 40,25 à 101,25 W). Écart-type nettement plus faible qu'en V14
(17,20 vs 30,46 W), mais confondu avec le tirage de colonnes différent --
pas une comparaison de variance valide en soi.

**Constat principal : une implémentation ~5x plus simple (pas de clustering
de forme, pas d'alpha adaptatif à la volatilité, code totalement
indépendant écrit en quelques heures) atterrit à une moyenne quasiment
identique à V14** (63,93 vs 63,24 W, non apparié). Confirme que la
performance de ce dépôt vient essentiellement de la recette générale
(TabPFN + copies périodiques + voisins corrélés + blend adaptatif au bord),
pas d'un réglage fragile propre à V14 -- cohérent avec le bilan de §17/§18
("aucune nouvelle feature n'aide sur un pipeline déjà mature").

### 21.2 `tabpfn_v14_concat72.py` : les 72 features zero-start concaténées à V14 (protocole apparié)

Garde tout le pipeline V14 (clustering de forme, k voisins pondérés
corrélation+euclidien, alpha adaptatif volatilité/continuité/accord
périodique, blend final) **inchangé**, et ajoute les 72 features
indépendantes de §21.1 à la matrice passée à TabPFN (147 colonnes au total
en général -- voir bug ci-dessous). Même protocole apparié 10×10 que la
référence (`tabpfn_v14_confirm_results.json`, graines 42-51, mêmes
colonnes) :

| repeat | seed | V14 seul (réf) | V14 + 72 zero-start | Δ |
|---:|---:|---:|---:|---:|
| 1 | 42 | 110,80 | 107,66 | -3,14 |
| 2 | 43 | 63,63 | 63,69 | +0,06 |
| 3 | 44 | 48,34 | 48,94 | +0,60 |
| 4 | 45 | 30,20 | 30,72 | +0,52 |
| 5 | 46 | 38,09 | 37,70 | -0,39 |
| 6 | 47 | 47,46 | 47,84 | +0,38 |
| 7 | 48 | 55,74 | 52,85 | -2,89 |
| 8 | 49 | 47,13 | 47,55 | +0,42 |
| 9 | 50 | 130,32 | 128,23 | -2,09 |
| 10 | 51 | 60,63 | 57,35 | -3,28 |
| **moyenne** | | **63,24 ± 30,46** | **62,25 ± 29,54** | **-0,98** |

5/10 victoires, delta moyen -0,98 W, **delta médian -0,16 W** (proche de
zéro -- le gain moyen est tiré par 4 repeats à forte amélioration, dont les
deux repeats les plus difficiles du lot, 1 et 9), IC bootstrap 95% sur le
delta moyen = **[-1,99 ; -0,03]** (entièrement négatif mais effleure zéro).

**Verdict : signal positif, réel mais modeste, premier de ce niveau après
22 idées rejetées en §16-18.** Ne franchit pas la barre de confirmation
habituelle du dépôt (V14 vs V12 avait été confirmé à 9/10 victoires, cf.
`generate_submission_tabpfn_v14.py`) -- ici seulement 5/10, et la médiane
quasi nulle indique que la moitié des colonnes ne bénéficient pas de ces
features (probablement les colonnes "faciles" où V14 seul sature déjà). Le
gain semble concentré sur les colonnes/repeats difficiles (grosses lacunes,
MAE de référence élevée), hypothèse cohérente avec l'idée que des features
de rang/échelle indépendantes aident surtout quand le clustering de forme
de V14 est moins fiable. **Signal à surveiller pour un futur round, pas
encore un candidat de remplacement de V14 en l'état** (delta trop proche de
zéro en médiane, écart-type des deltas important).

### 21.3 Bug trouvé et corrigé en cours de route

`tabpfn_v14_concat72.py` contenait une assertion `design.shape[1] != 147 →
raise` calibrée sur un seul run de fumée (colonne `holed_68`, 40 voisins
sélectionnés). Le nombre de voisins de V14 est en réalité adaptatif (5 à 40,
`corr_threshold=0.15`, cf. `adaptive_top_k_corr_euclid_tunable`) : la
largeur concaténée varie donc légitimement d'une colonne à l'autre (121 à
147 features observées), et le premier run réel a crashé dès qu'une colonne
avec moins de 40 voisins retenus est apparue (repeat 1 seul, sur un lot de
10 colonnes, n'en contenait aucune -- d'où le smoke test initial "vert" qui
ne l'a pas détecté). Corrigé pour ne vérifier que les invariants réels
(largeur zero-start = 72, largeur totale = v14 + 72) ; les diagnostics
par colonne (`v14_features`, `zero_features`, `n_features`) restent
disponibles dans `tabpfn_v14_concat72_results.json` pour audit.

## 22. JEPA (world model façon Yann LeCun) — testé sur demande, rejeté (`jepa_bench.py`)

Demande explicite de tester une architecture JEPA (Joint-Embedding Predictive
Architecture, Balestriero & LeCun 2025 — LeJEPA) : encodeur contexte +
encodeur cible EMA (stop-gradient) + prédicteur en espace latent +
régularisation variance/covariance anti-collapse (VICReg), en PyTorch,
malgré l'avis a priori défavorable donné avant implémentation (§7ter : la
même famille "modèle global auto-supervisé" avait déjà plafonné à 74,66 W
avec un LSTM/Transformer).

**Implémentation** (`jepa_bench.py`, deux phases indépendantes) :
- Pré-entraînement auto-supervisé sur un pool de 4 000 courbes donneuses
  (jamais `y_train`), trous synthétiques tirés de la distribution empirique
  réelle (`sample_mask`, réutilisé de `lstm_bench.py`) ; tokens par instant
  (pas par patch, pour coller aux trous réels irréguliers) ; early stopping
  sur un split temporel (repli au step 200, val loss JEPA = 0,2126).
- Pour donner une vraie chance au modèle (pas juste rejouer le LSTM), chaque
  instant reçoit un canal de **consensus voisins** (k=40, même logique que
  `evaluate.make_knn_model` / la feature set V12-V14), en plus du calendrier
  — fourni au contexte ET aux requêtes du prédicteur.
- Fine-tuning léger (tête de régression uniquement, prédicteur/encodeur
  gelés) sur les vrais trous des colonnes dev à chaque repeat, blend avec
  l'interpolation via `continuous_alpha` (même formule que le reste du
  dépôt).

**Résultat**, protocole `10×10` (mêmes graines de split `random_state=42+repeat`
que le reste du dépôt ; ⚠️ sélection des 10 colonnes de val via
`np.random.RandomState` alors que les scripts TabPFN/LightGBM utilisent
`np.random.default_rng` — mêmes graines mais pas les mêmes colonnes tirées,
donc comparaison de **moyennes**, pas un delta apparié repeat-par-repeat) :

| Repeat | MAE |
|---:|---:|
| 1 | 137,10 |
| 2 | 71,03 |
| 3 | 56,87 |
| 4 | 37,25 |
| 5 | 47,07 |
| 6 | 61,18 |
| 7 | 69,23 |
| 8 | 54,72 |
| 9 | 149,06 |
| 10 | 69,30 |
| **moyenne** | **75,28 ± 35,44** |

| Modèle | MAE (W) |
|---|---:|
| V0 interpolation linéaire | 97,17 |
| **JEPA (ce round)** | **75,28** |
| LSTM/Transformer seul | 76,10 |
| LSTM/Transformer + blend alpha | 74,66 |
| R0 LightGBM (features V12, voisins) | 69,75 |
| TabPFN V12 | 66,24 |
| **TabPFN V14 (actuel)** | **63,24** |

**Verdict : confirme exactement le pronostic du §7ter, rejeté.** Le JEPA
atterrit pile dans la fourchette de la famille "modèle global" (74-77 W),
entre le LSTM seul et le LSTM+blend — le canal de consensus voisins (feature
V12/V14) l'empêche d'être pire que le LSTM sans lui, mais ne suffit pas à
combler l'écart avec R0/V12/V14. Confirme le diagnostic répété du dépôt :
la variable qui compte n'est pas l'architecture de prédiction (reconstruction
brute vs. espace latent façon JEPA) ni même les features utilisées, mais
l'**adaptation par colonne cible** au moment de l'inférence (TabPFN en
contexte, KNN par corrélation) — qu'un modèle à poids partagés, aussi
sophistiqué soit son objectif auto-supervisé, n'a par construction pas.
**Aucun changement à la soumission** : `submission_tabpfn_v14.csv` reste la
référence. Non testé faute d'intérêt après ce résultat : `--finetune-predictor`
(dégeler le prédicteur), un pool de pré-entraînement plus large.
