# Catalogue des idées déjà testées

Inventaire consolidé, toutes sources confondues (`benchmark.md`,
`tabpfn_v15_feature_results.json`, `tabpfn_column_feature_screen_results.md`,
`boosted_columns_feature_search_results.md`, `boosted_columns_results.md`,
`tabpfn_v16_hypotheses_results.json`, `tabpfn_column_individual_results.json`,
`TABFM_LORA.md`). Sert à ne pas retester une idée déjà tranchée. Légende :
✅ retenu · ➖ neutre/non concluant · ❌ rejeté · 🐛 cassé (bug identifié,
jamais vraiment testé).

## 1. Features temporelles seules (axe lignes : une seule courbe dans le temps)

| Idée | Source | Verdict |
|---|---|---|
| Copies périodiques ±48/96/144/336 (+ ±24 en V14) | V12/V14, §6/§13 | ✅ pilier de V12→V14 |
| J−14/J+14 | §9 (V13 criblage) | ❌ +0,558 W |
| J−4/−5/−6 et symétriques | §9 | ❌ +1,981 W |
| Profil heure × jour de semaine | §9 | ❌ +5,780 W |
| Profil heure × ouvré/week-end | §9 | ❌ +0,709 W |
| Moyenne/MAD par demi-heure | §9 | ❌ +0,339 W |
| Pente/courbure du consensus (V13) | §9 | ❌ +0,14 W global (6/10 gagnés mais rejeté) |
| Harmoniques calendrier ordre 2-4 | §9 | ❌ +0,090 W |
| Profil Fourier quotidien propre | §9 | ❌ +0,550 W ; repris comme `fourier_calendar` (screen parallèle) | ❌ +0,315 W |
| Saisonnalité week-end/semaine séparée | §15bis | ❌ −4,39 W vs réf −4,98 W |
| Copies périodiques masquées si mauvais day-type | §15bis | ❌ |
| Flag correspondance day-type (additif) | §15bis | ❌ |
| Tendance locale fenêtre 72/144 pts | §15bis | ❌ |
| `gap_geometry` (durée trou, distance bord, log) | V15 catalogue | ➖ seule ; ❌ +4,608 W ajoutée à V14 (round 2026-08-15) |
| `boundary_values`/`boundary_slopes` | V15 catalogue | ➖ seules ; ❌ boundary_values +26,7 W ajoutée à V14 |
| `curvature`, `hermite_bridge` | V15 catalogue | ➖/❌ |
| `local_volatility`, `regime_shift` | V15 catalogue | ➖ |
| `daytype_profile`, `recent_profile` | V15 catalogue | ➖ |
| `adjusted_periodic`, `periodic_residuals` | V15 catalogue | ➖ (meilleur combo confirm 10×10 LightGBM : −0,017 W, bruit) |
| `short_lags`, `periodic_lags` (catalogue 25 "new::") | screen parallèle | ❌ |
| `rolling_location/quantiles/dispersion` | screen parallèle | ❌ (pires du lot) |
| `local_ramps`, `local_acceleration` | screen parallèle | ➖/❌ |
| `boundary_trends`, `polynomial_bridges`, `log_bridge` | screen parallèle | ➖ |
| `ewma_context` | screen parallèle | ➖ meilleure indiv. de ce criblage (−0,885 W) MAIS ❌ **+17,99 W** rejouée sur V14 réel (round 2026-08-15) |
| `weekday_slot_profile`, `daytype_slot_profile` | screen parallèle | ➖ |
| `recent_weighted_profile`, `homologous_day_trend` | screen parallèle | ➖ |
| `curve_load_shape/ramp_signature/autocorrelation/spectral_signature` | screen parallèle | ❌ **constantes par courbe** (np.repeat) → structurellement inertes en fit par-courbe (V14) ; utiles seulement en archi "colonnes empilées" |
| `load_period_ratios` | screen parallèle | 🐛 même défaut (constante par courbe) ; +0,000 W bit-à-bit sur V14 |
| `shape_match_block` (bloc historique par alignement libre 2 bords) | round 2026-08-15 (moi) | 🐛 constante à 0 sur 100% des points d'entraînement (jamais recalculée aux points observés) → inapprenable, +0,000 W bit-à-bit |
| PCHIP (remplacement) | §7 | ❌ 1808 W, explosions extrapolation |
| PCHIP (feature) | §7 | ❌ 108,49 W |
| Interpolation forcée trous ≤3 | §7 | ❌ 71,36 W |
| Lags directs avant/après | §7 | ❌ 107,36 W |
| Imputation itérative 2 passes (A4) | §7bis | ✅ sur LightGBM (68,89 W) ; ❌ ne transfère pas à TabPFN (+0,08 W, 3/10) |
| Imputation itérative 3 passes | §7bis | ❌ 2 passes suffisent |
| Alpha adaptatif (durée trou + distance bord + volatilité + accord périodique) | V12→V14 | ✅ le levier le plus productif du repo (§11/§13/§15) |
| Blend alpha appris (Ridge OOF 60 puis 300 colonnes) | §14/§15ter | ❌ rejeté, formule à la main gagne (budget d'entraînement jugé insuffisant même à 300 colonnes) |
| Blend alpha appris (GBM OOF non-linéaire) | §15ter | ❌ pire que Ridge (surapprentissage sur 4 méta-features) |

## 2. Features spatiales seules (axe colonnes : autres courbes donneuses)

| Idée | Source | Verdict |
|---|---|---|
| KNN top-k corrélation + euclidien (k=40) | V8→V12 | ✅ pilier |
| k=100 (TabPFN) | §7 | ❌ 79,68 vs 79,24 W |
| k=300 (TabPFN, seuil corrélation) | §12 | ❌ +0,76 W, 6/10 |
| k=500 (CatBoost GPU) | §11 | ❌ +... dans le bruit, 2x plus lent |
| k=10 000 (CatBoost) | §7 | ❌ porté par 1 seul repeat, 4/10 |
| Clustering de forme 15-25 clusters | V6→V12 | ✅ |
| Clustering par profil journalier (proxy 48-dim vs courbe brute) | §15ter | ➖ neutre (identique au bruit près) |
| Seuil de corrélation 0,85 | §7 | ❌ 73,71 W, trop strict |
| Agrégats de 500 voisins (moyenne/médiane) | §7bis (A1) | ❌ +0,05 W |
| Complétion bas-rang ACP r=20 | §7bis (A2) | ❌ +0,29 W |
| SoftImpute/NMF | §7 | ❌ 84,04 W |
| `neighbor_reliability`, `neighbor_distribution` | V15 catalogue | ❌ pires du lot V15 (+0,529/+1,339 W) |
| `neighbor_local_corr` | V15 catalogue | ❌ +0,441 W |
| `neighbor_dynamics` | V15 catalogue | ➖ (−0,017 W confirm 10×10 LightGBM, bruit) ; ❌ **+3,25 W** rejouée sur V14 réel |
| `neighbor_consensus`, `neighbor_rolling` | V12/V14 | ✅ retenu dans V14 (`neighbor_rolling`) |
| `donor_trimmed_consensus`, `donor_slot_reliability`, `donor_local_calibration`, `donor_low_rank` (SVD) | screen parallèle | ❌ les 4 pires du classement 40-familles |
| Médiane/MAD des donneurs | §9 | ➖ candidat isolé (−0,25 W) mais fait partie de V13, rejeté au global |
| Quantiles des donneurs | §9 | ❌ +0,269 W |
| Incertitude/désaccord donneurs | §9 | ❌ +0,246 W |
| Consensus glissant 3/7/13 pts | §9 | ❌ |
| Consensus donneurs pondéré | §9 | ❌ +0,559 W |
| Indice de consommation collective (médiane 3000 courbes) | §15quater | ❌ −4,43 W (vs réf −4,98 W) |
| Flag heures pleines/creuses (tarif EDF) | §15quater | ❌ |
| MAD inter-voisins | §15quater | ❌ |
| Largeur intervalle 10-90e percentile inter-voisins | §15quater | ❌ |
| `diverse_donor_consensus` (consensus sur sous-ensemble diversifié, greedy max-min) | round 2026-08-15 (moi) | ➖ seul signal positif du round (−0,72 W, **1 seul repeat, non confirmé**) |

## 3. Features combinées lignes × colonnes

| Idée | Source | Verdict |
|---|---|---|
| Sélection locale de voisins, fenêtre ±96 (A3) | §7bis | ❌ piège de mesure : fenêtre couvre 100% des points, aucun effet réel |
| Sélection locale ±12 (A3b) | §7bis | ➖ non concluant (4/10) |
| Sélection locale ±6 (A3c) | §7bis | ➖ marginal (6/10, −0,17 W) |
| `local_reselect` (resélection corrélation locale par trou, k=20, fenêtre ±96) | round 2026-08-15 (moi) | ➖ quasi nul (−0,05 W, 1 repeat) |
| `adjusted_periodic`, `periodic_residuals` (copie périodique ajustée au niveau local) | V15 catalogue | ➖ |
| `donor_slot_reliability` (fiabilité voisin conditionnée au créneau calendaire) | screen parallèle | ❌ +0,747 W |
| Itératif + agrégats (A4d) | §7bis | ❌ +0,10 W, les agrégats n'ajoutent rien |
| 1000 donneurs pseudo-masqués + modèle global | §7 | ❌ 77,20 W, gain marginal sur un modèle déjà non compétitif |
| `donor_residual_synchrony` (médiane résidus périodiques inter-donneurs) | round 2026-08-16 (moi) | ❌ +0,44 W |
| `micro_shift_consensus` (consensus donneurs, micro-décalage ±1h optimisé) | round 2026-08-16 (moi) | ❌ +2,26 W (1 repli KNN) |

## 4. Changements d'architecture / de régresseur

| Idée | Source | Verdict |
|---|---|---|
| Ridge local → LightGBM/XGBoost/CatBoost | §3 | ✅ progression V4→H1 |
| Ridge → TabPFN | §4 | ✅ −5,61 W (V8) |
| Sortie médiane explicite TabPFN | §4 | ✅ (G6, −0,74 W) |
| Mean/median blend TabPFN | tabpfn_v14_ablation (idée #15) | non concluant isolément, jamais confirmé gagnant |
| Ensemble de graines TabPFN (SeedEnsemble) | tabpfn_v14_ablation | non retenu dans V14 final |
| Stacking classique (LightGBM/CatBoost/ExtraTrees médiane) | §5/§7 | ❌ 79,39-86,96 W, surapprentissage/pas d'OOF |
| Modèle global (1 seul modèle partagé, vraies courbes) | §7/§7ter | ❌ 91,52 W seul, 77,44 W + périodicités : plafond structurel, ne capte pas l'info inter-courbes |
| LSTM/GRU/Transformer séquentiel (5 architectures, 15 configs) | §7ter | ❌ 74,66-76,10 W : bat l'interpolation mais reste 8-10 W derrière le pipeline à voisins |
| Fine-tuning LoRA de TabFM (poids réels, rétropropagation) | `TABFM_LORA.md`, `tabfm_lora_finetune.py` | en cours / résultats non synthétisés ici, cf. `tabfm_lora_results.json` |
| Architecture "colonnes empilées" (1 fit TabPFN sur 8000 lignes poolées de ~100 courbes) | `tabpfn_column_individual_bench.py` | ❌ **confirmé perdant face à V14 sur les mêmes 5 seeds** (68,17 vs 58,21 W, 0/5) |
| Arbres boostés (XGBoost) sur archi "colonnes empilées" | `boosted_columns_results.md` | XGBoost > LightGBM/CatBoost dans cette archi (82,1 W) ; features `dynamics+profiles` retenues (−1,656 W, 5/5) -- mais l'archi elle-même perd contre V14 |
| Imputation itérative sur TabPFN | §7bis | ❌ +0,08 W (3/10) : TabPFN gère déjà nativement les NaN périodiques |
| `k` élargi + hyperparamètres GPU CatBoost | §11 | ❌ −0,35 W dans le bruit, 2× plus lent |
| Blend alpha appris à grande échelle (300 col., Ridge + GBM) | §15ter | ❌ rejeté de façon robuste (protocole 3× plus large) |
| Clustering par profil (proxy DTW 48-dim) | §15ter | ➖ neutre |
| Réimplémentation indépendante 72 features (`tabpfn_zero_start.py`, sans clustering ni alpha volatilité) | round 2026-08-17 (moi), §21 | ➖ moyenne quasi identique à V14 (63,93 vs 63,24 W, non apparié) -- valide la recette générale, pas un remplaçant |
| Concaténation des 72 features zero-start dans la matrice V14 (`tabpfn_v14_concat72.py`) | round 2026-08-17 (moi), §21 | ➖ signal positif modeste non confirmé (5/10, médiane −0,16 W, IC bootstrap [−1,99;−0,03]) -- premier signal de ce niveau après 22 idées rejetées, à revoir avec plus de repeats |

## 4bis. Ablation par suppression de composants V14 (round 2026-08-17)

| Composant retiré/atténué | Verdict repeat 1 | Verdict 10×10 |
|---|---|---|
| `volatility_alpha`, `periodic_alpha_strength`, `extra_lags=(24,)`, `neighbor_rolling` (retirés un par un) | ❌ tous dégradent (+2 à +3,3 W) | non confirmé (inutile de tester, déjà pires en repeat 1) |
| `revert_to_v12_recipe` (tout retiré) | ❌ 122,39 W = score V12 repeat 1 documenté, bit-à-bit | -- (sert de sanity check du harnais) |
| `vol_cap` 0,6→0,4/0,3, `alpha_scale` 1,0→0,8/0,7 (recalibrage, pas suppression) | ➖ signal positif plafonnant à -0,4 W | ❌ **+0,003/+0,088 W, 3/10 victoires** -- s'évapore |

**Conclusion : aucun composant de V14 n'est superflu ; la recette est déjà
bien optimisée.** Troisième occurrence du piège "signal repeat-1 qui ne
survit pas à la confirmation 10×10" (après V13 §9 et le combo V15
§15quater) -- ne plus se fier à un delta measuré sur un seul repeat, même
petit et de direction plausible.

## 5. Bilan chiffré

- **~77 idées de features/architecture distinctes testées** à travers ce
  dépôt (comptage approximatif : 15 axes V13 + 15 groupes V15 + 25 familles
  "new::" + 5 hypothèses rounds 2026-08-15/16 + ~15 idées historiques §3-§8).
- Sur le seul pipeline V14 réel (rounds §16/§17/§18 de `benchmark.md`) : **24
  idées testées (ajout de features + suppression de composants + recalibrage
  de poids), 0 gain confirmé sur le protocole robuste 10×10.** Tout signal
  positif observé jusqu'ici ne survivait qu'au niveau d'un seul repeat.
- **Taux de succès très faible** : la quasi-totalité des features ajoutées
  au régresseur (TabPFN ou LightGBM) sur un pipeline déjà mature
  n'apportent rien ou dégradent — seuls des changements structurels ont
  historiquement payé : Ridge→arbres (§3), arbres→TabPFN (§4), et surtout le
  réglage du poids alpha interpolation/modèle (V12→V14, §11/§13).
- Deux causes récurrentes de faux positifs identifiées : (a) surapprentissage
  au screen (V13 6/10 gagnant puis rejeté, combo V15 −1,61 W en screen puis
  −0,017 W confirmé) ; (b) features qui ne varient pas dans le jeu
  d'entraînement d'une régression mono-courbe (constantes par courbe,
  cf. §16 de `benchmark.md`) — un piège structurel spécifique à l'archi
  "un fit par colonne" de V14, à vérifier systématiquement pour toute
  nouvelle feature avant de la tester.
