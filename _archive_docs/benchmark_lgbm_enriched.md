# LightGBM enrichi — Optuna 50 trials

## Configuration

- Modèle retenu après audit : LightGBM, objectif MAE.
- Features : pipeline V14, 15 groupes V15, 50 descripteurs nouveaux des
  distributions donneuses, imputation itérative en deux passes.
- HPO : 50 trials TPE, graine 42, trois folds temporels et early stopping.
- Confirmation : cinq splits externes de colonnes 90/10, graines 42 à 46 ;
  les 407 colonnes susceptibles d'être évaluées ont été exclues de l'HPO.

## Résultats

- Meilleure MAE de CV temporelle Optuna : **73,22 W**.
- Confirmation LightGBM enrichi : **84,82 ± 47,10 W**.
- Scores par graine : `174,45`, `87,67`, `60,44`, `42,51`, `59,02` W.
- TabPFN V14 apparié : **58,21 ± 28,57 W** ; LightGBM gagne `0/5` repeats.
- Le LightGBM rapide historique à deux passes demeure le meilleur arbre du
  dépôt (**68,89 ± 31,82 W** sur le protocole 10×10 historique).

## Décision

Le candidat enrichi est rejeté : l'ajout simultané de toutes les familles et
des 50 distributions sur-apprend malgré la CV temporelle et l'early stopping.
La génération de soumission est donc volontairement bloquée à ce stade. Le
générateur reprenable reste disponible pour une future configuration qui
franchirait le contrôle externe.

Artefacts : `gbm_optuna_study.db`, `gbm_hpo_datasets.npz`,
`gbm_best_params.json`, `gbm_optuna_vs_tabpfn_results.json`.
