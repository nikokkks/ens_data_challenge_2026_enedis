# Benchmark arbres boostés colonne-centré

## Comparaison des modèles (screening 3 × 20 courbes)

- lightgbm: 87.531 W
- catboost: 83.895 W
- xgboost: 82.115 W

Modèle retenu : **xgboost**.

## Ablations de features

- boundaries: 82.262 W (delta +0.147 W, 0/3 repeats gagnés)
- dynamics: 81.828 W (delta -0.287 W, 2/3 repeats gagnés)
- profiles: 81.918 W (delta -0.198 W, 2/3 repeats gagnés)
- donor_quality: 82.181 W (delta +0.065 W, 2/3 repeats gagnés)
- curve_mask_stats: 82.419 W (delta +0.303 W, 1/3 repeats gagnés)

Features retenues : **dynamics, profiles**.

## Confirmation (5 × 100 courbes)

- Baseline 96 features : 79.413 W
- Modèle enrichi : 77.757 W
- Gain : 1.656 W
- Repeats gagnés : 5/5
