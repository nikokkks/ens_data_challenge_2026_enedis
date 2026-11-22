#!/bin/bash
# Lance chaque config de xgboost_bench.py dans un process Python SEPARE (mémoire
# limitée sur cette machine -> évite d'accumuler la mémoire de 10 configs dans
# un seul process, et un crash ne fait perdre que la config en cours, pas tout
# le run). Saute les configs déjà présentes dans xgboost_bench_results.json.
cd /home/nicolas/PycharmProjects/hackathon_enedis
CONFIGS="X0_baseline T1_mae_objective T2_regularized T3_capacity T4_cluster_mean_feat T5_lag_feats T6_early_stop T7_k15 T8_k35 T9_ensemble_ridge"

for cfg in $CONFIGS; do
    already_done=$(.venv/bin/python -c "
import json, os
if os.path.exists('xgboost_bench_results.json'):
    d = json.load(open('xgboost_bench_results.json'))
    print('yes' if '$cfg' in d else 'no')
else:
    print('no')
" 2>/dev/null)
    if [ "$already_done" = "yes" ]; then
        echo "=== $cfg deja fait, on saute ===" >> xgboost_bench_output.log
        continue
    fi
    tries=0
    ok="no"
    while [ $tries -lt 3 ] && [ "$ok" = "no" ]; do
        tries=$((tries+1))
        echo "=== $cfg tentative $tries $(date) ===" >> xgboost_bench_output.log
        .venv/bin/python -u xgboost_bench.py "$cfg" >> xgboost_bench_output.log 2>&1
        exit_code=$?
        echo "=== $cfg tentative $tries terminee, exit=$exit_code $(date) ===" >> xgboost_bench_output.log
        ok=$(.venv/bin/python -c "
import json, os
if os.path.exists('xgboost_bench_results.json'):
    d = json.load(open('xgboost_bench_results.json'))
    print('yes' if '$cfg' in d else 'no')
else:
    print('no')
" 2>/dev/null)
        sleep 1
    done
done
echo "=== TOUTES LES CONFIGS TRAITEES $(date) ===" >> xgboost_bench_output.log
