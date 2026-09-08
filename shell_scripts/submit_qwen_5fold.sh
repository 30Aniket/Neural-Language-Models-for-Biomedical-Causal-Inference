#!/usr/bin/env bash
# Queue all 12 Qwen3.5-Base jobs: 2 models x 3 configs x 2 datasets, one GPU each, in parallel.
#   bash submit_qwen_5fold.sh
# Narrow with env overrides, e.g. CONFIGS=(paper) or MODELS=(qwen4b).
set -euo pipefail
MODELS=("${MODELS[@]:-qwen4b qwen9b}")
CONFIGS=("${CONFIGS[@]:-paper r64 r128}")
DATASETS=(Analgesics-induced_acute_liver_failure Tramadol-related_mortalities)
mkdir -p logs; n=0
for M in ${MODELS[@]}; do
  for C in ${CONFIGS[@]}; do
    for D in "${DATASETS[@]}"; do
      DTAG=$([[ "$D" == Analg* ]] && echo ailf || echo tram)
      jid=$(sbatch --parsable slurm/qwen_peft.slurm "$D" "$C" "$M")
      echo "queued ${M}/${C}/${DTAG} -> job ${jid}"; n=$((n+1))
    done
  done
done
echo "submitted ${n} jobs."
