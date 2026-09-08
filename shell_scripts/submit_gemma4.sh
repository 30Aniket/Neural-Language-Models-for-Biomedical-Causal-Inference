#!/usr/bin/env bash
# Queue the 4 Gemma4 r8/paper jobs: {gemma4_12b, gemma4_26b} x {AILF, TRAM}, one GPU each.
#   bash submit_gemma4.sh
set -euo pipefail
MODELS=("${MODELS[@]:-gemma4_12b gemma4_26b}")
CONFIGS=("${CONFIGS[@]:-paper}")           # r8 only
DATASETS=(Analgesics-induced_acute_liver_failure Tramadol-related_mortalities)
mkdir -p logs; n=0
for M in ${MODELS[@]}; do
  for C in ${CONFIGS[@]}; do
    for D in "${DATASETS[@]}"; do
      DTAG=$([[ "$D" == Analg* ]] && echo ailf || echo tram)
      jid=$(sbatch --parsable slurm/gemma4_peft.slurm "$D" "$C" "$M")
      echo "queued ${M}/${C}/${DTAG} -> job ${jid}"; n=$((n+1))
    done
  done
done
echo "submitted ${n} jobs."
