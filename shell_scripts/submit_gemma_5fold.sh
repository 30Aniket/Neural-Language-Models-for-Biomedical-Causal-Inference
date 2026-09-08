#!/usr/bin/env bash
# Queue all 12 Gemma/MedGemma jobs: 2 models x 3 configs x 2 datasets, one GPU each.
#   bash submit_gemma_5fold.sh
# Priority-first variant: MODELS/CONFIGS can be narrowed, e.g. CONFIGS=(paper) for the
# 4 clean-comparison jobs only.  HF_TOKEN must be exported (gated models).
set -euo pipefail
MODELS=("${MODELS[@]:-gemma3 medgemma}")
CONFIGS=("${CONFIGS[@]:-paper r64 r128}")
DATASETS=(Analgesics-induced_acute_liver_failure Tramadol-related_mortalities)
mkdir -p logs
n=0
for M in ${MODELS[@]}; do
  for C in ${CONFIGS[@]}; do
    for D in "${DATASETS[@]}"; do
      DTAG=$([[ "$D" == Analg* ]] && echo ailf || echo tram)
      jid=$(sbatch --parsable slurm/gemma_peft.slurm "$D" "$C" "$M")
      echo "queued ${M}/${C}/${DTAG} -> job ${jid}"; n=$((n+1))
    done
  done
done
echo "submitted ${n} jobs."
