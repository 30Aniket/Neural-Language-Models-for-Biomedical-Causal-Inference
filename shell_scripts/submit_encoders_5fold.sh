#!/usr/bin/env bash
# ==========================================================================
# submit_encoders_5fold.sh -- fire all 8 encoder jobs (4 models x 2 datasets)
# at TRUE 5-fold CV. Each job runs all 5 folds internally on one GPU.
#
# RUN ON THE LOGIN NODE, from $SHARE/$USER/biocausal_cluster:
#     bash submit_encoders_5fold.sh                    # no email
#     bash submit_encoders_5fold.sh you@warwick.ac.uk  # END/FAIL alerts
#
# PREREQUISITE: pretrained checkpoints already cached (env/prefetch_models.sh),
# and venv_modernbert already built (env/setup_env_modernbert.sh) for ModernBERT.
# 8 jobs x 1 GPU = 8 GPUs, well within your 15-GPU cap -> all run at once.
# ==========================================================================
set -euo pipefail

EMAIL="${1:-}"
MAILOPT=()
[[ -n "${EMAIL}" ]] && MAILOPT=(--mail-user="${EMAIL}" --mail-type=END,FAIL)

DATASETS=(Analgesics-induced_acute_liver_failure Tramadol-related_mortalities)
# model -> its slurm script (ModernBERT uses its own venv via modernbert.slurm)
MODELS=(albert biobert pubmedbert modernbert)

echo "Submitting 8 encoder jobs (4 models x 2 datasets) at true 5-fold CV..."
for M in "${MODELS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    case "${DS}" in Analgesics*) DTAG=ailf ;; Tramadol*) DTAG=tram ;; *) DTAG="${DS}" ;; esac
    jid="$(sbatch --parsable --job-name="${M}_${DTAG}" "${MAILOPT[@]}" "slurm/${M}.slurm" "${DS}")"
    echo "  ${M}_${DTAG}  ->  job ${jid}"
  done
done

echo ""
echo "Monitor:  squeue -u \$USER"
echo ""
echo "Progress (each -> 5 folds):"
echo "  for m in ${MODELS[*]}; do for d in ${DATASETS[*]}; do \\"
echo "    echo -n \"\$m \$d: \"; ls dat/\$d/proc/cross_val_\${m}_temp/df_res*.csv 2>/dev/null | wc -l; done; done"
echo ""
echo "When all reach 5, evaluate:  python run_evaluation.py --root \"\$(pwd)\""
