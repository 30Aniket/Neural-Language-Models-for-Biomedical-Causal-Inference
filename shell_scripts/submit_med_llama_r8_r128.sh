#!/usr/bin/env bash
# ==========================================================================
# submit_med_llama_r8_r128.sh -- fire exactly 4 jobs:
#   Med-LLaMA-8B QLoRA, configs {paper (r8), r128} x {AILF, TRAM}.
# Each job runs all 20 folds on one GPU. Job names: mlm_<rank>_<ailf|tram>.
# Speed-tuned: SDPA attention + batch 16 (effective batch 32 unchanged).
#
# RUN ON THE LOGIN NODE, from $SHARE/$USER/biocausal_cluster:
#     bash submit_med_llama_r8_r128.sh                    # no email
#     bash submit_med_llama_r8_r128.sh you@warwick.ac.uk  # END/FAIL alerts
#
# PREREQUISITE: run env/prefetch_med_llama.sh first (Med-LLaMA is gated + not cached).
# ==========================================================================
set -euo pipefail

EMAIL="${1:-}"
MAILOPT=()
[[ -n "${EMAIL}" ]] && MAILOPT=(--mail-user="${EMAIL}" --mail-type=END,FAIL)

DATASETS=(Analgesics-induced_acute_liver_failure Tramadol-related_mortalities)
CONFIGS=(paper r128)               # r8 (paper) and r128 only -> 4 jobs total
declare -A RANKOF=( [paper]=8 [r128]=128 )

echo "Submitting 4 jobs (paper + r128, both datasets)..."
for DS in "${DATASETS[@]}"; do
  case "${DS}" in Analgesics*) DTAG=ailf ;; Tramadol*) DTAG=tram ;; *) DTAG="${DS}" ;; esac
  for CF in "${CONFIGS[@]}"; do
    JN="mlm_${RANKOF[$CF]}_${DTAG}"
    jid="$(sbatch --parsable --job-name="${JN}" "${MAILOPT[@]}" slurm/med_llama_peft.slurm "${DS}" "${CF}")"
    echo "  ${JN}  ->  job ${jid}"
  done
done

echo ""
echo "Monitor:  squeue -u \$USER            # 4 jobs: mlm_8_ailf, mlm_8_tram, mlm_128_ailf, mlm_128_tram"
echo ""
echo "Progress (each -> 20):"
echo "  for d in ${DATASETS[*]}; do for c in ${CONFIGS[*]}; do \\"
echo "    echo -n \"\$d \$c: \"; ls dat/\$d/proc/cross_val_med_llama_\${c}_temp/df_res*.csv 2>/dev/null | wc -l; done; done"
echo ""
echo "If a job hits the 48h walltime mid-run, resubmit that one line (resumable):"
echo "  sbatch slurm/med_llama_peft.slurm Analgesics-induced_acute_liver_failure r128"
