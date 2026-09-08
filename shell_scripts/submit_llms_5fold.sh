#!/usr/bin/env bash
# ==========================================================================
# submit_llms_5fold.sh -- fire all 12 LLM jobs at TRUE 5-fold:
#   {BioMistral-7B, Med-LLaMA-8B} x {paper(r8), r64, r128} x {AILF, TRAM}.
# Each job runs all 5 folds on one GPU. Job names: bmis_<rank>_<ds> / mlm_<rank>_<ds>.
#
#   bash submit_llms_5fold.sh                    # no email
#   bash submit_llms_5fold.sh you@warwick.ac.uk  # END/FAIL alerts
#   bash submit_llms_5fold.sh you@warwick.ac.uk biomistral   # only BioMistral (6 jobs)
#   bash submit_llms_5fold.sh you@warwick.ac.uk med_llama    # only Med-LLaMA (6 jobs)
#
# Resumable: if a job hits the 48h walltime, resubmit the same line to continue.
# ==========================================================================
set -euo pipefail
EMAIL="${1:-}"; WHICH="${2:-both}"
MAILOPT=(); [[ -n "${EMAIL}" ]] && MAILOPT=(--mail-user="${EMAIL}" --mail-type=END,FAIL)

DATASETS=(Analgesics-induced_acute_liver_failure Tramadol-related_mortalities)
CONFIGS=(paper r64 r128)
declare -A RANKOF=( [paper]=8 [r64]=64 [r128]=128 )

submit_model () {
  local model="$1" slurm="$2" prefix="$3"
  for DS in "${DATASETS[@]}"; do
    case "${DS}" in Analgesics*) DTAG=ailf ;; Tramadol*) DTAG=tram ;; *) DTAG="${DS}" ;; esac
    for CF in "${CONFIGS[@]}"; do
      JN="${prefix}_${RANKOF[$CF]}_${DTAG}"
      jid="$(sbatch --parsable --job-name="${JN}" "${MAILOPT[@]}" "${slurm}" "${DS}" "${CF}")"
      echo "  ${JN}  ->  job ${jid}"
    done
  done
}

echo "Submitting LLM 5-fold jobs (${WHICH})..."
[[ "${WHICH}" == "both" || "${WHICH}" == "biomistral" ]] && submit_model biomistral slurm/biomistral_peft.slurm bmis
[[ "${WHICH}" == "both" || "${WHICH}" == "med_llama" ]] && submit_model med_llama  slurm/med_llama_peft.slurm  mlm

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Progress (each -> 5):"
echo "  for m in biomistral med_llama; do for c in paper r64 r128; do for d in ${DATASETS[*]}; do \\"
echo "    echo -n \"\$m \$c \$d: \"; ls dat/\$d/proc/cross_val_\${m}_\${c}_temp/df_res*.csv 2>/dev/null|wc -l; done;done;done"
echo ""
echo "Resume a walltime-killed job, e.g.:"
echo "  sbatch slurm/biomistral_peft.slurm Tramadol-related_mortalities r128"
