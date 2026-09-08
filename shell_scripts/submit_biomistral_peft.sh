#!/usr/bin/env bash
# ==========================================================================
# submit_biomistral_peft.sh -- fire all 10 jobs (5 LoRA configs x 2 datasets).
# Each job trains ALL 20 folds on one GPU (encoder style). Tidy queue: 10 jobs.
# Job names: bmis_<rank>_<ailf|tram>  (e.g. bmis_8_ailf, bmis_256_tram).
#
# RUN ON THE LOGIN NODE, from $SHARE/$USER/biocausal_cluster:
#     bash submit_biomistral_peft.sh                    # no email
#     bash submit_biomistral_peft.sh you@warwick.ac.uk  # END/FAIL alerts
#
# RESUMABLE: if a job is killed at the 48h walltime part-way through its 20 folds,
# just resubmit that one line -- it skips finished folds.
# ==========================================================================
set -euo pipefail

EMAIL="${1:-}"
MAILOPT=()
[[ -n "${EMAIL}" ]] && MAILOPT=(--mail-user="${EMAIL}" --mail-type=END,FAIL)

DATASETS=(Analgesics-induced_acute_liver_failure Tramadol-related_mortalities)
CONFIGS=(paper r16 r64 r128 r256)
declare -A RANKOF=( [paper]=8 [r16]=16 [r64]=64 [r128]=128 [r256]=256 )

echo "Submitting 10 jobs (5 configs x 2 datasets), each runs all 20 folds..."
for DS in "${DATASETS[@]}"; do
  case "${DS}" in Analgesics*) DTAG=ailf ;; Tramadol*) DTAG=tram ;; *) DTAG="${DS}" ;; esac
  for CF in "${CONFIGS[@]}"; do
    JN="bmis_${RANKOF[$CF]}_${DTAG}"
    jid="$(sbatch --parsable --job-name="${JN}" "${MAILOPT[@]}" slurm/biomistral_peft.slurm "${DS}" "${CF}")"
    echo "  ${JN}  ->  job ${jid}"
  done
done

echo ""
echo "Monitor:  squeue -u \$USER            # 10 lines, named bmis_<rank>_<ailf|tram>"
echo ""
echo "Progress (each cell should reach 20):"
echo "  for d in ${DATASETS[*]}; do for c in ${CONFIGS[*]}; do \\"
echo "    echo -n \"\$d \$c: \"; ls dat/\$d/proc/cross_val_biomistral_\${c}_temp/df_res*.csv 2>/dev/null | wc -l; done; done"
echo ""
echo "If a job hits the 48h walltime mid-run, resubmit that one line, e.g.:"
echo "  sbatch slurm/biomistral_peft.slurm Analgesics-induced_acute_liver_failure r256"
echo ""
echo "When all reach 20, evaluate:  python run_evaluation.py --root \"\$(pwd)\""
