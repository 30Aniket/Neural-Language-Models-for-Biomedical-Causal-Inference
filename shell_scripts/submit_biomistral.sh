#!/usr/bin/env bash
# ==========================================================================
# submit_biomistral.sh -- submit the 20-fold BioMistral array for BOTH datasets.
#
# RUN ON THE BLYTHE LOGIN NODE, from $SHARE/$USER/biocausal_cluster:
#     bash submit_biomistral.sh                       # no email
#     bash submit_biomistral.sh you@warwick.ac.uk     # END/FAIL email alerts
#
# Each dataset submits one array of 20 tasks (5 running at a time = 15 GPUs).
# After BOTH arrays finish (40 df_res files total), submit the best-fold jobs
# printed at the end to save one checkpoint per dataset.
# ==========================================================================
set -euo pipefail

EMAIL="${1:-}"
MAILOPT=()
[[ -n "${EMAIL}" ]] && MAILOPT=(--mail-user="${EMAIL}" --mail-type=END,FAIL)

DATASETS=(Analgesics-induced_acute_liver_failure Tramadol-related_mortalities)

for DS in "${DATASETS[@]}"; do
  jid="$(sbatch --parsable "${MAILOPT[@]}" slurm/biomistral.slurm "${DS}")"
  echo "submitted BioMistral 20-fold array for ${DS}  ->  job ${jid}  (tasks ${jid}_0 .. ${jid}_19)"
done

echo ""
echo "Monitor:   squeue -u \$USER"
echo "Progress:  for d in ${DATASETS[*]}; do echo -n \"\$d: \"; ls dat/\$d/proc/cross_val_biomistral_temp/df_res*.csv 2>/dev/null | wc -l; done   # each -> 20"
echo ""
echo "When BOTH arrays are done (40 df_res files), save the best checkpoints:"
echo "  sbatch slurm/biomistral_best.slurm Analgesics-induced_acute_liver_failure"
echo "  sbatch slurm/biomistral_best.slurm Tramadol-related_mortalities"
