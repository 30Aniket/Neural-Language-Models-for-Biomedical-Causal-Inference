#!/bin/bash
# ==========================================================================
# notify.slurm  --  completion notifier for a training batch.
# Submitted automatically by submit_all.sh with --dependency=afterany on the
# training jobs, the job IDs as positional arguments, and NOTIFY_MODEL exported.
#
# It runs only after all training jobs have finished (any state), prints an
# sacct status table + artefact locations, and (if submit_all.sh was given an
# email) causes SLURM to send the "all done" email via --mail-type=END.
#
# Tiny CPU-only job. If your default/compute partition is named differently,
# check with `sinfo` and edit --partition below.
# ==========================================================================
#SBATCH --job-name=notify
#SBATCH --output=logs/summary-%j.out
#SBATCH --error=logs/summary-%j.err
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=00:05:00

set -uo pipefail

MODEL="${NOTIFY_MODEL:-albert}"      # exported by submit_all.sh

DATASETS=(
  "Analgesics-induced_acute_liver_failure"
  "Tramadol-related_mortalities"
)

echo "=================================================================="
echo " ${MODEL} batch - completion summary"
echo " Generated : $(date)"
echo " Notifier  : job ${SLURM_JOB_ID} on $(hostname)"
echo " Training jobs watched: $*"
echo "=================================================================="
echo

overall_ok=1
for jid in "$@"; do
  echo "----- training job ${jid} -----"
  # Human-readable status table for the job and its steps.
  sacct -j "${jid}" \
        --format=JobID%14,JobName%16,Partition%10,State%16,Elapsed%12,MaxRSS%10,ExitCode%8 || true
  # Flag non-success in the parent job's state.
  state="$(sacct -n -X -j "${jid}" --format=State 2>/dev/null | head -n1 | tr -d ' ')"
  case "${state}" in
    COMPLETED) : ;;
    "" )       echo "  (could not read state for ${jid})"; overall_ok=0 ;;
    * )        echo "  >> job ${jid} ended in state: ${state}"; overall_ok=0 ;;
  esac
  echo
done

echo "------------------------------------------------------------------"
echo " Artefact locations (per dataset):"
for ds in "${DATASETS[@]}"; do
  echo "  [${ds}]"
  echo "    weights : outputs/${MODEL}/${ds}/weights/split_<dev><test>/"
  echo "    curves  : outputs/${MODEL}/${ds}/train_logs/"
  echo "    run log : outputs/${MODEL}/${ds}/run_logs/"
  echo "    preds   : dat/${ds}/proc/cross_val_${MODEL}_temp/"
  # completed-split counts, if the dirs exist
  wdir="outputs/${MODEL}/${ds}/weights"
  if [[ -d "${wdir}" ]]; then
    n=$(find "${wdir}" -maxdepth 1 -type d -name 'split_*' 2>/dev/null | wc -l)
    echo "    splits completed: ${n} / 20"
    [[ "${n}" -lt 20 ]] && overall_ok=0
  fi
done
echo "------------------------------------------------------------------"

if [[ "${overall_ok}" -eq 1 ]]; then
  echo " RESULT: all watched training jobs COMPLETED successfully (20/20 splits)."
else
  echo " RESULT: one or more jobs did NOT complete cleanly, or fewer than 20 splits"
  echo "         were produced - check the tables above and the matching"
  echo "         logs/${MODEL}-<jobid>.err files."
fi
echo "=================================================================="

# Non-zero exit on failure so the SLURM email subject reflects FAIL vs END.
[[ "${overall_ok}" -eq 1 ]] || exit 1
