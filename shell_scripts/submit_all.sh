#!/usr/bin/env bash
# ==========================================================================
# submit_all.sh  --  submit cross-validation training for BOTH datasets in one go,
# then queue a small "notifier" job that runs after both finish, writes a status
# summary, and (optionally) emails you.
#
# RUN ON THE BLYTHE LOGIN NODE, from $SHARE/$USER/biocausal_cluster:
#
#     bash submit_all.sh                              # albert, no email
#     bash submit_all.sh you@warwick.ac.uk            # albert, with email
#     bash submit_all.sh you@warwick.ac.uk biobert    # biobert, with email
#     bash submit_all.sh you@warwick.ac.uk pubmedbert
#     bash submit_all.sh "" biobert                   # biobert, no email
#
# Behaviour:
#   * The two dataset jobs are independent and may run in parallel (one L40 each).
#   * If an email is given: you get a FAIL email if either training job dies,
#     and an END/FAIL email from the notifier once both have finished.
#   * The notifier always writes logs/summary-<jobid>.out with sacct details and
#     the artefact locations, whether or not email is configured.
#
# NOTE on email: SLURM only delivers mail if the cluster has a mail relay
# configured and it accepts your address. If no email arrives, rely on the
# summary file / `sacct`. This does not affect the training itself.
# ==========================================================================
set -euo pipefail

# ---- args -----------------------------------------------------------------
MAIL_USER="${1:-${MAIL_USER:-}}"          # 1st arg, or MAIL_USER env, or empty
MODEL="${2:-albert}"                       # 2nd arg: albert | biobert (default albert)
MODEL="$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"
# ---------------------------------------------------------------------------

SLURM_SCRIPT="slurm/${MODEL}.slurm"
NOTIFY_SCRIPT="slurm/notify.slurm"
DATASETS=(
  "Analgesics-induced_acute_liver_failure"
  "Tramadol-related_mortalities"
)

# sanity checks
if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "ERROR: ${SLURM_SCRIPT} not found." >&2
  echo "       Available models:" >&2
  ls slurm/*.slurm 2>/dev/null | grep -v notify | sed 's#slurm/#         #; s#\.slurm##' >&2
  echo "       Run this from the project root ($SHARE/\$USER/biocausal_cluster)." >&2
  exit 1
fi
[[ -f "${NOTIFY_SCRIPT}" ]] || { echo "ERROR: ${NOTIFY_SCRIPT} not found." >&2; exit 1; }
mkdir -p logs

# short label for each dataset (for readable job names in squeue)
label() { case "$1" in *Analgesics*) echo ailf ;; *Tramadol*) echo tram ;; *) echo ds ;; esac; }

# Short model tag. squeue's default NAME column is only 8 characters wide, so
# "biobert_ailf" and "biobert_tram" both display as "biobert_" - indistinguishable.
# These 3-char tags keep every job name at exactly 8 chars so it shows in full.
tag() { case "$1" in albert) echo alb ;; biobert) echo bio ;; pubmedbert) echo pmb ;; biomistral) echo bmi ;; modernbert) echo mbt ;; *) echo "${1:0:3}" ;; esac; }
MTAG="$(tag "${MODEL}")"

echo "=================================================================="
echo "Model    : ${MODEL}   (${SLURM_SCRIPT})"
echo "Datasets : ${#DATASETS[@]}"
[[ -n "${MAIL_USER}" ]] && echo "Email    : ${MAIL_USER}" || echo "Email    : (none; summary file only)"
echo "=================================================================="

JOB_IDS=()
for ds in "${DATASETS[@]}"; do
  lbl="$(label "${ds}")"
  # per-training-job mail: only on FAIL, so you're alerted immediately if one breaks
  train_mail=()
  [[ -n "${MAIL_USER}" ]] && train_mail=(--mail-type=FAIL --mail-user="${MAIL_USER}")

  jid="$(sbatch --parsable "${train_mail[@]}" --job-name="${MTAG}_${lbl}" "${SLURM_SCRIPT}" "${ds}")"
  jid="${jid%%;*}"                        # strip ";cluster" suffix if present
  echo "  submitted ${lbl} (${ds}) -> job ${jid}"
  JOB_IDS+=("${jid}")
done

# dependency string: run notifier after ALL training jobs finish (any exit state)
DEP="afterany:$(IFS=:; echo "${JOB_IDS[*]}")"

# notifier mail: END+FAIL = the single "all done" signal
notify_mail=()
[[ -n "${MAIL_USER}" ]] && notify_mail=(--mail-type=END,FAIL --mail-user="${MAIL_USER}")

notify_jid="$(sbatch --parsable --dependency="${DEP}" \
                     --job-name="${MTAG}_note" \
                     --export=ALL,NOTIFY_MODEL="${MODEL}" \
                     "${notify_mail[@]}" "${NOTIFY_SCRIPT}" "${JOB_IDS[@]}")"
notify_jid="${notify_jid%%;*}"
echo "  submitted notifier -> job ${notify_jid} (waits on: ${DEP})"

echo "------------------------------------------------------------------"
echo "Monitor:   squeue -u \$USER"
echo "Live log:  tail -f logs/${MODEL}-<jobid>.out"
echo "Progress:  grep -c 'saved weights' logs/${MODEL}-<jobid>.out   # of 20 splits"
echo "Summary:   logs/summary-${notify_jid}.out   (written when everything finishes)"
echo "=================================================================="
