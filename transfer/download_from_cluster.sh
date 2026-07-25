#!/usr/bin/env bash
# ==========================================================================
# download_from_cluster.sh  --  RUN ON YOUR LOCAL MACHINE
#
# Pulls training artefacts back from your Blythe SHARE directory so you can do
# evaluation / testing locally. For the given model + dataset it brings back:
#   * outputs/<model>/<dataset>/weights/     (per-split learned weights, safetensors)
#   * outputs/<model>/<dataset>/train_logs/  (per-split training-curve CSVs)
#   * outputs/<model>/<dataset>/run_logs/    (full text logs)
#   * dat/<dataset>/proc/cross_val_<model>_temp/  (per-split prediction CSVs)
#   * logs/                                   (SLURM .out/.err, summaries)
#
# Source is $SHARE on Blythe:
#     /springbrook/share/dcsresearch/<user>/biocausal_cluster
# via the data-transfer node blythedata.scrtp.warwick.ac.uk
#
# Usage:
#     bash transfer/download_from_cluster.sh <user> [dataset] [model]
#
# Examples:
#     bash transfer/download_from_cluster.sh scrtp_username Analgesics-induced_acute_liver_failure albert
#     bash transfer/download_from_cluster.sh scrtp_username Tramadol-related_mortalities biobert
#
# Options:
#     NO_WEIGHTS=1 ...    skip the weights and fetch only predictions, curves and
#                         logs - handy for a quick look. (ALBERT weights are
#                         ~45 MB/split, BioBERT ~420 MB/split, so 20 BioBERT
#                         splits is roughly 8 GB per dataset.)
# ==========================================================================
set -euo pipefail

USER_NAME="${1:?Usage: download_from_cluster.sh <scrtp_username> [dataset_folder] [model]}"
DATASET="${2:-Analgesics-induced_acute_liver_failure}"
MODEL="${3:-albert}"
MODEL="$(echo "${MODEL}" | tr '[:upper:]' '[:lower:]')"

DATA_NODE="blythedata.scrtp.warwick.ac.uk"
REMOTE_BASE="${REMOTE_BASE:-/springbrook/share/dcsresearch}"      # your $SHARE
REMOTE_DIR="${REMOTE_BASE}/${USER_NAME}/biocausal_cluster"
SSH_KEY="${SSH_KEY:-}"

LOCAL_OUT="$(cd "$(dirname "$0")/.." && pwd)/downloaded"
mkdir -p "${LOCAL_OUT}"

RSYNC_SSH="ssh"
[[ -n "${SSH_KEY}" ]] && RSYNC_SSH="ssh -i ${SSH_KEY}"

echo ">>> Model   : ${MODEL}"
echo ">>> Dataset : ${DATASET}"
echo ">>> From    : ${USER_NAME}@${DATA_NODE}:${REMOTE_DIR}"
echo ">>> To      : ${LOCAL_OUT}/"

# Optionally skip the large weight folders.
EXCLUDES=()
if [[ -n "${NO_WEIGHTS:-}" ]]; then
  echo ">>> NO_WEIGHTS set - skipping weights/ directories."
  EXCLUDES+=(--exclude 'weights/')
fi

# Use --relative (-R) with a /./ pivot so the tree below the project root is
# recreated under LOCAL_OUT (i.e. outputs/..., dat/..., logs/...).
rsync -avzP -R "${EXCLUDES[@]}" -e "${RSYNC_SSH}" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE_DIR}/./outputs/${MODEL}/${DATASET}" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE_DIR}/./dat/${DATASET}/proc/cross_val_${MODEL}_temp" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE_DIR}/./logs" \
    "${LOCAL_OUT}/"

echo ""
echo ">>> Download complete. Files are under: ${LOCAL_OUT}/"
echo ">>> Reload a split's weights locally with:"
echo "    from transformers import AutoModelForSequenceClassification, AutoTokenizer"
echo "    p='downloaded/outputs/${MODEL}/${DATASET}/weights/split_13'"
echo "    model = AutoModelForSequenceClassification.from_pretrained(p)"
echo "    tok   = AutoTokenizer.from_pretrained(p)"
