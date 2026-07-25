#!/usr/bin/env bash
# ==========================================================================
# upload_to_cluster.sh  --  RUN ON YOUR LOCAL MACHINE
#
# Uploads the project (code + prepared data) to your Blythe SHARE directory.
#
# *** Target is $SHARE, NOT your home directory. ***
# Blythe home has a 2 GB / 100k-file quota and the login banner says all
# project work belongs on the share:
#     /springbrook/share/dcsresearch/<user>/biocausal_cluster
# Blythe also uses a SEPARATE data-transfer node: blythedata.scrtp.warwick.ac.uk
# (do NOT rsync through the login node).
#
# Usage:
#     bash transfer/upload_to_cluster.sh scrtp_username
#
# Optional overrides:
#     SSH_KEY=~/.ssh/mykey  bash transfer/upload_to_cluster.sh scrtp_username
#     REMOTE_BASE=/springbrook/share/other  bash transfer/upload_to_cluster.sh scrtp_username
# ==========================================================================
set -euo pipefail

USER_NAME="${1:?Usage: upload_to_cluster.sh <scrtp_username>}"
DATA_NODE="blythedata.scrtp.warwick.ac.uk"
REMOTE_BASE="${REMOTE_BASE:-/springbrook/share/dcsresearch}"      # your $SHARE
REMOTE_DIR="${REMOTE_BASE}/${USER_NAME}/biocausal_cluster"        # absolute path on Blythe
SSH_KEY="${SSH_KEY:-}"                                            # optional non-default key

# Local project root = parent of this transfer/ folder
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/"

RSYNC_SSH="ssh"
[[ -n "${SSH_KEY}" ]] && RSYNC_SSH="ssh -i ${SSH_KEY}"

echo ">>> Uploading ${LOCAL_DIR}"
echo ">>>        to ${USER_NAME}@${DATA_NODE}:${REMOTE_DIR}/"

# Make sure the remote directory tree exists on the share before rsyncing.
${RSYNC_SSH} "${USER_NAME}@${DATA_NODE}" "mkdir -p '${REMOTE_DIR}/logs'"

# -a archive, -v verbose, -z compress, -P progress+resume, --exclude scratch/caches
rsync -avzP -e "${RSYNC_SSH}" \
    --exclude '.git/' \
    --exclude 'venv/' \
    --exclude '.hf_cache/' \
    --exclude '.pip_cache/' \
    --exclude '.cache/' \
    --exclude '__pycache__/' \
    --exclude 'outputs/' \
    --exclude 'downloaded/' \
    --exclude 'results/' \
    --exclude 'logs/*.out' \
    --exclude 'logs/*.err' \
    "${LOCAL_DIR}" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE_DIR}/"

echo ""
echo ">>> Upload complete."
echo ">>> Next:"
echo "      ssh ${USER_NAME}@blythe.scrtp.warwick.ac.uk"
echo "      cd \$SHARE/\$USER/biocausal_cluster"
echo "      bash env/setup_env.sh"
