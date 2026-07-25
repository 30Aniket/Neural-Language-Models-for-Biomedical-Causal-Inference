#!/usr/bin/env bash
# ==========================================================================
# download_all.sh  --  RUN ON YOUR LOCAL MAC
#
# Pulls EVERYTHING from the Blythe project back to your machine:
#   * outputs/<model>/<dataset>/weights/split_XX/   all 20 splits x 3 models x 2 datasets
#   * outputs/<model>/<dataset>/train_logs/         per-split training-curve CSVs
#   * outputs/<model>/<dataset>/run_logs/           per-run text logs
#   * dat/<dataset>/proc/cross_val_<model>_temp/    per-split prediction CSVs
#   * logs/                                         SLURM .out/.err + summaries
#   * src/, slurm/, env/, data/, transfer/          the exact code that produced it all
#
# Total ~37 GB. Resumable: if it dies, just run it again - rsync skips what it has.
#
# Usage:
#     bash download_all.sh
#     SSH_KEY=~/.ssh/mykey bash download_all.sh      # non-default key
#     DEST=/Volumes/ExtDisk/biocausal bash download_all.sh   # e.g. external drive
# ==========================================================================
set -euo pipefail

USER_NAME="${USER_NAME:-scrtp_username}"
DATA_NODE="blythedata.scrtp.warwick.ac.uk"
REMOTE="/springbrook/share/dcsresearch/${USER_NAME}/biocausal_cluster"
DEST="${DEST:-$(pwd)/biocausal_results}"
SSH_KEY="${SSH_KEY:-}"

RSH="ssh"
[[ -n "${SSH_KEY}" ]] && RSH="ssh -i ${SSH_KEY}"

echo "=================================================================="
echo " Source : ${USER_NAME}@${DATA_NODE}:${REMOTE}"
echo " Dest   : ${DEST}"
echo "=================================================================="

# ---- 0. disk space check -------------------------------------------------
AVAIL=$(df -g "$(dirname "${DEST}")" 2>/dev/null | awk 'NR==2{print $4}')
echo ">>> Free space at destination: ${AVAIL:-?} GB  (need ~40 GB)"
if [[ -n "${AVAIL}" && "${AVAIL}" -lt 40 ]]; then
  echo "!!! WARNING: less than 40 GB free. Set DEST=/Volumes/... to use another disk." >&2
  read -r -p "    Continue anyway? [y/N] " yn
  [[ "${yn}" == "y" || "${yn}" == "Y" ]] || exit 1
fi

mkdir -p "${DEST}"

# ---- 1. SMALL FILES FIRST (fast - lets you start analysing immediately) ---
echo ""
echo ">>> [1/3] Predictions, training curves, run logs, SLURM logs (~80 MB)..."
rsync -avzP -e "${RSH}" \
    --exclude 'weights/' \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/outputs/" \
    "${DEST}/outputs/"

rsync -avzP -R -e "${RSH}" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./dat/Analgesics-induced_acute_liver_failure/proc/cross_val_albert_temp" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./dat/Analgesics-induced_acute_liver_failure/proc/cross_val_biobert_temp" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./dat/Analgesics-induced_acute_liver_failure/proc/cross_val_pubmedbert_temp" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./dat/Tramadol-related_mortalities/proc/cross_val_albert_temp" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./dat/Tramadol-related_mortalities/proc/cross_val_biobert_temp" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./dat/Tramadol-related_mortalities/proc/cross_val_pubmedbert_temp" \
    "${DEST}/"

rsync -avzP -e "${RSH}" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/logs/" \
    "${DEST}/logs/"

# ---- 2. CODE (tiny, but makes the results self-documenting) --------------
echo ""
echo ">>> [2/3] Code that produced these results (~200 KB)..."
rsync -avzP -R -e "${RSH}" \
    --exclude '__pycache__/' \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./src" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./slurm" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./env" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./data" \
    "${DEST}/"

# ---- 3. WEIGHTS (the big one) --------------------------------------------
# No -z: safetensors are dense binary, compression costs CPU and saves nothing.
echo ""
echo ">>> [3/3] Model weights - 120 checkpoints, ~37 GB. This is the slow part."
echo "    (Interrupt any time with Ctrl+C; re-run this script to resume.)"
rsync -avP --stats -e "${RSH}" \
    --include '*/' --include 'weights/***' --exclude '*' \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/outputs/" \
    "${DEST}/outputs/"

# ---- 4. verify -----------------------------------------------------------
echo ""
echo "=================================================================="
echo " VERIFICATION"
echo "=================================================================="
N_W=$(find "${DEST}/outputs" -name 'model.safetensors' 2>/dev/null | wc -l | tr -d ' ')
N_P=$(find "${DEST}/dat" -name 'df_res*.csv' 2>/dev/null | wc -l | tr -d ' ')
N_C=$(find "${DEST}/outputs" -name 'train_curve_*.csv' 2>/dev/null | wc -l | tr -d ' ')
echo "  model.safetensors  : ${N_W}   (expect 120 = 3 models x 2 datasets x 20 splits)"
echo "  prediction CSVs    : ${N_P}   (expect 120)"
echo "  training curves    : ${N_C}   (expect 120)"
echo "  total size         : $(du -sh "${DEST}" | cut -f1)"
echo ""
[[ "${N_W}" == "120" && "${N_P}" == "120" ]] \
  && echo "  RESULT: complete." \
  || echo "  RESULT: incomplete - re-run this script to fetch what is missing."
echo "=================================================================="
