#!/usr/bin/env bash
# ==========================================================================
# download_all.sh  --  RUN ON YOUR LOCAL MAC
#
#   bash download_all.sh /path/to/destination
#   bash download_all.sh ~/Desktop/biocausal_results
#   SSH_KEY=~/.ssh/id_rsa bash download_all.sh ~/results
#   WITH_31B=1 bash download_all.sh ~/results          # include gemma4_31b too
#   WITH_HF_CACHE=1 bash download_all.sh ~/results     # + the ~45 GB of base weights
#
# Pulls the RESULTS the cluster produced, for every model except gemma4_31b:
#   dat/<dataset>/proc/cross_val_*/     per-fold prediction CSVs (df_res*.csv)
#   outputs/<model>/<dataset>/
#       train_logs/                     per-fold training-curve CSVs
#       run_logs/                       persistent per-run text logs
#       weights/split_XX/               ENCODERS ONLY -- see note below
#   logs/                               SLURM .out / .err
#
# NOT downloaded: src/, slurm/, env/, data/, transfer/ and the top-level .sh/.py
# scripts. You already have those locally. Nothing here depends on them.
#
# gemma4_31b is EXCLUDED by default because both streams are still running.
# Re-run with WITH_31B=1 once they finish; rsync will fetch only what is new.
#
# NOTE ON WEIGHTS. Only the four encoders (albert, biobert, pubmedbert,
# modernbert) call trainer.save_model(). Every QLoRA decoder driver deletes its
# checkpoint directory after writing predictions, so no adapter weights exist on
# the cluster for BioMistral, Med-LLaMA, Qwen, Gemma, MedGemma, MediPhi or Phi-4.
# Their reproducible artefacts are the prediction CSVs and training curves.
#
# Skipped by default: .hf_cache (public base weights, re-downloadable),
# venv_* , and the JIT caches. Pass WITH_HF_CACHE=1 to include the cache.
#
# Resumable: rsync skips whatever is already present. Ctrl+C and re-run freely.
# ==========================================================================
set -euo pipefail

DEST="${1:-${DEST:-$(pwd)/biocausal_results}}"
USER_NAME="${USER_NAME:-cstqrd}"
DATA_NODE="${DATA_NODE:-blythedata.scrtp.warwick.ac.uk}"
REMOTE="${REMOTE:-/springbrook/share/dcsresearch/${USER_NAME}/biocausal_cluster}"
SSH_KEY="${SSH_KEY:-}"
WITH_HF_CACHE="${WITH_HF_CACHE:-0}"
WITH_31B="${WITH_31B:-0}"

case "${DEST}" in
  -h|--help|"") echo "usage: bash download_all.sh /path/to/destination" >&2; exit 2 ;;
esac
# catch the classic "~/Users/me/..." double expansion
if [[ "${DEST}" == "${HOME}/Users/"* || "${DEST}" == "${HOME}/home/"* ]]; then
  echo "!!! DEST looks doubled: ${DEST}" >&2
  echo "    '~' already expands to ${HOME}. Did you mean ${DEST/${HOME}\/Users\/*\//${HOME}/}?" >&2
  read -r -p "    Continue with ${DEST} anyway? [y/N] " yn
  [[ "${yn}" == "y" || "${yn}" == "Y" ]] || exit 1
fi

RSH="ssh"
[[ -n "${SSH_KEY}" ]] && RSH="ssh -i ${SSH_KEY}"

SKIP_31B=()
if [[ "${WITH_31B}" != "1" ]]; then
  SKIP_31B=(--exclude 'gemma4_31b*' --exclude 'cross_val_gemma4_31b*')
fi

COMMON_EXCLUDES=(
  --exclude '.hf_cache/'   --exclude 'venv_*/'      --exclude '__pycache__/'
  --exclude '.triton_cache/' --exclude '.torchinductor_cache/'
  --exclude '.torch/'      --exclude '.torch_ext/'  --exclude '.nv_cache/'
  --exclude '.cache/'      --exclude 'scratch/'     --exclude '*.bak_*'
  --exclude '.DS_Store'
)

echo "=================================================================="
echo " Source : ${USER_NAME}@${DATA_NODE}:${REMOTE}"
echo " Dest   : ${DEST}"
echo " gemma4_31b: $([[ "${WITH_31B}" == "1" ]] && echo 'included' || echo 'EXCLUDED (still running)')"
echo " HF cache  : $([[ "${WITH_HF_CACHE}" == "1" ]] && echo 'INCLUDED (+45 GB)' || echo 'skipped')"
echo "=================================================================="

mkdir -p "${DEST}"

# free space (df -g is macOS; -BG on Linux)
AVAIL=$(df -g "${DEST}" 2>/dev/null | awk 'NR==2{print $4}') \
  || AVAIL=$(df -BG "${DEST}" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}')
# measured 2026-09-07: 15.2 GB of encoder weights + ~0.5 GB of results
NEED=$([[ "${WITH_HF_CACHE}" == "1" ]] && echo 65 || echo 20)
echo ">>> Free space at destination: ${AVAIL:-unknown} GB  (need ~${NEED} GB)"
if [[ -n "${AVAIL:-}" && "${AVAIL}" -lt "${NEED}" ]]; then
  echo "!!! Less than ${NEED} GB free." >&2
  read -r -p "    Continue anyway? [y/N] " yn
  [[ "${yn}" == "y" || "${yn}" == "Y" ]] || exit 1
fi

# ---- 1. everything small: predictions, curves, logs, code ----------------
echo ""
echo ">>> [1/3] Predictions, training curves, run logs, SLURM logs, data (~200 MB)"
rsync -avzP -R -e "${RSH}" "${COMMON_EXCLUDES[@]}" "${SKIP_31B[@]}" \
    --exclude 'weights/' \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./outputs" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./dat" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./logs" \
    "${DEST}/"

# ---- 2. encoder weights (the big, slow part) -----------------------------
# No -z: safetensors are dense binary; compression burns CPU for nothing.
echo ""
echo ">>> [2/3] Encoder weights. Slow. Ctrl+C and re-run to resume."
rsync -avP --stats -e "${RSH}" "${SKIP_31B[@]}" \
    --include '*/' --include 'weights/***' --exclude '*' \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/outputs/" \
    "${DEST}/outputs/"

# ---- 2b. optional HF base-model cache ------------------------------------
if [[ "${WITH_HF_CACHE}" == "1" ]]; then
  echo ""
  echo ">>> [2b] Hugging Face base weights (~45 GB). These are public downloads."
  rsync -avP --stats -e "${RSH}" \
      "${USER_NAME}@${DATA_NODE}:${REMOTE}/.hf_cache/" "${DEST}/.hf_cache/"
fi

# ---- 3. verify -----------------------------------------------------------
echo ""
echo "=================================================================="
echo " VERIFICATION"
echo "=================================================================="
set +e          # find returns 1 on a missing path; pipefail + errexit would abort
printf "  %-34s %6s %7s %8s\n" "MODEL / DATASET" "FOLDS" "CURVES" "WEIGHTS"
total_p=0
if [[ -d "${DEST}/dat" ]]; then
  while IFS= read -r d; do
    key=$(basename "$d" | sed 's/^cross_val_//; s/_temp$//')
    # the prediction dir and the outputs dir disagree for exactly one model:
    # cross_val_modernbert_temp  <->  outputs/bioclinicalmodernbert/
    okey="$key"; [[ "$key" == "modernbert" ]] && okey="bioclinicalmodernbert"
    ds=$(basename "$(dirname "$(dirname "$d")")")
    dtag=ailf; [[ "$ds" == Tram* ]] && dtag=tram
    np=$(find "$d" -name 'df_res*.csv' 2>/dev/null | wc -l | tr -d ' ')
    nc=$(find "${DEST}/outputs/${okey}/${ds}/train_logs" -name 'train_curve_*.csv' 2>/dev/null | wc -l | tr -d ' ')
    nw=$(find "${DEST}/outputs/${okey}/${ds}/weights" -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
    note=""; [[ "${np}" == "0" ]] && note="   <-- empty, safe to delete"
    printf "  %-34s %6s %7s %8s%s\n" "${key} / ${dtag}" "${np}" "${nc}" "${nw}" "${note}"
    total_p=$((total_p + np))
  done < <(find "${DEST}/dat" -type d -name 'cross_val_*' 2>/dev/null | sort)
fi
set -e
echo ""
echo "  total prediction CSVs : ${total_p}"
echo "  total safetensors     : $(find "${DEST}/outputs" -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')  (encoders only, by design)"
echo "  SLURM .out/.err       : $(find "${DEST}/logs" -type f 2>/dev/null | wc -l | tr -d ' ')"
echo "  total size            : $(du -sh "${DEST}" 2>/dev/null | cut -f1)"
echo ""
echo "  Any row showing fewer than 5 folds is still running or was cut short."
if [[ "${WITH_31B}" != "1" ]]; then
  echo "  gemma4_31b was skipped. Once both streams finish:"
  echo "      WITH_31B=1 bash download_all.sh ${DEST}"
fi
echo "=================================================================="
