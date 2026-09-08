#!/usr/bin/env bash
# ==========================================================================
# prefetch_biomistral.sh -- RUN ONCE ON THE LOGIN NODE, BEFORE SUBMITTING JOBS
#
# Downloads BioMistral/BioMistral-7B (~14.5 GB) into the shared HF cache so the
# array jobs run OFFLINE (HF_HUB_OFFLINE=1) and never race to download the same
# weights. Safe to re-run: an already-cached model is skipped instantly.
#
# USAGE:
#     cd $SHARE/$USER/biocausal_cluster
#     bash env/prefetch_biomistral.sh
# ==========================================================================
set -euo pipefail

PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv_biomistral"
MODEL_ID="BioMistral/BioMistral-7B"

echo ">>> Project dir : ${PROJECT_DIR}"
module purge || true
module load "${PYTHON_MODULE}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export HF_HOME="${PROJECT_DIR}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"
unset HF_HUB_OFFLINE

echo ">>> Downloading ${MODEL_ID} into ${HF_HOME} ..."
MODEL_ID="${MODEL_ID}" python - <<'PY'
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
mid = os.environ["MODEL_ID"]
tok = AutoTokenizer.from_pretrained(mid)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForSequenceClassification.from_pretrained(mid, num_labels=2)
print(f"  cached OK: {mid}  ({sum(p.numel() for p in m.parameters()):,} params)")
PY

echo ">>> Verifying OFFLINE load (exactly as the jobs will)..."
HF_HUB_OFFLINE=1 MODEL_ID="${MODEL_ID}" python - <<'PY'
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
mid = os.environ["MODEL_ID"]
AutoTokenizer.from_pretrained(mid)
AutoModelForSequenceClassification.from_pretrained(mid, num_labels=2)
print("  offline load OK")
PY

echo ">>> Cache size: $(du -sh "${HF_HOME}" 2>/dev/null | cut -f1)"
echo ">>> DONE. You can now submit:  bash submit_biomistral.sh <your-email>"
