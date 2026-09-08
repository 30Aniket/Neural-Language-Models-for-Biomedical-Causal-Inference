#!/usr/bin/env bash
# ==========================================================================
# prefetch_med_llama.sh -- RUN ONCE ON THE LOGIN NODE, before submitting jobs.
#
# Med-LLaMA (YBXL/Med-LLaMA3-8B) is NOT yet cached (unlike BioMistral) and is very
# likely GATED. Steps:
#   1) On huggingface.co, request/accept access to YBXL/Med-LLaMA3-8B (and Meta
#      Llama-3 if it derives from a gated base).
#   2) Create a read token: https://huggingface.co/settings/tokens
#   3) export it and run this:
#        export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
#        bash env/prefetch_med_llama.sh
#
# Downloads ~16 GB into .hf_cache so the training jobs run OFFLINE (HF_HUB_OFFLINE=1)
# and never race to download. Safe to re-run (already-cached files are skipped).
# ==========================================================================
set -euo pipefail

PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"
CUDA_MODULE="CUDA/12.4.0"
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv_biomistral"
MODEL_ID="YBXL/Med-LLaMA3-8B"

: "${HF_TOKEN:?ERROR: export HF_TOKEN=hf_... first (Med-LLaMA is gated).}"

echo ">>> Project: ${PROJECT_DIR}"
module purge || true
module load "${PYTHON_MODULE}" "${CUDA_MODULE}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export HF_HOME="${PROJECT_DIR}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"
unset HF_HUB_OFFLINE

echo ">>> Downloading ${MODEL_ID} into ${HF_HOME} (this is ~16 GB)..."
MODEL_ID="${MODEL_ID}" python - <<'PY'
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
mid = os.environ["MODEL_ID"]; tok = os.environ.get("HF_TOKEN")
t = AutoTokenizer.from_pretrained(mid, trust_remote_code=True, token=tok)
if t.pad_token is None: t.pad_token = t.eos_token
m = AutoModelForSequenceClassification.from_pretrained(mid, num_labels=2, trust_remote_code=True, token=tok)
print(f"  cached OK: {mid}  ({sum(p.numel() for p in m.parameters()):,} params)")
PY

echo ">>> Verifying OFFLINE load..."
HF_HUB_OFFLINE=1 MODEL_ID="${MODEL_ID}" python - <<'PY'
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
mid = os.environ["MODEL_ID"]
AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
AutoModelForSequenceClassification.from_pretrained(mid, num_labels=2, trust_remote_code=True)
print("  offline load OK")
PY

echo ">>> Cache size: $(du -sh "${HF_HOME}" 2>/dev/null | cut -f1)"
echo ">>> DONE. Now submit:  bash submit_med_llama_peft.sh <your-email>"
