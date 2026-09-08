#!/usr/bin/env bash
# ==========================================================================
# setup_and_prefetch_gemma.sh  --  RUN ON BLYTHE, ONCE
#
# Builds venv_gemma (Gemma3/MedGemma need transformers>=4.55, so a separate venv
# from venv_biomistral), warms the HF cache for both GATED checkpoints, and runs
# the loader smoke test before you commit GPU time to 12 jobs.
#
# Prereqs: accept the licence for BOTH google/gemma-3-4b-it and
# google/medgemma-1.5-4b-it on huggingface.co while logged in, then:
#     export HF_TOKEN=hf_xxx
# ==========================================================================
set -euo pipefail
PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"; CUDA_MODULE="CUDA/12.4.0"
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
ROOT="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
cd "$ROOT"
module purge; module load "$PYTHON_MODULE" "$CUDA_MODULE"

# ---- 1. venv ------------------------------------------------------------
python -m venv venv_gemma
source venv_gemma/bin/activate
pip install -U pip wheel
# Reuse the SAME torch build as your venv_biomistral if possible (match versions).
# Otherwise install a CUDA-12 build:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install \
  "transformers>=4.55.0" \
  "peft>=0.13.0" \
  "bitsandbytes>=0.44.0" \
  "accelerate>=0.34.0" \
  "datasets>=2.20.0" \
  scikit-learn pandas numpy sentencepiece

# ---- 2. prefetch gated weights on the login node ------------------------
export HF_HOME="$ROOT/.hf_cache"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
for m in ["google/gemma-3-4b-it", "google/medgemma-1.5-4b-it"]:
    print("prefetching", m)
    snapshot_download(m, token=os.environ.get("HF_TOKEN"))
print("cache warmed at", os.environ["HF_HOME"])
PY

# ---- 3. smoke test each config BEFORE queuing 12 jobs -------------------
# Loads the model + LoRA wrap and prints trainable params, no training, no data.
# Run this on a GPU (grab one interactively): the 4-bit load needs a device.
#   srun --partition=gpu --gres=gpu:lovelace_l40:1 --time=00:20:00 --pty bash
#   source venv_gemma/bin/activate && export HF_TOKEN=... HF_HOME=$ROOT/.hf_cache
D="Analgesics-induced_acute_liver_failure"
for M in gemma3 medgemma; do
  for C in paper r64 r128; do
    echo "=== dry run: $M / $C ==="
    HF_HUB_OFFLINE=1 python "src/${D}/gemma_peft_train.py" \
        --model "$M" --config "$C" --dataset "$D" --dry-run-model
  done
done
echo ""
echo "Expected trainable params (identical for gemma3 & medgemma):"
echo "  paper -> 2,228,224   |   r64 -> 119,209,984   |   r128 -> 238,419,968"
echo "If all six print DRY RUN OK with those counts, submit: bash submit_gemma_5fold.sh"
