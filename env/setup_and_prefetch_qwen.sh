#!/usr/bin/env bash
# ==========================================================================
# setup_and_prefetch_qwen.sh  --  RUN ON BLYTHE, ONCE
#
# Builds venv_qwen (Qwen3.5 needs transformers>=5.2.0 -> a separate, newer venv than
# venv_gemma), prefetches both Apache-2.0 base checkpoints (NOT gated, no HF_TOKEN
# needed), and runs BOTH smoke checks before you queue 12 jobs:
#   --dry-run-model : builds model + LoRA, prints trainable params (fast)
#   --smoke         : a REAL 4-step train end-to-end (catches Trainer/API breakage that
#                     the dry-run misses -- this is the Gemma "tokenizer=" lesson)
# ==========================================================================
set -euo pipefail
PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"; CUDA_MODULE="CUDA/12.4.0"
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
ROOT="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
cd "$ROOT"
module purge; module load "$PYTHON_MODULE" "$CUDA_MODULE"

# ---- 1. venv (transformers 5.2 stack) -----------------------------------
python -m venv venv_qwen
source venv_qwen/bin/activate
pip install -U pip wheel
# Match the SAME working torch build as venv_gemma if you can; else a CUDA-12 build:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install \
  "transformers>=5.2.0" \
  "peft>=0.14.0" \
  "bitsandbytes>=0.45.0" \
  "accelerate>=1.0.0" \
  "datasets>=2.20.0" \
  scikit-learn pandas numpy sentencepiece

# ---- 2. prefetch weights on the login node ------------------------------
export HF_HOME="$ROOT/.hf_cache"
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-9B-Base"]:
    print("prefetching", m); snapshot_download(m)
print("cache warmed.")
PY

# ---- 3. smoke tests on a GPU (grab one interactively) -------------------
#   srun --partition=gpu --gres=gpu:lovelace_l40:1 --time=00:30:00 --pty bash
#   module load ... ; source venv_qwen/bin/activate ; export HF_HOME=$ROOT/.hf_cache
export TRITON_CACHE_DIR="$ROOT/.triton_cache"; mkdir -p "$TRITON_CACHE_DIR"
D="Analgesics-induced_acute_liver_failure"
for M in qwen4b qwen9b; do
  for C in paper r64 r128; do
    echo "=== dry-run: $M / $C ==="
    HF_HUB_OFFLINE=1 python "src/${D}/qwen_peft_train.py" --model "$M" --config "$C" --dataset "$D" --dry-run-model
  done
  echo "=== SMOKE (real 4-step train): $M / paper ==="
  HF_HUB_OFFLINE=1 python "src/${D}/qwen_peft_train.py" --model "$M" --config paper --dataset "$D" --smoke
done
echo ""
echo "If dry-runs print trainable params AND both SMOKE runs print 'SMOKE OK', submit: bash submit_qwen_5fold.sh"
echo "(Note: paper-config param counts will be SMALL -- q/v exist only in the attention layers of the GDN hybrid.)"
