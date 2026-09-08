#!/usr/bin/env bash
# RUN ON BLYTHE, ONCE. Builds venv_gemma4 (Gemma 4 = April-2026 models, need a recent
# transformers), prefetches both GATED checkpoints, and smoke-tests before 4 long jobs.
# Prereq: accept the licence for BOTH google/gemma-4-12B and google/gemma-4-26B-A4B
# on huggingface.co, then: export HF_TOKEN=hf_xxx
set -euo pipefail
PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"; CUDA_MODULE="CUDA/12.4.0"
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
ROOT="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"; cd "$ROOT"
module purge; module load "$PYTHON_MODULE" "$CUDA_MODULE"

python -m venv venv_gemma4; source venv_gemma4/bin/activate
pip install -U pip wheel
# Gemma 4 support needs a recent transformers. Start with latest; if the model type is
# unrecognised, that's the thing to bump. Reuse your venv_qwen torch build if you can.
pip install -U "transformers>=4.57.0" "peft>=0.14.0" "bitsandbytes>=0.45.0" \
               "accelerate>=1.0.0" "datasets>=2.20.0" scikit-learn pandas numpy sentencepiece

export HF_HOME="$ROOT/.hf_cache"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
for m in ["google/gemma-4-12B", "google/gemma-4-26B-A4B"]:
    print("prefetching", m); snapshot_download(m, token=os.environ.get("HF_TOKEN"))
print("cache warmed.")
PY

# smoke on a GPU (submit as a batch job to the gpu partition -- interactive gpu isn't available):
#   the dry-run prints trainable params; --smoke runs a REAL 4-step train end-to-end.
echo "Prefetch done. Now smoke-test via a gpu batch job (see the smoke sbatch in the notes)."
