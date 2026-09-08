#!/usr/bin/env bash
# ==========================================================================
# setup_env_biomistral.sh -- build a SEPARATE venv for BioMistral-7B.
#
# WHY A SEPARATE VENV?
# BioMistral needs DeepSpeed + a recent transformers/accelerate for ZeRO-3 full
# fine-tuning. Keeping it isolated from the encoder venvs avoids disturbing the
# verbatim ALBERT/BioBERT/PubMedBERT/ModernBERT environments.
#
# RUN ONCE, ON THE LOGIN NODE:
#     cd $SHARE/$USER/biocausal_cluster/env
#     bash setup_env_biomistral.sh
# Then warm the model cache:  bash prefetch_biomistral.sh
# ==========================================================================
set -euo pipefail

PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"     # verify: module av Python
TORCH_VERSION="2.5.0"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"

SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv_biomistral"
REQ_FILE="$(cd "$(dirname "$0")" && pwd)/requirements_biomistral.txt"

echo ">>> Project dir : ${PROJECT_DIR}"
echo ">>> Venv target : ${VENV_DIR}"

case "${VENV_DIR}" in "${HOME}"/*) echo "ERROR: refusing to build in \$HOME (2 GB quota)." >&2; exit 1 ;; esac
[[ -d "${SHARE_DIR}" ]] || { echo "ERROR: ${SHARE_DIR} not found." >&2; exit 1; }

module purge
module load "${PYTHON_MODULE}"
echo ">>> Using: $(python --version)"

python -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export PIP_CACHE_DIR="${PROJECT_DIR}/.pip_cache"
mkdir -p "${PIP_CACHE_DIR}"

pip install --upgrade pip wheel
echo ">>> Installing torch ${TORCH_VERSION} (CUDA 12.4)..."
pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_CUDA_INDEX}"
echo ">>> Installing BioMistral requirements (transformers + deepspeed + accelerate)..."
pip install -r "${REQ_FILE}"

echo ">>> Verifying imports and DeepSpeed..."
python - <<'PY'
import torch, transformers, deepspeed, accelerate
print(f"torch        {torch.__version__} (cuda {torch.version.cuda})")
print(f"transformers {transformers.__version__}")
print(f"deepspeed    {deepspeed.__version__}")
print(f"accelerate   {accelerate.__version__}")
from transformers import MistralForSequenceClassification  # noqa: F401
print("MistralForSequenceClassification import: OK")
PY

echo ""
echo ">>> DONE. venv_biomistral ready."
echo ">>> venv size: $(du -sh "${VENV_DIR}" 2>/dev/null | cut -f1)"
echo ">>> NEXT: warm the model cache on the login node:"
echo "      bash env/prefetch_biomistral.sh"
