#!/usr/bin/env bash
# ==========================================================================
# setup_env_peft.sh -- add PEFT + bitsandbytes to the EXISTING venv_biomistral.
#
# No new venv and no model download: the BioMistral-7B weights are already cached
# in .hf_cache from the full-FT work, and the 4-bit loader quantizes those on the fly.
#
# RUN ONCE, ON THE LOGIN NODE:
#     cd $SHARE/$USER/biocausal_cluster
#     bash env/setup_env_peft.sh
# ==========================================================================
set -euo pipefail

PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"
CUDA_MODULE="CUDA/12.4.0"
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv_biomistral"
REQ_FILE="$(cd "$(dirname "$0")" && pwd)/requirements_peft.txt"

echo ">>> venv: ${VENV_DIR}"
[[ -d "${VENV_DIR}" ]] || { echo "ERROR: ${VENV_DIR} not found. Build it with setup_env_biomistral.sh first." >&2; exit 1; }

module purge
module load "${PYTHON_MODULE}" "${CUDA_MODULE}"
export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA}}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export PIP_CACHE_DIR="${PROJECT_DIR}/.pip_cache"
mkdir -p "${PIP_CACHE_DIR}"

echo ">>> Installing PEFT + bitsandbytes..."
pip install -r "${REQ_FILE}"

echo ">>> Verifying imports + bitsandbytes CUDA..."
export BITSANDBYTES_NOWELCOME=1
python - <<'PY'
import torch, transformers, peft, bitsandbytes as bnb
print(f"torch        {torch.__version__} (cuda {torch.version.cuda})")
print(f"transformers {transformers.__version__}")
print(f"peft         {peft.__version__}")
print(f"bitsandbytes {bnb.__version__}")
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: F401
from transformers import BitsAndBytesConfig  # noqa: F401
print("PEFT + 4-bit imports: OK")
print("CUDA available:", torch.cuda.is_available())
PY

echo ""
echo ">>> DONE. venv_biomistral now has PEFT + bitsandbytes."
echo ">>> No prefetch needed (BioMistral-7B is already cached; 4-bit quantizes on load)."
echo ">>> Next: bash submit_biomistral_peft.sh <your-email>"
