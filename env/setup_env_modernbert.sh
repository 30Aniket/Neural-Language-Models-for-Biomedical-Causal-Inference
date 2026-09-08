#!/usr/bin/env bash
# ==========================================================================
# setup_env_modernbert.sh -- build a SEPARATE venv for BioClinical ModernBERT.
#
# WHY A SEPARATE VENV?
# BioClinical ModernBERT (the ModernBERT architecture) requires transformers >= 4.48.
# Your main venv is pinned at transformers==4.41.2 because the repo's ALBERT/BioBERT
# scripts use an older Trainer API. Upgrading the shared venv would risk those verbatim
# scripts, so ModernBERT gets its own isolated environment. The other four models are
# untouched.
#
# RUN ONCE, ON THE LOGIN NODE:
#     cd $SHARE/$USER/biocausal_cluster/env
#     bash setup_env_modernbert.sh
# ==========================================================================
set -euo pipefail

PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"     # verify with: module av Python
TORCH_VERSION="2.5.0"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"

SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv_modernbert"        # <-- separate from the main venv
REQ_FILE="$(cd "$(dirname "$0")" && pwd)/requirements_modernbert.txt"

echo ">>> Project dir : ${PROJECT_DIR}"
echo ">>> Venv target : ${VENV_DIR}  (separate from the transformers==4.41.2 venv)"

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
echo ">>> Installing ModernBERT requirements (transformers >= 4.48)..."
pip install -r "${REQ_FILE}"

echo ">>> Verifying ModernBERT is importable..."
python - <<'PY'
import transformers, torch
print(f"torch        {torch.__version__} (cuda {torch.version.cuda})")
print(f"transformers {transformers.__version__}")
from transformers import AutoConfig
assert tuple(int(x) for x in transformers.__version__.split('.')[:2]) >= (4, 48), \
    "transformers too old for ModernBERT"
# ModernBertForSequenceClassification must be resolvable
from transformers import ModernBertForSequenceClassification  # noqa
print("ModernBertForSequenceClassification import: OK")
PY

echo ""
echo ">>> Pre-fetching the BioClinical-ModernBERT-base checkpoint into the shared cache..."
echo "    (done here, in venv_modernbert, because the main prefetch runs under"
echo "     transformers 4.41.2 which cannot instantiate ModernBERT.)"
export HF_HOME="${PROJECT_DIR}/.hf_cache"
mkdir -p "${HF_HOME}"
unset HF_HUB_OFFLINE
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
mid = "thomas-sounack/BioClinical-ModernBERT-base"
AutoTokenizer.from_pretrained(mid)
m = AutoModelForSequenceClassification.from_pretrained(mid, num_labels=2)
print(f"  cached OK: {mid}  ({sum(p.numel() for p in m.parameters()):,} params)")
PY
echo ">>> verifying offline load..."
HF_HUB_OFFLINE=1 python - <<'PY'
from transformers import AutoTokenizer, AutoModelForSequenceClassification
mid = "thomas-sounack/BioClinical-ModernBERT-base"
AutoTokenizer.from_pretrained(mid)
AutoModelForSequenceClassification.from_pretrained(mid, num_labels=2)
print("  offline load OK")
PY

echo ""
echo ">>> DONE. venv_modernbert ready and checkpoint cached."
echo ">>> venv size: $(du -sh "${VENV_DIR}" 2>/dev/null | cut -f1)"
echo ">>> The ModernBERT SLURM job activates THIS venv, not the main one."
