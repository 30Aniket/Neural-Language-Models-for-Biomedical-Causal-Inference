#!/usr/bin/env bash
# ==========================================================================
# setup_env.sh
# Build a reproducible Python virtual environment for ALBERT/BioBERT training
# on the Blythe HPC cluster.
#
# *** IMPORTANT: everything is built on $SHARE, NOT in $HOME. ***
# Blythe home directories have a 2 GB / 100k-file quota, which a torch+CUDA
# venv would blow immediately. The login banner also states that all project
# work belongs on the share directory:
#     $HOME  = /springbrook/home/c/cstqrd        (2 GB   - keep empty-ish)
#     $SHARE = /springbrook/share/dcsresearch    (~10 TB - project work here)
#
# RUN THIS ONCE, ON THE LOGIN NODE (it only downloads packages; no GPU needed):
#     [cstqrd@login01(blythe) ~]$ cd $SHARE/$USER/biocausal_cluster/env
#     [cstqrd@login01(blythe) env]$ bash setup_env.sh
#
# Blythe uses a FLAT module system, so Python is loaded directly with
# `module load Python/<version>`. Verify the exact name available to you with:
#     module av Python
# and edit PYTHON_MODULE below if needed. Python 3.11 is recommended (torch 2.5).
# ==========================================================================
set -euo pipefail

# ---- EDIT THESE IF NEEDED -------------------------------------------------
PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"   # check with: module av Python
TORCH_VERSION="2.5.0"                          # matches the repo stack
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"  # CUDA 12.4 wheels for L40
# ---------------------------------------------------------------------------

# ---- Resolve the project location on $SHARE -------------------------------
# $SHARE is set for you at login. Fall back to the literal path if it is unset.
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv"
HF_CACHE="${PROJECT_DIR}/.hf_cache"
# ---------------------------------------------------------------------------

REQ_FILE="$(cd "$(dirname "$0")" && pwd)/requirements_albert.txt"

echo ">>> Share dir   : ${SHARE_DIR}"
echo ">>> Project dir : ${PROJECT_DIR}"
echo ">>> Venv target : ${VENV_DIR}"

if [[ ! -d "${SHARE_DIR}" ]]; then
  echo "ERROR: share directory ${SHARE_DIR} not found. Check \$SHARE (run: echo \$SHARE)." >&2
  exit 1
fi

# Guard against accidentally building in $HOME (the 2 GB trap).
case "${VENV_DIR}" in
  "${HOME}"/*) echo "ERROR: refusing to build the venv inside \$HOME (2 GB quota)." >&2; exit 1 ;;
esac

mkdir -p "${PROJECT_DIR}" "${HF_CACHE}"

echo ">>> Purging modules and loading Python..."
module purge
module load "${PYTHON_MODULE}"
echo ">>> Using: $(python --version) at $(which python)"

echo ">>> Creating virtual environment at ${VENV_DIR}..."
python -m venv "${VENV_DIR}"           # clean venv (NOT --system-site-packages) for reproducibility
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Keep pip's cache off $HOME too (it defaults to ~/.cache/pip and gets large).
export PIP_CACHE_DIR="${PROJECT_DIR}/.pip_cache"
mkdir -p "${PIP_CACHE_DIR}"

echo ">>> Upgrading pip / wheel..."
pip install --upgrade pip wheel

echo ">>> Installing PyTorch ${TORCH_VERSION} (CUDA build)..."
pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_CUDA_INDEX}"

echo ">>> Installing pinned project requirements..."
pip install -r "${REQ_FILE}"

echo ">>> Verifying install..."
python - <<'PY'
import torch, transformers, datasets, sklearn, pandas, numpy
print(f"torch        {torch.__version__}  (cuda build: {torch.version.cuda})")
print(f"transformers {transformers.__version__}")
print(f"datasets     {datasets.__version__}")
print(f"sklearn      {sklearn.__version__}")
print(f"pandas       {pandas.__version__}")
print(f"numpy        {numpy.__version__}")
print("NOTE: torch.cuda.is_available() is False on the login node (no GPU) - that is expected.")
PY

echo ""
echo ">>> Disk usage check:"
echo "    venv size  : $(du -sh "${VENV_DIR}" 2>/dev/null | cut -f1)"
echo "    home usage : $(du -sh "${HOME}" 2>/dev/null | cut -f1)   (quota 2 G - keep small)"
echo ""
echo ">>> DONE. Environment ready at ${VENV_DIR}"
echo ">>> The SLURM job loads '${PYTHON_MODULE}' then activates this venv."
echo ">>> If you changed PYTHON_MODULE here, change it in slurm/albert.slurm too."
