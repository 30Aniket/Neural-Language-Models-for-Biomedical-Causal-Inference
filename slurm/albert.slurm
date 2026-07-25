#!/bin/bash
# ==========================================================================
# albert.slurm  --  ALBERT cross-validation training on the Blythe GPU partition
#
# Blythe GPU nodes: 3x Nvidia L40 (48 GB VRAM) per node, CUDA cc 8.9.
# Rule: no more than 10 CPUs per GPU. We use 1 GPU + 10 CPUs.
# Your dcsresearch account allows up to gres/gpu=15, so running both datasets
# in parallel (2 GPUs total) is comfortably within limits.
#
# *** All paths point at $SHARE, never $HOME (2 GB quota). ***
#     $SHARE = /springbrook/share/dcsresearch
#     project = $SHARE/$USER/biocausal_cluster
#
# SUBMIT (from the project root on $SHARE, dataset folder name as 1st arg):
#     sbatch slurm/albert.slurm Analgesics-induced_acute_liver_failure
#     sbatch slurm/albert.slurm Tramadol-related_mortalities
#
# stdout/stderr (all print + logging output) go to logs/albert-<jobid>.out/.err
# ==========================================================================
#SBATCH --job-name=albert
#SBATCH --output=logs/albert-%j.out
#SBATCH --error=logs/albert-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=24:00:00

set -euo pipefail

# ---- must match env/setup_env.sh ------------------------------------------
PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"     # verify with: module av Python
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv"
# ---------------------------------------------------------------------------

# 1st positional arg = dataset folder name (defaults to AILF if omitted)
DATASET="${1:-Analgesics-induced_acute_liver_failure}"

echo "==================================================================="
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "Dataset       : ${DATASET}"
echo "Project dir   : ${PROJECT_DIR}"
echo "Start time    : $(date)"
echo "==================================================================="

# Work from the project root on $SHARE regardless of where sbatch was invoked.
cd "${PROJECT_DIR}"

module purge
module load "${PYTHON_MODULE}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Keep runs deterministic / avoid thread oversubscription on the shared node.
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

# All caches on $SHARE - HF_HOME defaults to ~/.cache/huggingface which would
# eat the 2 GB home quota with pretrained weights.
# NOTE: TRANSFORMERS_CACHE is deliberately NOT set (deprecated in favour of HF_HOME).
export HF_HOME="${PROJECT_DIR}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}"

# Read the pretrained checkpoint from the warm cache only - never download here.
# Two jobs starting at once would otherwise race on the same .incomplete blob in
# the shared cache and both crash. Populate the cache first with:
#     bash env/prefetch_models.sh
export HF_HUB_OFFLINE=1

# Preflight: confirm the checkpoint is cached. Depending on which env vars are
# set, HF stores it either at $HF_HOME/hub/models--... (HF_HOME semantics) or at
# $HF_HOME/models--... (legacy TRANSFORMERS_CACHE semantics), so search for both.
if ! find "${HF_HOME}" -maxdepth 3 -type d -name "models--textattack--albert-base-v2-imdb" -not -path "*/.locks/*" 2>/dev/null | grep -q .; then
  echo "ERROR: checkpoint not found anywhere under ${HF_HOME}" >&2
  echo "       Run 'bash env/prefetch_models.sh' on the login node first." >&2
  exit 1
fi
echo ">>> Cached checkpoint: $(find "${HF_HOME}" -maxdepth 3 -type d -name 'models--textattack--albert-base-v2-imdb' -not -path "*/.locks/*" | head -n1)"

export PYTHONUNBUFFERED=1                            # stream prints live into the .out file

echo ">>> GPU visible to this job:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
echo ">>> python  : $(which python)  ($(python --version 2>&1))"
echo ">>> HF_HOME : ${HF_HOME}"

# ---------------------------------------------------------------------------
# PER-JOB SCRATCH DIRECTORY  -- do not remove.
# The repo's TrainingArguments hardcode output_dir="results" and logging_dir="logs"
# RELATIVE TO THE WORKING DIRECTORY. If two jobs share a working directory (e.g.
# ALBERT and BioBERT both running on the same dataset), they write to the same
# results/ folder and each split's shutil.rmtree("results") destroys the other
# job's checkpoints. That causes "Parent directory results/checkpoint-N does not
# exist" crashes and, worse, one model silently loading the OTHER model's
# checkpoint as its "best model" - producing invalid results.
#
# So each job runs in its own scratch dir. The training script resolves dat/ and
# outputs/ from __file__ (not the cwd), and Python puts the script's own directory
# on sys.path, so `import albert_train_src` still works when invoked by full path.
# ---------------------------------------------------------------------------
SCRATCH="${PROJECT_DIR}/scratch/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH}"
cd "${SCRATCH}"
echo ">>> Scratch dir: ${SCRATCH}  (isolates results/ and logs/ from other jobs)"

echo ">>> Launching ALBERT training..."
srun python "${PROJECT_DIR}/src/${DATASET}/albert_train.py" --dataset "${DATASET}"

# Clean up the (now empty) scratch dir.
cd "${PROJECT_DIR}" && rm -rf "${SCRATCH}"

echo "==================================================================="
echo "End time      : $(date)"
echo "Artefacts     : ${PROJECT_DIR}/outputs/albert/${DATASET}/  (weights/, train_logs/, run_logs/)"
echo "Predictions   : ${PROJECT_DIR}/dat/${DATASET}/proc/cross_val_albert_temp/"
echo "==================================================================="
