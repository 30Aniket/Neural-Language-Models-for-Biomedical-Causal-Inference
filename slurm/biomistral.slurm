#!/bin/bash
# ==========================================================================
# biomistral.slurm -- BioMistral-7B FULL fine-tuning, 20-fold CV as a job ARRAY.
#
# One array task = ONE cross-validation fold = one fresh process on 3x L40 running
# DeepSpeed ZeRO-3 (bf16, no offload). 20 folds, throttled to 5 running at once
# (5 x 3 = 15 GPUs = your per-user cap). Predictions-only: no checkpoints saved
# during the array (the best fold is re-trained + saved separately afterwards).
#
# SUBMIT (from the project root on $SHARE), ONE dataset at a time:
#     sbatch slurm/biomistral.slurm Analgesics-induced_acute_liver_failure
#     sbatch slurm/biomistral.slurm Tramadol-related_mortalities
# (or use submit_biomistral.sh to do both with email alerts).
# ==========================================================================
#SBATCH --job-name=bmi
#SBATCH --output=logs/bmi_%A_%a.out
#SBATCH --error=logs/bmi_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:3
#SBATCH --time=48:00:00
#SBATCH --array=0-19%5

set -euo pipefail

DATASET="${1:?Usage: sbatch slurm/biomistral.slurm <dataset>}"

# 20 CV folds as <dev><test> tags (i != j), indexed by SLURM_ARRAY_TASK_ID 0..19.
SPLITS=(01 02 03 04 10 12 13 14 20 21 23 24 30 31 32 34 40 41 42 43)
SPLIT="${SPLITS[$SLURM_ARRAY_TASK_ID]}"

PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"
CUDA_MODULE="CUDA/12.4.0"
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv_biomistral"

cd "${PROJECT_DIR}"

module purge
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"
# DeepSpeed inspects the system CUDA toolkit via CUDA_HOME; EBROOTCUDA is set by the
# CUDA module. We build NO custom ops (torch AdamW, no offload), so nothing is compiled.
export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA}}"
export DS_BUILD_OPS=0
export DS_BUILD_FP_QUANTIZER=0
# --- NCCL transport settings for single-node 3xGPU (avoid first-collective hangs) ---
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_CUDA_ARCH_LIST="8.9"   # L40 = Ada / compute_89 (skips arch probing)
[[ -d "${VENV_DIR}" ]] || { echo "ERROR: ${VENV_DIR} missing. Run env/setup_env_biomistral.sh first." >&2; exit 1; }
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export HF_HOME="${PROJECT_DIR}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" logs

if ! find "${HF_HOME}" -maxdepth 3 -type d -name "models--BioMistral--BioMistral-7B" -not -path "*/.locks/*" 2>/dev/null | grep -q .; then
  echo "ERROR: BioMistral-7B not cached under ${HF_HOME}. Run 'bash env/prefetch_biomistral.sh' first." >&2
  exit 1
fi

# Node-local SSD scratch for DeepSpeed's (large) transient checkpoints -- NOT on $SHARE.
export FOLD_SCRATCH="${TMPDIR:-/tmp}/biomistral_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${FOLD_SCRATCH}"
trap 'rm -rf "${FOLD_SCRATCH}"' EXIT

echo "==================================================================="
echo "BioMistral-7B | dataset ${DATASET} | fold ${SPLIT} (array ${SLURM_ARRAY_TASK_ID})"
echo "Job      : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}   Node: $(hostname)"
echo "Scratch  : ${FOLD_SCRATCH}"
echo "Start    : $(date)"
echo "==================================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
echo ">>> python: $(which python)  transformers $(python -c 'import transformers;print(transformers.__version__)')  deepspeed $(python -c 'import deepspeed;print(deepspeed.__version__)')"

echo ">>> Launching BioMistral-7B fold ${SPLIT} on 3x L40 (torchrun, ZeRO-3)..."
torchrun --standalone --nnodes=1 --nproc_per_node=3 \
    "src/${DATASET}/biomistral_train.py" --dataset "${DATASET}" --split "${SPLIT}"

echo "==================================================================="
echo "End      : $(date)"
echo "Preds    : ${PROJECT_DIR}/dat/${DATASET}/proc/cross_val_biomistral_temp/df_res${SPLIT}.csv"
echo "==================================================================="
