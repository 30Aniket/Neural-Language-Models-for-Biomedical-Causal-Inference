#!/bin/bash
# ==========================================================================
# biomistral_best.slurm -- "re-train the winner" for ONE dataset.
#
# After the 20-fold array for a dataset has finished, this:
#   1) reads the 20 predictions, picks the fold with the highest test AUC,
#   2) re-trains ONLY that fold (deterministic, seed=42) on 3x L40 ZeRO-3,
#   3) saves that single model + tokenizer to
#        outputs/biomistral/<dataset>/best_checkpoint/
# so you keep exactly ONE deployable checkpoint per dataset (~15 GB), instead of
# ~1 TB of per-fold checkpoints.
#
# RUN ONLY AFTER the matching 20-fold array is complete (20 df_res files present).
# SUBMIT:
#     sbatch slurm/biomistral_best.slurm Analgesics-induced_acute_liver_failure
#     sbatch slurm/biomistral_best.slurm Tramadol-related_mortalities
# ==========================================================================
#SBATCH --job-name=bmi_best
#SBATCH --output=logs/bmi_best_%j.out
#SBATCH --error=logs/bmi_best_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:3
#SBATCH --time=12:00:00

set -euo pipefail

DATASET="${1:?Usage: sbatch slurm/biomistral_best.slurm <dataset>}"

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
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export HF_HOME="${PROJECT_DIR}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p logs

# 1) pick the best fold by test AUC (tag printed to stdout; ranking to stderr/log)
echo ">>> Selecting best fold for ${DATASET} by test AUC..."
BEST_SPLIT="$(python pick_best_fold.py --dataset "${DATASET}")"
echo ">>> Best fold: ${BEST_SPLIT}"

CKPT_DIR="${PROJECT_DIR}/outputs/biomistral/${DATASET}/best_checkpoint"
mkdir -p "${CKPT_DIR}"

# node-local scratch for transient training checkpoints
export FOLD_SCRATCH="${TMPDIR:-/tmp}/biomistral_best_${SLURM_JOB_ID}"
mkdir -p "${FOLD_SCRATCH}"
trap 'rm -rf "${FOLD_SCRATCH}"' EXIT

echo "==================================================================="
echo "Re-training winning fold ${BEST_SPLIT} of ${DATASET} to save a checkpoint"
echo "Checkpoint -> ${CKPT_DIR}"
echo "Start      : $(date)   Node: $(hostname)"
echo "==================================================================="

# 2)+3) re-train just that fold and save the final (best-on-dev) model
torchrun --standalone --nnodes=1 --nproc_per_node=3 \
    "src/${DATASET}/biomistral_train.py" \
    --dataset "${DATASET}" --split "${BEST_SPLIT}" --save_final "${CKPT_DIR}"

echo "==================================================================="
echo "End        : $(date)"
echo "Saved      : ${CKPT_DIR}"
echo "==================================================================="
