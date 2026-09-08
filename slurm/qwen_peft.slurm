#!/bin/bash
# Qwen3.5-4B-Base / Qwen3.5-9B-Base QLoRA, TRUE 5-fold (all 5 folds in one single-GPU job).
#   sbatch slurm/qwen_peft.slurm <dataset> <config> <model>   (config: paper|r64|r128; model: qwen4b|qwen9b)
# Header cloned from your working gemma_peft.slurm; venv_qwen (transformers>=5.2). Resumable.
#SBATCH --job-name=qwen
#SBATCH --output=logs/qwen_%j.out
#SBATCH --error=logs/qwen_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=48:00:00
set -euo pipefail
DATASET="${1:?Usage: sbatch slurm/qwen_peft.slurm <dataset> <config> <model>}"
CONFIG="${2:?config: paper|r64|r128}"
MODEL="${3:?model: qwen4b|qwen9b}"
case "${CONFIG}" in paper) RANK=8 ;; r64) RANK=64 ;; r128) RANK=128 ;; *) RANK="${CONFIG}" ;; esac
case "${DATASET}" in Analgesics*) DTAG=ailf ;; Tramadol*) DTAG=tram ;; *) DTAG="${DATASET}" ;; esac
scontrol update JobId="${SLURM_JOB_ID}" JobName="${MODEL}_${RANK}_${DTAG}" 2>/dev/null || true

PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"; CUDA_MODULE="CUDA/12.4.0"
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv_qwen"
cd "${PROJECT_DIR}"
module purge; module load "${PYTHON_MODULE}" "${CUDA_MODULE}"
source "${VENV_DIR}/bin/activate"
export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA}}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 BITSANDBYTES_NOWELCOME=1
export HF_HOME="${PROJECT_DIR}/.hf_cache"; export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"; export HF_HUB_OFFLINE=1
# keep JIT/compile caches OFF the small HOME quota (Errno 122 lesson from Gemma)
export TRITON_CACHE_DIR="${PROJECT_DIR}/.triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.torchinductor_cache"
export TORCH_HOME="${PROJECT_DIR}/.torch"
export TORCH_EXTENSIONS_DIR="${PROJECT_DIR}/.torch_ext"
export CUDA_CACHE_PATH="${PROJECT_DIR}/.nv_cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" logs \
         "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_HOME}" \
         "${TORCH_EXTENSIONS_DIR}" "${CUDA_CACHE_PATH}"
export HF_TOKEN="${HF_TOKEN:-}"   # Qwen is Apache-2.0 (not gated); passed if set
export FOLD_SCRATCH="${TMPDIR:-/tmp}/qwen_${SLURM_JOB_ID}"; mkdir -p "${FOLD_SCRATCH}"
trap 'rm -rf "${FOLD_SCRATCH}"' EXIT
echo "=== ${MODEL} QLoRA | ${DATASET} | ${CONFIG} | 5-fold | $(hostname) | $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python "src/${DATASET}/qwen_peft_train.py" --model "${MODEL}" --dataset "${DATASET}" --config "${CONFIG}"
echo "=== done | ${MODEL} | ${DATASET} | ${CONFIG} | $(date) ==="
