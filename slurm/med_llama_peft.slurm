#!/bin/bash
# Med-LLaMA-8B QLoRA, TRUE 5-fold (all 5 folds in one single-GPU job). No array.
#   sbatch slurm/biomistral_peft.slurm <dataset> <config>   (config: paper|r64|r128)
# Resumable: skips folds whose df_res exists. Job name set at submit + here.
#SBATCH --job-name=mlm
#SBATCH --output=logs/mlm_%j.out
#SBATCH --error=logs/mlm_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=48:00:00
set -euo pipefail
DATASET="${1:?Usage: sbatch slurm/biomistral_peft.slurm <dataset> <config>}"
CONFIG="${2:?config: paper|r64|r128}"
case "${CONFIG}" in paper) RANK=8 ;; r64) RANK=64 ;; r128) RANK=128 ;; *) RANK="${CONFIG}" ;; esac
case "${DATASET}" in Analgesics*) DTAG=ailf ;; Tramadol*) DTAG=tram ;; *) DTAG="${DATASET}" ;; esac
scontrol update JobId="${SLURM_JOB_ID}" JobName="mlm_${RANK}_${DTAG}" 2>/dev/null || true

PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"; CUDA_MODULE="CUDA/12.4.0"
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv_biomistral"
cd "${PROJECT_DIR}"
module purge; module load "${PYTHON_MODULE}" "${CUDA_MODULE}"
source "${VENV_DIR}/bin/activate"
export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA}}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 BITSANDBYTES_NOWELCOME=1
export HF_HOME="${PROJECT_DIR}/.hf_cache"; export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"; export HF_HUB_OFFLINE=1
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" logs
export HF_TOKEN="${HF_TOKEN:-}"
export FOLD_SCRATCH="${TMPDIR:-/tmp}/mlm_${SLURM_JOB_ID}"; mkdir -p "${FOLD_SCRATCH}"
trap 'rm -rf "${FOLD_SCRATCH}"' EXIT
echo "=== Med-LLaMA-8B QLoRA | ${DATASET} | ${CONFIG} | 5-fold | $(hostname) | $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
python "src/${DATASET}/med_llama_peft_train.py" --dataset "${DATASET}" --config "${CONFIG}"
echo "=== done | ${DATASET} | ${CONFIG} | $(date) ==="
