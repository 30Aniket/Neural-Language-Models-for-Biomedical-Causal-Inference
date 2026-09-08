#!/bin/bash
# mediphi_4b QLoRA r8, true 5-fold on one L40. One PROCESS PER FOLD.
#   sbatch slurm/mediphi_4b_peft.slurm <dataset> [config]
# MIT-licensed and ungated: no HF_TOKEN or licence acceptance needed.
#SBATCH --job-name=mediphi_4b
#SBATCH --output=logs/mediphi_4b_%j.out
#SBATCH --error=logs/mediphi_4b_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=24:00:00
set -euo pipefail

DATASET="${1:?Usage: sbatch slurm/mediphi_4b_peft.slurm <dataset> [config]}"
CONFIG="${2:-paper}"
MODEL_NAME="mediphi_4b"

PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"; CUDA_MODULE="CUDA/12.4.0"
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
cd "${PROJECT_DIR}"
module purge; module load "${PYTHON_MODULE}" "${CUDA_MODULE}"
# newest stack in the project; Phi3ForCausalLM is native there, no remote code
source "${PROJECT_DIR}/${VENV:-venv_gemma4}/bin/activate"
export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA}}"

export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 BITSANDBYTES_NOWELCOME=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME="${PROJECT_DIR}/.hf_cache"; export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"; export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR="${PROJECT_DIR}/.triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.torchinductor_cache"
export TORCH_HOME="${PROJECT_DIR}/.torch"
export TORCH_EXTENSIONS_DIR="${PROJECT_DIR}/.torch_ext"
export CUDA_CACHE_PATH="${PROJECT_DIR}/.nv_cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" logs \
         "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_HOME}" \
         "${TORCH_EXTENSIONS_DIR}" "${CUDA_CACHE_PATH}"

# TRAIN_BS * ACCUM must stay 32 -- the driver refuses otherwise
TRAIN_BS="${TRAIN_BS:-16}"; ACCUM="${ACCUM:-2}"; EVAL_BS="${EVAL_BS:-16}"
export GRAD_CKPT="${GRAD_CKPT:-0}"

# folds are short enough to redo, so scratch stays node-local and fast
export FOLD_SCRATCH="${TMPDIR:-/tmp}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${FOLD_SCRATCH}"
trap 'rm -rf "${FOLD_SCRATCH}"' EXIT

echo "=== mediphi_4b QLoRA | ${DATASET} | ${CONFIG} | $(hostname) | $(date) ==="
echo "batch: train=${TRAIN_BS} x accum=${ACCUM} = $((TRAIN_BS*ACCUM)) | eval=${EVAL_BS} | grad_ckpt=${GRAD_CKPT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# One PROCESS PER FOLD. On the Gemma 12B runs the fp32 embedding table survived
# `del model` and accumulated 3.77 GiB per fold; letting the OS reclaim at exit
# makes that impossible. Finished folds exit in seconds via the driver's skip.
FOLDS="${FOLDS:-10 21 32 43 04}"
for TAG in ${FOLDS}; do
  echo "---------- mediphi_4b | ${DATASET} | fold ${TAG} | $(date) ----------"
  python "src/${DATASET}/mediphi_4b_peft_train.py" \
      --dataset "${DATASET}" --config "${CONFIG}" --folds "${TAG}" \
      --per_device_train_batch_size "${TRAIN_BS}" \
      --gradient_accumulation_steps "${ACCUM}" \
      --per_device_eval_batch_size "${EVAL_BS}" \
      ${EXTRA_ARGS:-} || { echo "fold ${TAG} FAILED (exit $?)"; exit 1; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
done

echo "=== done | mediphi_4b | ${DATASET} | ${CONFIG} | $(date) ==="
