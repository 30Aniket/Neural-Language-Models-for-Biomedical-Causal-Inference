#!/bin/bash
# gemma4_12b QLoRA, true 5-fold on one L40. Resumable at fold AND mid-fold level.
#   sbatch slurm/gemma4_12b_peft.slurm <dataset> <config>
# Chain several with --dependency=singleton (see submit_gemma_large.sh) so a
# 48h walltime kill is picked up automatically by the next job in the queue.
#SBATCH --job-name=gemma4_12b
#SBATCH --output=logs/gemma4_12b_%j.out
#SBATCH --error=logs/gemma4_12b_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=48:00:00
set -euo pipefail

DATASET="${1:?Usage: sbatch slurm/gemma4_12b_peft.slurm <dataset> [config]}"
CONFIG="${2:-paper}"
MODEL_NAME="gemma4_12b"
case "${DATASET}" in Analgesics*) DTAG=ailf ;; Tramadol*) DTAG=tram ;; *) DTAG="${DATASET}" ;; esac

PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"; CUDA_MODULE="CUDA/12.4.0"
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv_gemma4"
cd "${PROJECT_DIR}"
module purge; module load "${PYTHON_MODULE}" "${CUDA_MODULE}"
source "${VENV_DIR}/bin/activate"
export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA}}"

export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 BITSANDBYTES_NOWELCOME=1
# fragmentation is what turned a 114 MB allocation into a fatal OOM on 3 Sep
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME="${PROJECT_DIR}/.hf_cache"; export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"; export HF_HUB_OFFLINE=1
# keep JIT/compile caches off the small HOME quota (Errno 122 lesson)
export TRITON_CACHE_DIR="${PROJECT_DIR}/.triton_cache"
export TORCHINDUCTOR_CACHE_DIR="${PROJECT_DIR}/.torchinductor_cache"
export TORCH_HOME="${PROJECT_DIR}/.torch"
export TORCH_EXTENSIONS_DIR="${PROJECT_DIR}/.torch_ext"
export CUDA_CACHE_PATH="${PROJECT_DIR}/.nv_cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" logs \
         "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_HOME}" \
         "${TORCH_EXTENSIONS_DIR}" "${CUDA_CACHE_PATH}"
export HF_TOKEN="${HF_TOKEN:-}"   # gated checkpoint: accept the licence first

# batch geometry -- TRAIN_BS * ACCUM must stay 32 (the driver refuses otherwise)
TRAIN_BS="${TRAIN_BS:-8}"; ACCUM="${ACCUM:-4}"; EVAL_BS="${EVAL_BS:-16}"

# 12B folds are short enough to redo, so scratch stays node-local and fast.
export FOLD_SCRATCH="${TMPDIR:-/tmp}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${FOLD_SCRATCH}"
trap 'rm -rf "${FOLD_SCRATCH}"' EXIT

echo "=== gemma4_12b QLoRA | ${DATASET} | ${CONFIG} | $(hostname) | $(date) ==="
echo "batch: train=${TRAIN_BS} x accum=${ACCUM} = $((TRAIN_BS*ACCUM)) | eval=${EVAL_BS}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# One PROCESS PER FOLD. The fp32 embedding table survives `del model` inside a
# long-lived process (measured: +3.77 GiB per fold on the 12B), so we let the OS
# reclaim it at exit instead. Folds already holding a df_res<tag>.csv exit in
# seconds via the driver's own skip logic.
FOLDS="${FOLDS:-10 21 32 43 04}"
for TAG in ${FOLDS}; do
  echo "---------- ${MODEL_NAME} | ${DATASET} | fold ${TAG} | $(date) ----------"
  python "src/${DATASET}/${MODEL_NAME}_peft_train.py" \
      --dataset "${DATASET}" --config "${CONFIG}" --folds "${TAG}" \
      --per_device_train_batch_size "${TRAIN_BS}" \
      --gradient_accumulation_steps "${ACCUM}" \
      --per_device_eval_batch_size "${EVAL_BS}" \
      ${EXTRA_ARGS:-} || { echo "fold ${TAG} FAILED (exit $?)"; exit 1; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
done

echo "=== done | gemma4_12b | ${DATASET} | ${CONFIG} | $(date) ==="
