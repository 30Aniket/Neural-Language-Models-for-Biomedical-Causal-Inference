#!/bin/bash
# ==========================================================================
# biomistral_smoke.slurm -- ONE-FOLD SMOKE TEST for BioMistral-7B.
#
# Runs a single fold on the REAL config (3x L40, DeepSpeed ZeRO-3, bf16, real
# batch size) but capped to ~50 training steps via --smoke, so it finishes in
# minutes and proves the whole pipeline end-to-end: venv, torchrun on 3 GPUs,
# ZeRO-3 init, model load, data + tokenize, a few train steps, eval, predict,
# isotonic calibration, and df_res writing.
#
# Outputs are ISOLATED (cross_val_biomistral_smoke / outputs/biomistral_smoke)
# so this never touches your real results and is ignored by run_evaluation.py.
#
# SUBMIT (from the project root on $SHARE):
#     sbatch slurm/biomistral_smoke.slurm                                   # AILF, fold 01
#     sbatch slurm/biomistral_smoke.slurm Tramadol-related_mortalities 12   # pick dataset+fold
# ==========================================================================
#SBATCH --job-name=bmi_smoke
#SBATCH --output=logs/bmi_smoke_%j.out
#SBATCH --error=logs/bmi_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:3
#SBATCH --time=03:00:00

set -euo pipefail

DATASET="${1:-Analgesics-induced_acute_liver_failure}"
SPLIT="${2:-01}"

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
export NCCL_P2P_DISABLE=1   # P2P is BROKEN on these L40 nodes (first-BROADCAST hangs). MUST stay disabled.
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_CUDA_ARCH_LIST="8.9"   # L40 = Ada / compute_89 (skips arch probing)
# Reduce allocator fragmentation/flushing. expandable_segments is NOT supported on this
# platform, so we cap split size instead (universally supported) to curb the per-step
# "cache flushes" thrashing under memory pressure.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
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
  echo "ERROR: BioMistral-7B not cached. Run 'bash env/prefetch_biomistral.sh' first." >&2
  exit 1
fi

export FOLD_SCRATCH="${TMPDIR:-/tmp}/biomistral_smoke_${SLURM_JOB_ID}"
mkdir -p "${FOLD_SCRATCH}"
trap 'rm -rf "${FOLD_SCRATCH}"' EXIT

echo "==================================================================="
echo "SMOKE TEST | BioMistral-7B | ${DATASET} | fold ${SPLIT} | 3x L40 ZeRO-3"
echo "Job ${SLURM_JOB_ID} | Node $(hostname) | Start $(date)"
echo "==================================================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true
echo ">>> transformers $(python -c 'import transformers;print(transformers.__version__)')  deepspeed $(python -c 'import deepspeed;print(deepspeed.__version__)')"

echo ">>> Launching smoke fold (~50 steps)..."
torchrun --standalone --nnodes=1 --nproc_per_node=3 \
    "src/${DATASET}/biomistral_train.py" --dataset "${DATASET}" --split "${SPLIT}" --smoke

echo "==================================================================="
echo "SMOKE TEST DONE | $(date)"
SMOKE_CSV="${PROJECT_DIR}/dat/${DATASET}/proc/cross_val_biomistral_smoke/df_res${SPLIT}.csv"
if [[ -f "${SMOKE_CSV}" ]]; then
  echo "PASS: predictions written -> ${SMOKE_CSV}"
  echo "----- header + first rows -----"
  head -3 "${SMOKE_CSV}"
  echo "rows: $(($(wc -l < "${SMOKE_CSV}") - 1))"
else
  echo "FAIL: expected predictions not found at ${SMOKE_CSV}" >&2
  exit 1
fi
echo "==================================================================="
