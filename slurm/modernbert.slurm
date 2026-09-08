#!/bin/bash
# ==========================================================================
# modernbert.slurm -- BioClinical ModernBERT full fine-tuning on Blythe.
#
# Standard encoder (~150M params), full fine-tuning (no PEFT/quantization),
# single L40 - same shape as the ALBERT/BioBERT/PubMedBERT jobs.
#
# THE ONE DIFFERENCE: it activates venv_modernbert (transformers >= 4.48),
# because ModernBERT does not load under the main venv's transformers 4.41.2.
#
# SUBMIT (from the project root on $SHARE):
#     sbatch slurm/modernbert.slurm Analgesics-induced_acute_liver_failure
#     sbatch slurm/modernbert.slurm Tramadol-related_mortalities
# ==========================================================================
#SBATCH --job-name=modernbert
#SBATCH --output=logs/modernbert-%j.out
#SBATCH --error=logs/modernbert-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5960
#SBATCH --gres=gpu:lovelace_l40:1
#SBATCH --time=24:00:00

set -euo pipefail

PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv_modernbert"        # <-- the SEPARATE venv

DATASET="${1:-Analgesics-induced_acute_liver_failure}"

echo "==================================================================="
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : $(hostname)"
echo "Model    : BioClinical ModernBERT (thomas-sounack/BioClinical-ModernBERT-base)"
echo "Dataset  : ${DATASET}"
echo "Venv     : ${VENV_DIR}  (transformers >= 4.48)"
echo "Start    : $(date)"
echo "==================================================================="

cd "${PROJECT_DIR}"

module purge
module load "${PYTHON_MODULE}"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "ERROR: ${VENV_DIR} not found. Run 'bash env/setup_env_modernbert.sh' first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export HF_HOME="${PROJECT_DIR}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${PROJECT_DIR}/.cache"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}"
export HF_HUB_OFFLINE=1

if ! find "${HF_HOME}" -maxdepth 3 -type d -name "models--thomas-sounack--BioClinical-ModernBERT-base" -not -path "*/.locks/*" 2>/dev/null | grep -q .; then
  echo "ERROR: BioClinical-ModernBERT-base not cached under ${HF_HOME}." >&2
  echo "       Run 'bash env/prefetch_models.sh' on the login node first." >&2
  exit 1
fi
echo ">>> Cached checkpoint: $(find "${HF_HOME}" -maxdepth 3 -type d -name 'models--thomas-sounack--BioClinical-ModernBERT-base' -not -path '*/.locks/*' | head -n1)"

export PYTHONUNBUFFERED=1
echo ">>> GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo ">>> python: $(which python)  transformers $(python -c 'import transformers;print(transformers.__version__)')"

# Per-job scratch so results/ can't collide with a concurrent job.
SCRATCH="${PROJECT_DIR}/scratch/${SLURM_JOB_ID}"
mkdir -p "${SCRATCH}"
cd "${SCRATCH}"

echo ">>> Launching BioClinical ModernBERT training..."
srun python "${PROJECT_DIR}/src/${DATASET}/modernbert_train.py" --dataset "${DATASET}"

echo "==================================================================="
echo "End      : $(date)"
echo "Weights  : ${PROJECT_DIR}/outputs/modernbert/${DATASET}/weights/"
echo "Preds    : ${PROJECT_DIR}/dat/${DATASET}/proc/cross_val_modernbert_temp/"
echo "==================================================================="

cd "${PROJECT_DIR}" && rm -rf "${SCRATCH}"
