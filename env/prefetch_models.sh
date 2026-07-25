#!/usr/bin/env bash
# ==========================================================================
# prefetch_models.sh  --  RUN ONCE ON THE LOGIN NODE, BEFORE SUBMITTING JOBS
#
# WHY THIS EXISTS
# ---------------
# When two training jobs start simultaneously and both try to download the same
# checkpoint into the same shared HF cache on $SHARE, they race: one process
# renames the ".incomplete" blob into place while the other still expects it,
# producing:
#     FileNotFoundError: ...blobs/<hash>.incomplete
# Downloading once here, serially, removes the race. The SLURM jobs then run
# with HF_HUB_OFFLINE=1 and simply read the warm cache.
#
# Also avoids every job re-downloading the same weights, and works even if the
# compute nodes have no outbound internet access.
#
# Safe to re-run: already-cached models are skipped instantly.
#
# USAGE:
#     cd $SHARE/$USER/biocausal_cluster
#     bash env/prefetch_models.sh
# ==========================================================================
set -euo pipefail

PYTHON_MODULE="Python/3.11.5-GCCcore-13.2.0"     # must match setup_env.sh / *.slurm
SHARE_DIR="${SHARE:-/springbrook/share/dcsresearch}"
PROJECT_DIR="${PROJECT_DIR:-${SHARE_DIR}/${USER}/biocausal_cluster}"
VENV_DIR="${PROJECT_DIR}/venv"

# Models to warm the cache with. Add future checkpoints here as you reach them
# (PubMedBERT, BioClinical ModernBERT, BioMistral, Med-LLaMA...).
MODELS=(
  "textattack/albert-base-v2-imdb"           # ALBERT   ~12M params
  "dmis-lab/biobert-base-cased-v1.1"         # BioBERT ~110M params
  "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"   # PubMedBERT ~110M params
)

echo ">>> Project dir : ${PROJECT_DIR}"

module purge || true          # 'slurm' module refuses to unload; that is fine
module load "${PYTHON_MODULE}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export HF_HOME="${PROJECT_DIR}/.hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"
unset HF_HUB_OFFLINE          # we DO want network access here
unset TRANSFORMERS_CACHE      # deprecated; HF_HOME governs

echo ">>> HF_HOME     : ${HF_HOME}"
echo ">>> Downloading ${#MODELS[@]} model(s) serially..."

# The Python snippets below are kept in a temp file so this script has no
# nested heredocs (which are fragile to edit).
PYFILE="$(mktemp)"
cat > "${PYFILE}" <<'PYEOF'
import os, sys
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
mid = os.environ["MODEL_ID"]
AutoConfig.from_pretrained(mid)
AutoTokenizer.from_pretrained(mid)
m = AutoModelForSequenceClassification.from_pretrained(
        mid, num_labels=2, ignore_mismatched_sizes=True)
mode = "offline load OK" if os.environ.get("HF_HUB_OFFLINE") == "1" else "cached OK"
print(f"  {mode}: {mid}  ({sum(p.numel() for p in m.parameters()):,} params)")
PYEOF
trap 'rm -f "${PYFILE}"' EXIT

for m in "${MODELS[@]}"; do
  echo ""
  echo "--- ${m} ---"
  MODEL_ID="${m}" python "${PYFILE}"
done

echo ""
echo ">>> Verifying every model can be read OFFLINE (exactly as the jobs will)..."
for m in "${MODELS[@]}"; do
  HF_HUB_OFFLINE=1 MODEL_ID="${m}" python "${PYFILE}"
done

echo ""
echo ">>> Cache size: $(du -sh "${HF_HOME}" 2>/dev/null | cut -f1)"
echo ">>> DONE. You can now submit jobs, e.g.:"
echo "      bash submit_all.sh <your-email> albert"
echo "      bash submit_all.sh <your-email> biobert"
