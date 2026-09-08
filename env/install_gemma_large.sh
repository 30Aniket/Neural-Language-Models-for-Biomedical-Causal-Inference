#!/usr/bin/env bash
# Install the rewritten large-Gemma drivers + SLURM scripts.
# Run from the biocausal_cluster root, with this bundle unpacked alongside it:
#
#   bash install_gemma_large.sh
#
# Existing files are backed up to *.bak_<timestamp> before being replaced.
# Nothing outside the three Gemma models is touched -- the encoders, XGBoost,
# BioMistral, Med-LLaMA, Gemma3-4B, MedGemma and Qwen runs are untouched, and
# the output paths and column names are identical, so run_evaluation.py needs
# no changes.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
STAMP="$(date +%Y%m%d_%H%M%S)"

DATASETS=(Analgesics-induced_acute_liver_failure Tramadol-related_mortalities)
DRIVERS=(gemma_large_peft_src.py gemma3_12b_peft_train.py
         gemma4_12b_peft_train.py gemma4_31b_peft_train.py
         medgemma_4b_peft_train.py medgemma_27b_peft_train.py
         mediphi_4b_peft_train.py phi4_14b_peft_train.py)
SLURMS=(gemma3_12b_peft.slurm gemma4_12b_peft.slurm gemma4_31b_peft.slurm
        medgemma_4b_peft.slurm medgemma_27b_peft.slurm
        mediphi_4b_peft.slurm phi4_14b_peft.slurm)

[[ -d "${PROJECT_DIR}/src" && -d "${PROJECT_DIR}/dat" ]] || {
  echo "error: ${PROJECT_DIR} does not look like biocausal_cluster (no src/ and dat/)" >&2
  exit 1
}

install_one () {   # install_one <source> <destination>
  local s="$1" d="$2"
  if [[ -f "$d" ]] && ! cmp -s "$s" "$d"; then
    cp -p "$d" "${d}.bak_${STAMP}"
    echo "  backed up $(basename "$d") -> $(basename "$d").bak_${STAMP}"
  fi
  install -m 0644 "$s" "$d"
  echo "  installed $d"
}

echo "installing drivers into both dataset source trees"
for D in "${DATASETS[@]}"; do
  mkdir -p "${PROJECT_DIR}/src/${D}"
  for F in "${DRIVERS[@]}"; do
    install_one "${HERE}/src/${F}" "${PROJECT_DIR}/src/${D}/${F}"
  done
done

echo "installing SLURM scripts"
mkdir -p "${PROJECT_DIR}/slurm" "${PROJECT_DIR}/logs"
for F in "${SLURMS[@]}"; do
  install_one "${HERE}/slurm/${F}" "${PROJECT_DIR}/slurm/${F}"
done

for F in submit_gemma_large.sh submit_medgemma.sh submit_phi.sh cleanup_gemma_large.sh; do
  install -m 0755 "${HERE}/${F}" "${PROJECT_DIR}/${F}"
  echo "  installed ${PROJECT_DIR}/${F}"
done

echo
echo "syntax check"
for D in "${DATASETS[@]}"; do
  ( cd "${PROJECT_DIR}/src/${D}" && python3 -m py_compile "${DRIVERS[@]}" ) \
    && echo "  python OK: src/${D}"
done
for F in "${SLURMS[@]}"; do
  bash -n "${PROJECT_DIR}/slurm/${F}" && echo "  bash OK: slurm/${F}"
done
for F in submit_gemma_large.sh submit_medgemma.sh submit_phi.sh cleanup_gemma_large.sh; do
  bash -n "${PROJECT_DIR}/${F}" && echo "  bash OK: ${F}"
done

echo
echo "done."
echo "  inspect : bash cleanup_gemma_large.sh status"
echo "  preflight: bash submit_gemma_large.sh probe"
echo "  queue    : bash submit_gemma_large.sh all"
