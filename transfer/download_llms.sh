#!/usr/bin/env bash
# ==========================================================================
# download_llms.sh  --  RUN ON YOUR LOCAL MAC
#
# Pulls back EVERYTHING for the two biomedical LLMs -- BioMistral-7B and
# Med-LLaMA-8B -- across ALL LoRA configs, from your Blythe SHARE directory.
#
# For each model/config it brings back:
#   * dat/<dataset>/proc/cross_val_<model>_<config>_temp/df_res*.csv   (predictions)
#   * outputs/<model>_<config>/<dataset>/run_logs/*.log               (full run logs)
#   * outputs/<model>_<config>/<dataset>/train_logs/*.csv             (training curves)
#   * logs/bmp_*.out/.err  and  logs/mlm_*.out/.err                   (SLURM logs)
#
# NOTE: the PEFT jobs save LoRA adapter checkpoints to node-local /tmp scratch,
# which is deleted when each job ends -- so there are NO weight files to download
# (this is expected, not an error). If you ever want to keep adapters, we'd add a
# copy-to-SHARE step in the driver; for evaluation you only need the df_res CSVs.
#
# Usage:
#     bash download_llms.sh
#     SSH_KEY=~/.ssh/id_rsa bash download_llms.sh
#     DEST=/Volumes/ExtDisk/biocausal_llms bash download_llms.sh
# ==========================================================================
set -euo pipefail

USER_NAME="${USER_NAME:-cstqrd}"
DATA_NODE="blythedata.scrtp.warwick.ac.uk"
REMOTE="/springbrook/share/dcsresearch/${USER_NAME}/biocausal_cluster"
DEST="${DEST:-$(pwd)/biocausal_llm_results}"
SSH_KEY="${SSH_KEY:-}"

RSH="ssh"
[[ -n "${SSH_KEY}" ]] && RSH="ssh -i ${SSH_KEY}"

echo "=================================================================="
echo " Source : ${USER_NAME}@${DATA_NODE}:${REMOTE}"
echo " Dest   : ${DEST}"
echo "=================================================================="
mkdir -p "${DEST}"

# --- 1. PREDICTIONS: all cross_val_biomistral_*_temp and cross_val_med_llama_*_temp ---
# (-R with /./ pivot recreates dat/<dataset>/proc/... under DEST)
echo ""
echo ">>> [1/4] Prediction CSVs (df_res*.csv) for every BioMistral + Med-LLaMA config..."
rsync -avzP -R -e "${RSH}" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./dat/*/proc/cross_val_biomistral_*_temp" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./dat/*/proc/cross_val_med_llama_*_temp" \
    "${DEST}/" 2>/dev/null || echo "   (some config folders may not exist yet - that's fine)"

# --- 2. OUTPUTS: run logs + training curves for every config (NO weights - none exist) ---
echo ""
echo ">>> [2/4] Run logs + training curves (outputs/<model>_<config>/...)..."
rsync -avzP -R -e "${RSH}" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./outputs/biomistral_"* \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./outputs/med_llama_"* \
    "${DEST}/" 2>/dev/null || echo "   (no outputs/ dirs yet for some configs - fine)"

# --- 3. SLURM logs: bmp_* (BioMistral) and mlm_* (Med-LLaMA) .out/.err ---
echo ""
echo ">>> [3/4] SLURM .out/.err logs (bmp_* and mlm_*)..."
mkdir -p "${DEST}/logs"
rsync -avzP -e "${RSH}" \
    --include 'bmp_*' --include 'mlm_*' --exclude '*' \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/logs/" \
    "${DEST}/logs/" 2>/dev/null || echo "   (no matching SLURM logs - fine)"

# --- 4. the driver code that produced it all (tiny, self-documenting) ---
echo ""
echo ">>> [4/4] LLM driver code (src/*/biomistral_peft* and med_llama_peft*)..."
rsync -avzP -R -e "${RSH}" \
    --include '*/' \
    --include 'biomistral_peft*' --include 'med_llama_peft*' \
    --include '*biomistral_peft*.slurm' --include '*med_llama_peft*.slurm' \
    --exclude '*' \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./src" \
    "${USER_NAME}@${DATA_NODE}:${REMOTE}/./slurm" \
    "${DEST}/" 2>/dev/null || true

# ---- verification --------------------------------------------------------
echo ""
echo "=================================================================="
echo " VERIFICATION"
echo "=================================================================="
N_P=$(find "${DEST}/dat" -name 'df_res*.csv' 2>/dev/null | wc -l | tr -d ' ')
N_L=$(find "${DEST}/outputs" -name '*.log' 2>/dev/null | wc -l | tr -d ' ')
N_C=$(find "${DEST}/outputs" -name 'train_curve_*.csv' 2>/dev/null | wc -l | tr -d ' ')
echo "  prediction CSVs (df_res*) : ${N_P}"
echo "  run logs (*.log)          : ${N_L}"
echo "  training curves           : ${N_C}"
echo ""
echo "  per-config prediction counts:"
for model in biomistral med_llama; do
  for cfg in paper r16 r64 r128 r256; do
    c=$(find "${DEST}/dat" -path "*cross_val_${model}_${cfg}_temp*" -name 'df_res*.csv' 2>/dev/null | wc -l | tr -d ' ')
    [[ "${c}" != "0" ]] && echo "     ${model}_${cfg}: ${c} folds"
  done
done
echo ""
echo "  total size : $(du -sh "${DEST}" 2>/dev/null | cut -f1)"
echo "=================================================================="
echo ">>> Done. Predictions are under ${DEST}/dat/<dataset>/proc/cross_val_<model>_<config>_temp/"
