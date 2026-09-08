#!/usr/bin/env bash
# ==========================================================================
# clean_for_5fold.sh  --  RUN ON THE CLUSTER, from the project root:
#     cd $SHARE/$USER/biocausal_cluster
#     bash clean_for_5fold.sh          # dry run: shows what WOULD be deleted
#     bash clean_for_5fold.sh --yes    # actually delete
#
# Removes everything produced BY the jobs (predictions, logs, weights, curves,
# scratch) so you can re-run cleanly at 5-fold. KEEPS everything that is reused
# or must never be regenerated: the model cache, the venvs, compiled extensions,
# the DATA (df_together.csv / split.csv) and all CODE.
# ==========================================================================
set -euo pipefail

APPLY=0
[[ "${1:-}" == "--yes" || "${1:-}" == "-y" ]] && APPLY=1

echo "=================================================================="
if [[ "${APPLY}" -eq 1 ]]; then
  echo " MODE: DELETE  (removing job outputs)"
else
  echo " MODE: DRY RUN  (nothing deleted -- re-run with --yes to apply)"
fi
echo "=================================================================="

# --- things that get DELETED (job-produced, regenerated on 5-fold re-run) ---
# Note: dat/<ds>/proc/cross_val_* matches ALL prediction folders (encoders,
# xgb, biomistral_*, med_llama_*), but NOT df_together.csv / split.csv, which
# sit directly in proc/ and are never touched by this glob.
TARGETS=(
  "outputs"                              # run_logs, train_logs, weights, models
  "logs"                                 # SLURM .out/.err
  "scratch"                              # any leftover per-job scratch
)
# prediction folders (globbed separately so we can show each)
mapfile -t PRED_DIRS < <(ls -d dat/*/proc/cross_val_* 2>/dev/null || true)

echo ""
echo ">>> WILL DELETE:"
FOUND=0
for t in "${TARGETS[@]}"; do
  if [[ -e "$t" ]]; then
    sz=$(du -sh "$t" 2>/dev/null | cut -f1)
    echo "    $t/   (${sz})"
    FOUND=1
  fi
done
for d in "${PRED_DIRS[@]}"; do
  n=$(ls "$d"/df_res*.csv 2>/dev/null | wc -l | tr -d ' ')
  echo "    $d/   (${n} prediction files)"
  FOUND=1
done
[[ "${FOUND}" -eq 0 ]] && echo "    (nothing found -- already clean)"

echo ""
echo ">>> WILL KEEP (reused / never regenerated):"
for k in .hf_cache .cache venv_biomistral venv_modernbert venv_xgb \
         src slurm env transfer data eval_lora_configs.py; do
  [[ -e "$k" ]] && echo "    $k"
done
echo "    dat/*/proc/df_together.csv   (the data)"
echo "    dat/*/proc/split.csv         (the CV split assignment)"

if [[ "${APPLY}" -eq 1 ]]; then
  echo ""
  echo ">>> Deleting..."
  for t in "${TARGETS[@]}"; do
    [[ -e "$t" ]] && rm -rf "$t" && echo "    removed $t/"
  done
  for d in "${PRED_DIRS[@]}"; do
    rm -rf "$d" && echo "    removed $d/"
  done
  # sanity: confirm the data survived
  echo ""
  echo ">>> Confirming data + splits survived:"
  for ds in dat/*/proc; do
    for f in df_together.csv split.csv; do
      if [[ -f "$ds/$f" ]]; then echo "    OK  $ds/$f"; else echo "    !! MISSING $ds/$f"; fi
    done
  done
  echo ""
  echo ">>> Clean. Ready for 5-fold re-runs."
else
  echo ""
  echo ">>> Dry run only. Re-run with:  bash clean_for_5fold.sh --yes"
fi
echo "=================================================================="
