#!/usr/bin/env bash
# Queue the large-Gemma r8/paper runs: 3 models x 2 datasets = 6 job streams.
#
#   bash submit_gemma_large.sh probe    # measure memory + s/step first (10 min)
#   bash submit_gemma_large.sh 12b      # Gemma3-12B + Gemma4-12B, both datasets
#   bash submit_gemma_large.sh 31b      # Gemma4-31B, both datasets, chained x3
#   bash submit_gemma_large.sh all      # all six
#
# Each stream gets its own job name (<model>_<dtag>) and is submitted with
# --dependency=singleton, so queueing the same name N times gives you N
# sequential attempts: if one hits the 48h walltime the next starts and picks
# up where it left off. Completed folds are skipped via df_res<tag>.csv, and
# the 31B additionally resumes mid-fold from its last 200-step checkpoint.
#
# Override geometry per invocation, e.g.:
#   TRAIN_BS=8 ACCUM=4 bash submit_gemma_large.sh 31b
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-all}"
CONFIG="${CONFIG:-paper}"
DATASETS=(Analgesics-induced_acute_liver_failure Tramadol-related_mortalities)
mkdir -p logs

# how many chained attempts per stream (each up to 48h)
CHAIN_12B="${CHAIN_12B:-1}"
CHAIN_31B="${CHAIN_31B:-3}"

queue_stream () {   # queue_stream <model> <chain-length>
  local model="$1" chain="$2"
  for D in "${DATASETS[@]}"; do
    local dtag; [[ "$D" == Analg* ]] && dtag=ailf || dtag=tram
    local name="${model}_${dtag}"
    local prev=""
    for ((i = 1; i <= chain; i++)); do
      local jid
      jid=$(sbatch --parsable --job-name="${name}" --dependency=singleton \
              --export=ALL${TRAIN_BS:+,TRAIN_BS=$TRAIN_BS}${ACCUM:+,ACCUM=$ACCUM}${EVAL_BS:+,EVAL_BS=$EVAL_BS} \
              "slurm/${model}_peft.slurm" "$D" "$CONFIG")
      echo "  queued ${name} attempt ${i}/${chain} -> job ${jid}"
      prev="$jid"
    done
  done
}

case "$MODE" in
  probe)
    cat <<'EOF'
Preflight: 20 real training steps on the LONGEST sequences in fold 10, which is
the worst case for dynamic padding. Prints peak GiB and s/step so you can size
the batch before committing 48h allocations.

Grab an interactive GPU:

  srun --partition=gpu --gres=gpu:lovelace_l40:1 --cpus-per-task=8 \
       --mem-per-cpu=5960 --time=01:00:00 --pty bash

Then, in that shell:

  cd "${SHARE:-/springbrook/share/dcsresearch}/$USER/biocausal_cluster"
  module purge && module load Python/3.11.5-GCCcore-13.2.0 CUDA/12.4.0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
  export HF_HOME="$PWD/.hf_cache" XDG_CACHE_HOME="$PWD/.cache" HF_TOKEN=...
  export TRITON_CACHE_DIR="$PWD/.triton_cache" TORCH_HOME="$PWD/.torch"
  D=Analgesics-induced_acute_liver_failure

  # Gemma4-31B is the one worth probing -- 4x8 vs 8x4 is the open question
  source venv_gemma4/bin/activate
  python src/$D/gemma4_31b_peft_train.py --dataset $D --probe 20 \
      --per_device_train_batch_size 4 --gradient_accumulation_steps 8
  python src/$D/gemma4_31b_peft_train.py --dataset $D --probe 20 \
      --per_device_train_batch_size 8 --gradient_accumulation_steps 4

  # and a quick sanity pass on one 12B
  python src/$D/gemma4_12b_peft_train.py --dataset $D --probe 20

  # Gemma3-12B lives in the other venv
  deactivate && source venv_gemma/bin/activate
  python src/$D/gemma3_12b_peft_train.py --dataset $D --probe 20

Read the two PROBE lines at the end of each: take the largest batch that leaves
at least ~4 GiB of headroom, then pass it through as TRAIN_BS/ACCUM below.
EOF
    exit 0
    ;;
  12b)
    echo "queueing the four 12B streams (chain=${CHAIN_12B})"
    queue_stream gemma3_12b "$CHAIN_12B"
    queue_stream gemma4_12b "$CHAIN_12B"
    ;;
  31b)
    echo "queueing the two 31B streams (chain=${CHAIN_31B})"
    queue_stream gemma4_31b "$CHAIN_31B"
    ;;
  all)
    echo "queueing all six streams"
    queue_stream gemma3_12b "$CHAIN_12B"
    queue_stream gemma4_12b "$CHAIN_12B"
    queue_stream gemma4_31b "$CHAIN_31B"
    ;;
  *)
    echo "usage: bash submit_gemma_large.sh [probe|12b|31b|all]" >&2
    exit 2
    ;;
esac

cat <<'EOF'

watch:     squeue -u $USER -o '%.10i %.20j %.2t %.10M %.10L %R'
progress:  grep -h 'fold .* wall time\|\[mem\]' logs/gemma*_*.out | tail -20
folds:     for d in dat/*/proc/cross_val_gemma*_paper_temp; do \
             echo "$(ls $d | wc -l)/5  $d"; done
EOF
