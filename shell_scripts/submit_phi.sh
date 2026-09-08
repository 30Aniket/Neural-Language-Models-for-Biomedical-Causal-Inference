#!/usr/bin/env bash
# Queue the Microsoft Phi r8 runs: 2 models x 2 datasets = 4 jobs.
#
#   bash submit_phi.sh prefetch   # print the download commands (run these FIRST)
#   bash submit_phi.sh probe      # 20 real steps, peak memory + s/step
#   bash submit_phi.sh 4b         # microsoft/MediPhi,  both datasets
#   bash submit_phi.sh 14b        # microsoft/phi-4,    both datasets
#   bash submit_phi.sh all        # all four
#
# Geometry differs per model on purpose, to match each one's comparison target:
#   MediPhi 3.8B = 16 x 2  -> same as MedGemma-1.5-4B / Gemma3-4B
#   phi-4 ~14.7B = 8 x 4   -> same as Gemma4-12B
# Effective batch is 32 in both cases, so everything pools with the suite.
#
# Neither model is gated. No HF_TOKEN, no licence acceptance.
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-all}"
CONFIG="${CONFIG:-paper}"
DATASETS=(Analgesics-induced_acute_liver_failure Tramadol-related_mortalities)
mkdir -p logs

queue_stream () {   # queue_stream <model>
  local model="$1"
  for D in "${DATASETS[@]}"; do
    local dtag; [[ "$D" == Analg* ]] && dtag=ailf || dtag=tram
    local xp="ALL"
    [[ -n "${TRAIN_BS:-}"  ]] && xp="${xp},TRAIN_BS=${TRAIN_BS}"
    [[ -n "${ACCUM:-}"     ]] && xp="${xp},ACCUM=${ACCUM}"
    [[ -n "${EVAL_BS:-}"   ]] && xp="${xp},EVAL_BS=${EVAL_BS}"
    [[ -n "${GRAD_CKPT:-}" ]] && xp="${xp},GRAD_CKPT=${GRAD_CKPT}"
    [[ -n "${VENV:-}"      ]] && xp="${xp},VENV=${VENV}"
    local jid
    jid=$(sbatch --parsable --job-name="${model}_${dtag}" --dependency=singleton \
                 --export="${xp}" "slurm/${model}_peft.slurm" "$D" "$CONFIG")
    echo "  queued ${model}_${dtag} -> job ${jid}"
  done
}

case "$MODE" in
  prefetch)
    cat <<'EOF'
Run on the LOGIN node before submitting (jobs run with HF_HUB_OFFLINE=1):

  cd "${SHARE:-/springbrook/share/dcsresearch}/$USER/biocausal_cluster"
  module purge && module load Python/3.11.5-GCCcore-13.2.0
  source venv_gemma4/bin/activate
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
for m in ("microsoft/MediPhi", "microsoft/phi-4"):
    p = snapshot_download(m, cache_dir=os.path.join(os.getcwd(), ".hf_cache"))
    print("cached:", m, "->", p)
PY

Roughly 8 GB for MediPhi and 29 GB for phi-4. Neither is gated.
EOF
    exit 0
    ;;
  probe)
    cat <<'EOF'
20 real training steps on the LONGEST sequences in fold 10 -- worst case for
dynamic padding. Confirms peak memory and s/step before committing.

  srun --partition=gpu --gres=gpu:lovelace_l40:1 --cpus-per-task=8 \
       --mem-per-cpu=5960 --time=01:00:00 --pty bash

  cd "${SHARE:-/springbrook/share/dcsresearch}/$USER/biocausal_cluster"
  module purge && module load Python/3.11.5-GCCcore-13.2.0 CUDA/12.4.0
  source venv_gemma4/bin/activate
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
  export HF_HOME="$PWD/.hf_cache" XDG_CACHE_HOME="$PWD/.cache"
  export TRITON_CACHE_DIR="$PWD/.triton_cache" TORCH_HOME="$PWD/.torch"
  D=Analgesics-induced_acute_liver_failure

  python src/$D/mediphi_4b_peft_train.py --dataset $D --probe 20
  python src/$D/phi4_14b_peft_train.py   --dataset $D --probe 20

Check two things in the output:
  1. a "FUSED ATTENTION" warning -- confirms the qkv_proj fallback fired
  2. the PROBE peak line -- if phi-4 leaves under ~4 GiB headroom, resubmit
     that stream with GRAD_CKPT=1
EOF
    exit 0
    ;;
  4b)  echo "queueing microsoft/MediPhi (both datasets)"; queue_stream mediphi_4b ;;
  14b) echo "queueing microsoft/phi-4 (both datasets)";   queue_stream phi4_14b ;;
  all)
    echo "queueing all four Phi streams"
    queue_stream mediphi_4b
    queue_stream phi4_14b
    ;;
  *) echo "usage: bash submit_phi.sh [prefetch|probe|4b|14b|all]" >&2; exit 2 ;;
esac

cat <<'EOF'

watch:    squeue -u $USER -o '%.10i %.20j %.2t %.10M %.10L %R'
params:   grep -h "trainable params" logs/mediphi*.out logs/phi4*.out
targets:  grep -h "FUSED ATTENTION" logs/mediphi*.out logs/phi4*.out
memory:   grep -h "\[mem\] fold .* pre-build" logs/mediphi*.out logs/phi4*.out
folds:    bash cleanup_gemma_large.sh status
EOF
