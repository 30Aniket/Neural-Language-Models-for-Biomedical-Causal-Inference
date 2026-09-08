"""
BioMistral-7B full fine-tuning driver -- ONE cross-validation fold per invocation.

WHY ONE FOLD PER PROCESS (vs the encoders' 20-splits-in-a-loop)
--------------------------------------------------------------
Full fine-tuning a 7B model under DeepSpeed ZeRO-3 uses all three L40s and takes a
few hours per fold, so 20 folds cannot fit inside the 48h GPU walltime, and re-
initialising DeepSpeed 20 times inside one process is fragile. Instead each fold is
run as a *fresh process*, dispatched as one task of a SLURM job array. A fresh
process means a freshly-loaded pretrained model on every fold => zero cross-fold
weight leakage BY CONSTRUCTION (the strongest possible form of the CV fix; there is
no shared model object to leak).

WHAT IS IDENTICAL TO THE ENCODER PIPELINE
-----------------------------------------
The per-fold logic -- get_split -> lowercase+tokenize(max_len 128) -> train ->
predict(softmax) -> isotonic calibrate -> write df_res -- is the same as
modernbert_train.py. The df_res columns are `biomistral_temp` / `biomistral_temp_cal`
/ `label`, so run_evaluation.py discovers this model automatically from the
`cross_val_biomistral_temp` folder.

WHAT CHANGED (only what a 7B decoder under ZeRO-3 forces)
--------------------------------------------------------
  * DeepSpeed ZeRO-3 (bf16, no offload); optimiser/schedule declared in
    env/deepspeed_zero3.json with the encoders' exact hyper-parameters (see
    biomistral_train_src.py header).
  * Decoder-as-classifier: BioMistral has no pad token, so we set
    pad_token = eos_token and model.config.pad_token_id; use_cache=False for
    gradient checkpointing.
  * Launched with torchrun (3 procs, one per L40). Only rank 0 writes df_res /
    training-curve / the persistent run-log.
  * NO per-fold checkpoint saving (predictions-only run). The single best fold is
    re-trained afterwards with --save_final (see slurm/biomistral_best.slurm).

Usage (normally via slurm/biomistral.slurm):
  torchrun --standalone --nnodes=1 --nproc_per_node=3 \
      src/<dataset>/biomistral_train.py --dataset <dataset> --split <ij> [--save_final DIR]
"""
import os
import sys
import random
import logging
import datetime
import argparse

import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import biomistral_train_src

os.environ.setdefault("WANDB_DISABLED", "true")

MODEL_ID = "BioMistral/BioMistral-7B"
MODEL_NAME = "biomistral"
DEFAULT_DATASET = "Tramadol-related_mortalities"

# torchrun sets RANK/LOCAL_RANK/WORLD_SIZE; default to single-process for a bare run.
RANK = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
IS_MAIN = (RANK == 0)

parser = argparse.ArgumentParser(description="BioMistral-7B single-fold CV training (DeepSpeed ZeRO-3).")
parser.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="Dataset folder name under dat/ (default: %(default)s)")
parser.add_argument("--split", required=True,
                    help="Two-digit fold tag <dev><test>, e.g. 01 -> dev=0, test=1, train={2,3,4}")
parser.add_argument("--save_final", default=None,
                    help="If set, save the trained (best) model + tokenizer here. Used ONLY by the "
                         "retrain-the-winner step; the 20-fold array leaves this unset (predictions-only).")
parser.add_argument("--per_device_train_batch_size", type=int, default=1)   # 7B: 1/GPU to cut memory pressure (was 2)
parser.add_argument("--per_device_eval_batch_size", type=int, default=1)   # eval also memory-bound (had 4302 cache flushes); 1/GPU
parser.add_argument("--gradient_accumulation_steps", type=int, default=12)  # 1x3x12=36 effective (unchanged)
parser.add_argument("--smoke", action="store_true",
                    help="Fast end-to-end test: ~50 train steps, isolated output folders "
                         "(cross_val_biomistral_smoke / outputs/biomistral_smoke). Real-run "
                         "logic is untouched when this flag is absent.")
parser.add_argument("--max_steps", type=int, default=None, help="Override max_steps (advanced/testing).")
parser.add_argument("--warmup_steps", type=int, default=None, help="Override warmup_steps (advanced/testing).")
parser.add_argument("--eval_steps", type=int, default=None, help="Override eval/save/logging steps (advanced/testing).")
args, _ = parser.parse_known_args()
DATASET = args.dataset

# Faithful defaults; --smoke shrinks them for a quick pipeline check.
if args.smoke:
    # speed probe: 15 steps, eval only at the very end (step 15) so the expensive
    # eval pass doesn't dominate; enough to read a clean s/it and confirm the pipeline.
    MAX_STEPS, WARMUP_STEPS, EVAL_STEPS = 15, 5, 15
else:
    MAX_STEPS = args.max_steps if args.max_steps else 30000
    WARMUP_STEPS = args.warmup_steps if args.warmup_steps else 200
    EVAL_STEPS = args.eval_steps if args.eval_steps else 200

# Smoke runs write to ISOLATED folders so they never mix with real predictions and
# are never picked up by run_evaluation.py (which globs cross_val_*_temp).
RUN_KIND = "biomistral_smoke" if args.smoke else "biomistral"
CV_FOLDER = "cross_val_biomistral_smoke" if args.smoke else "cross_val_biomistral_temp"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_BASE = os.path.join(PROJECT_ROOT, "outputs", RUN_KIND, DATASET)
TRAINLOG_DIR = os.path.join(OUT_BASE, "train_logs")
RUNLOG_DIR = os.path.join(OUT_BASE, "run_logs")
DS_CONFIG = os.path.join(PROJECT_ROOT, "env", "deepspeed_zero3.json")
if IS_MAIN:
    for _d in (TRAINLOG_DIR, RUNLOG_DIR):
        os.makedirs(_d, exist_ok=True)

# ---- logging: console on every rank, persistent file on rank 0 only -------
RUN_TAG = os.environ.get("SLURM_JOB_ID") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_handlers = [logging.StreamHandler(sys.stdout)]
if IS_MAIN:
    _handlers.append(logging.FileHandler(
        os.path.join(RUNLOG_DIR, f"biomistral_{DATASET}_{args.split}_{RUN_TAG}.log")))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=_handlers)
logger = logging.getLogger("biomistral_train")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---- reconstruct the split dict from the two-digit tag --------------------
dev_i, test_j = int(args.split[0]), int(args.split[1])
assert dev_i != test_j, f"invalid split tag '{args.split}' (dev and test must differ)"
split = {"dev": [dev_i], "test": [test_j], "train": [x for x in range(5) if x != dev_i and x != test_j]}
split_tag = f'{split["dev"][0]}{split["test"][0]}'

if IS_MAIN:
    logger.info("=== BioMistral-7B training run started ===")
    logger.info(f"dataset={DATASET}  split={split_tag}  dev={split['dev']} test={split['test']} train={split['train']}")
    logger.info(f"world_size={os.environ.get('WORLD_SIZE','1')}  save_final={args.save_final}")

# Set a global seed (same as the encoder pipeline; fresh process => fresh model).
set_seed(42)

# ---- tokenizer + model (fresh load per process => no cross-fold leakage) ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:                       # Mistral ships no pad token
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=2,
    attn_implementation="eager",   # avoid cuDNN SDPA backward (produced NaN grads on L40 + bf16 + grad-ckpt)
)
model.config.pad_token_id = tokenizer.pad_token_id    # decoder seq-cls needs a pad id
model.config.use_cache = False                        # required with gradient checkpointing
if IS_MAIN:
    logger.info(f"model loaded: {MODEL_ID} "
                f"({sum(p.numel() for p in model.parameters()):,} parameters, all trainable / full FT)")

# ---- data (same columns + preprocessing as the encoders) ------------------
df_train, df_dev, df_test = biomistral_train_src.get_split(split, DATASET)
train_data = Dataset.from_pandas(df_train)
dev_data = Dataset.from_pandas(df_dev)
test_data = Dataset.from_pandas(df_test)


def preprocess_function(examples):
    examples["Temp_sentence"] = [sentence.lower() for sentence in examples["Temp_sentence"]]
    return tokenizer(examples["Temp_sentence"], truncation=True, max_length=128)


tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_dev = dev_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

# ---- train under DeepSpeed ZeRO-3 -----------------------------------------
# Transient DeepSpeed checkpoints (large) go to node-local SSD scratch, NOT $SHARE.
scratch_out = os.environ.get(
    "FOLD_SCRATCH", os.path.join(PROJECT_ROOT, "scratch", f"{DATASET}_{split_tag}_{RUN_TAG}"))
if IS_MAIN:
    os.makedirs(scratch_out, exist_ok=True)

trainer = biomistral_train_src.train_transformer(
    tokenizer, model, tokenized_train, tokenized_dev,
    deepspeed_config=DS_CONFIG, output_dir=scratch_out,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    max_steps=MAX_STEPS, warmup_steps=WARMUP_STEPS,
    eval_steps=EVAL_STEPS, save_steps=EVAL_STEPS,
)

# ---- predict + isotonic calibration (identical to encoders) ---------------
prob_dev, prob_test = biomistral_train_src.predict(trainer, tokenized_dev, tokenized_test)
prob_test_cal = biomistral_train_src.calibration(prob_dev, df_dev["label"].tolist(), prob_test)

# ---- persist outputs (rank 0 only) ----------------------------------------
if trainer.is_world_process_zero():
    # 1) training curve (loss / eval_loss / lr per logging step)
    try:
        log_hist = pd.DataFrame(trainer.state.log_history)
        log_hist.to_csv(os.path.join(TRAINLOG_DIR, f"train_curve_{split_tag}.csv"), index=False)
        logger.info(f"[split {split_tag}] saved training curve "
                    f"({len(log_hist)} rows, best_metric={getattr(trainer.state, 'best_metric', None)})")
    except Exception as e:
        logger.warning(f"[split {split_tag}] could not save training curve: {e}")

    # 2) predictions -> dat/<dataset>/proc/cross_val_biomistral_temp/df_res<ij>.csv
    df_res = pd.DataFrame(index=df_test.index)
    df_res["biomistral_temp"] = prob_test
    df_res["biomistral_temp_cal"] = prob_test_cal
    df_res["label"] = df_test["label"]
    out_dir = os.path.join(PROJECT_ROOT, "dat", DATASET, "proc", CV_FOLDER)
    os.makedirs(out_dir, exist_ok=True)
    df_res.to_csv(os.path.join(out_dir, f"df_res{split_tag}.csv"))
    logger.info(f"[split {split_tag}] wrote predictions -> {os.path.join(out_dir, f'df_res{split_tag}.csv')}")

# ---- optional: save the winning fold's model (retrain-the-winner step) ----
if args.save_final:
    # trainer.save_model gathers the ZeRO-3-sharded weights into a full checkpoint
    # (stage3_gather_16bit_weights_on_model_save=true in the DeepSpeed config).
    trainer.save_model(args.save_final)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.save_final)
        logger.info(f"[split {split_tag}] saved FINAL best-fold model -> {args.save_final}")

if IS_MAIN:
    logger.info("=== BioMistral-7B training run finished ===")
