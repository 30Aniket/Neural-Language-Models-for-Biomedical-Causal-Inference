"""
Med-LLaMA-8B PEFT (QLoRA) driver -- TRUE 5-FOLD CV, one job runs all 5 folds on a
SINGLE GPU (encoder style). Reuses the base paper's recipe (4-bit nf4 + LoRA +
paged_adamw_8bit). No DeepSpeed / ZeRO-3 / NCCL / P2P.

THREE configs, each with a rank-appropriate learning rate (the 20-fold sweep showed a
fixed lr=2e-4 destabilises high-rank LoRA -> here every rank trains at a stable lr so
each performs as well as it can):

  paper : r=8,   alpha=16,  q/v,         lr=2e-4, dropout=0.05  (~3.4M trainable; paper repro)
  r64   : r=64,  alpha=128, all-linear,  lr=1e-4, dropout=0.10  (~168M)
  r128  : r=128, alpha=256, all-linear,  lr=5e-5, dropout=0.10  (~336M)

TRUE 5-fold CV: each of the 5 splits is the TEST set exactly once (tags 10/21/32/43/04),
matching the encoders + XGBoost so the joint statistics are valid.

RESUMABLE: a fold whose df_res exists is skipped -> resubmit the same job to continue.

Efficient: SDPA attention + real batch 16 (effective batch 32), dataloader workers.

Outputs -> dat/<dataset>/proc/cross_val_biomistral_<config>_temp/df_res<tag>.csv
           columns: biomistral_<config>_temp / biomistral_<config>_temp_cal / label
"""
import os
import sys
import gc
import random
import shutil
import logging
import datetime
import argparse

import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import med_llama_peft_train_src as src

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
HF_TOKEN = os.environ.get("HF_TOKEN")   # optional; Med-LLaMA3-8B is public
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

MODEL_ID = "YBXL/Med-LLaMA3-8B"   # base paper's Med-LLaMA model
DEFAULT_DATASET = "Analgesics-induced_acute_liver_failure"
MAX_LEN = 128

# per-config: rank, alpha (=2r), target modules, learning rate, LoRA dropout
CONFIGS = {
    "paper": dict(r=8,   alpha=16,  targets=["q_proj", "v_proj"], lr=2e-4, dropout=0.05),
    "r64":   dict(r=64,  alpha=128, targets="all-linear",         lr=1e-4, dropout=0.10),
    "r128":  dict(r=128, alpha=256, targets="all-linear",         lr=5e-5, dropout=0.10),
}

ap = argparse.ArgumentParser(description="BioMistral-7B PEFT -- true 5-fold, single GPU.")
ap.add_argument("--dataset", default=DEFAULT_DATASET)
ap.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
ap.add_argument("--per_device_train_batch_size", type=int, default=16)
ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
ap.add_argument("--learning_rate", type=float, default=None, help="override the per-config lr")
args, _ = ap.parse_known_args()
DATASET = args.dataset
CFG = CONFIGS[args.config]
LR = args.learning_rate if args.learning_rate is not None else CFG["lr"]
MODEL_KEY = f"med_llama_{args.config}"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_BASE = os.path.join(PROJECT_ROOT, "outputs", MODEL_KEY, DATASET)
TRAINLOG_DIR = os.path.join(OUT_BASE, "train_logs")
RUNLOG_DIR = os.path.join(OUT_BASE, "run_logs")
PRED_DIR = os.path.join(PROJECT_ROOT, "dat", DATASET, "proc", f"cross_val_{MODEL_KEY}_temp")
for _d in (TRAINLOG_DIR, RUNLOG_DIR, PRED_DIR):
    os.makedirs(_d, exist_ok=True)

RUN_TAG = os.environ.get("SLURM_JOB_ID") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(RUNLOG_DIR, f"{MODEL_KEY}_{DATASET}_{RUN_TAG}.log"))])
logger = logging.getLogger("med_llama_peft")


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def build_quantized_lora_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, device_map="auto",
        quantization_config=bnb_config, attn_implementation="sdpa",
        trust_remote_code=True, token=HF_TOKEN)   # SDPA = faster
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    lora_config = LoraConfig(
        r=CFG["r"], lora_alpha=CFG["alpha"], target_modules=CFG["targets"],
        lora_dropout=CFG["dropout"], bias="none", task_type="SEQ_CLS",
        modules_to_save=["score"])
    model = get_peft_model(model, lora_config)
    return tokenizer, model


def full_process(split):
    tag = f"{split['dev'][0]}{split['test'][0]}"
    set_seed(42)
    tokenizer, model = build_quantized_lora_model()
    if not getattr(full_process, "_logged", False):
        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        logger.info(f"trainable params: {tp:,} / {tot:,} ({100*tp/tot:.3f}%)  lr={LR}  dropout={CFG['dropout']}")
        full_process._logged = True

    df_train, df_dev, df_test = src.get_split(split, DATASET)

    def preprocess(examples):
        out = tokenizer(examples["Temp_sentence"], truncation=True, max_length=MAX_LEN)
        out["labels"] = examples["label"]
        return out

    train_ds, dev_ds, test_ds = (Dataset.from_pandas(df_train),
                                 Dataset.from_pandas(df_dev), Dataset.from_pandas(df_test))
    tk_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    tk_dev = dev_ds.map(preprocess, batched=True, remove_columns=dev_ds.column_names)
    tk_test = test_ds.map(preprocess, batched=True, remove_columns=test_ds.column_names)

    scratch = os.path.join(os.environ.get("FOLD_SCRATCH", os.path.join(PROJECT_ROOT, "scratch")),
                           f"{MODEL_KEY}_{DATASET}_{tag}")
    os.makedirs(scratch, exist_ok=True)

    trainer = src.train_peft(
        tokenizer, model, tk_train, tk_dev, scratch,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=LR)

    prob_dev, prob_test = src.predict(trainer, tk_dev, tk_test)
    prob_test_cal = src.calibration(prob_dev, df_dev["label"].tolist(), prob_test)

    try:
        pd.DataFrame(trainer.state.log_history).to_csv(
            os.path.join(TRAINLOG_DIR, f"train_curve_{tag}.csv"), index=False)
    except Exception as e:
        logger.warning(f"[fold {tag}] curve save failed: {e}")

    df_res = pd.DataFrame(index=df_test.index)
    df_res[f"{MODEL_KEY}_temp"] = prob_test
    df_res[f"{MODEL_KEY}_temp_cal"] = prob_test_cal
    df_res["label"] = df_test["label"]
    df_res.to_csv(os.path.join(PRED_DIR, f"df_res{tag}.csv"))
    logger.info(f"[fold {tag}] wrote predictions -> df_res{tag}.csv")

    del trainer, model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    # TRUE 5-fold: each split is TEST exactly once, fixed dev, rest train.
    possible_splits = []
    for test in range(5):
        dev = (test + 1) % 5
        train = [x for x in range(5) if x != test and x != dev]
        possible_splits.append({"dev": [dev], "test": [test], "train": train})

    logger.info(f"=== Med-LLaMA-8B PEFT [{args.config}] | {DATASET} | TRUE 5-fold ===")
    logger.info(f"r={CFG['r']} alpha={CFG['alpha']} targets={CFG['targets']} lr={LR} dropout={CFG['dropout']} max_len={MAX_LEN}")

    done = ran = 0
    for split in possible_splits:
        tag = f"{split['dev'][0]}{split['test'][0]}"
        if os.path.exists(os.path.join(PRED_DIR, f"df_res{tag}.csv")):
            logger.info(f"[fold {tag}] already done -> skipping"); done += 1; continue
        logger.info(f"---- fold {tag} ({ran+done+1}/5) ----")
        full_process(split); ran += 1

    logger.info(f"=== [{args.config}] {DATASET} complete: {ran} trained, {done} skipped, {ran+done}/5 ===")
