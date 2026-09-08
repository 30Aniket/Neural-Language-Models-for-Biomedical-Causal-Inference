"""
Training script for BioBERT model using sequence classification.
This script performs cross-validation, trains the transformer, applies Isotonic Regression for calibration,
and saves the output predictions.

CLUSTER-READY VERSION
=====================
The training / prediction / calibration LOGIC is IDENTICAL to the original 2026 repo
(hsdslab/biomedical-causal-inference). The only additions are:
  * Python logging that mirrors every print/step to a persistent .log file
    (in addition to the SLURM slurm-<jobid>.out capture).
  * Saving of each split's learned BioBERT weights (safetensors + tokenizer via
    `trainer.save_model`, plus an optional torch state_dict .pt) so they can be
    downloaded and reused for local evaluation.
  * Dumping each split's full training curve (trainer.state.log_history) to a CSV.
  * The original scratch dirs ("results", "logs") are still cleaned to save disk,
    but ONLY after the useful information has been copied out.
All additions are guarded/commented with the marker  # [CLUSTER]  so they are easy to spot.
The default dataset is unchanged; an optional CLI arg lets one SLURM script serve both datasets.

NOTE ON THE .lower() CALL: the repo lowercases Temp_sentence even though
'dmis-lab/biobert-base-cased-v1.1' is a CASED model. This is preserved verbatim -
it is what produced the published results, and changing it would break comparability
with the source paper. Do not "fix" it without discussing with your supervisor.
"""
import numpy as np
import torch
from transformers import TrainingArguments, Trainer
import pandas as pd
import os
from transformers import DataCollatorWithPadding, AdamW, EarlyStoppingCallback, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
import random
import biobert_train_src
from datasets import Dataset
import shutil
from tqdm import tqdm

# [CLUSTER] extra std-lib imports for logging / paths / timestamps
import sys
import json
import logging
import argparse
import datetime

import os
os.environ["WANDB_DISABLED"] = "true"

# ---------------------------------------------------------------------------
# [CLUSTER] Configuration: dataset name + persistent output locations.
#   * DATASET defaults to the original hard-coded value for this folder, but can
#     be overridden on the command line so one SLURM script drives both datasets.
#   * All heavy artefacts (weights, logs, training curves) go under <project_root>/outputs
#     so they survive the deletion of the scratch "results"/"logs" dirs and can be
#     rsync'd back to your local machine.
# ---------------------------------------------------------------------------
MODEL_NAME = "BioBERT"
DEFAULT_DATASET = "Tramadol-related_mortalities"

parser = argparse.ArgumentParser(description="BioBERT cross-validation training (cluster-ready).")
parser.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="Dataset folder name under dat/ (default: %(default)s)")
parser.add_argument("--save_state_dict", action="store_true",
                    help="Also save a raw torch state_dict (.pt) alongside the safetensors weights.")
# parse_known_args so the script still works if launched with no args at all
args, _ = parser.parse_known_args()
DATASET = args.dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_BASE = os.path.join(PROJECT_ROOT, "outputs", MODEL_NAME.lower(), DATASET)
WEIGHTS_DIR = os.path.join(OUT_BASE, "weights")        # [CLUSTER] learned weights per split
TRAINLOG_DIR = os.path.join(OUT_BASE, "train_logs")    # [CLUSTER] training-curve CSVs per split
RUNLOG_DIR = os.path.join(OUT_BASE, "run_logs")        # [CLUSTER] full text log of this run
for _d in (WEIGHTS_DIR, TRAINLOG_DIR, RUNLOG_DIR):
    os.makedirs(_d, exist_ok=True)

# [CLUSTER] Configure logging: everything goes to console (captured by SLURM) AND a file.
RUN_TAG = os.environ.get("SLURM_JOB_ID") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = os.path.join(RUNLOG_DIR, f"biobert_{DATASET}_{RUN_TAG}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(_log_path)],
)
logger = logging.getLogger("biobert_train")
logger.info("=== BioBERT training run started ===")
logger.info(f"dataset={DATASET}  project_root={PROJECT_ROOT}")
logger.info(f"CUDA available: {torch.cuda.is_available()} | "
            f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1", num_labels=2, ignore_mismatched_sizes=True)

# [CLUSTER] log model size (BioBERT ~110M params vs ALBERT ~12M)
logger.info(f"model loaded: dmis-lab/biobert-base-cased-v1.1 "
            f"({sum(p.numel() for p in model.parameters()):,} parameters)")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def full_process(split, dataset, model, tokenizer):

    # Set a global seed
    set_seed(42)
    # [CV-FIX] Re-initialise the model from its pretrained checkpoint for THIS fold,
    # [CV-FIX] so fine-tuned weights never leak across cross-validation splits.
    model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=2, ignore_mismatched_sizes=True)

    # [CLUSTER] readable tag for this split, e.g. dev1/test3 -> "13"
    split_tag = f'{split["dev"][0]}{split["test"][0]}'
    logger.info(f"---- split dev={split['dev']} test={split['test']} train={split['train']} (tag {split_tag}) ----")

    df_train, df_dev, df_test = biobert_train_src.get_split(split, dataset)

    train_data = Dataset.from_pandas(df_train)
    test_data = Dataset.from_pandas(df_test)
    dev_data = Dataset.from_pandas(df_dev)

    def preprocess_function(examples):
        # Lowercase the text separately
        examples["Temp_sentence"] = [sentence.lower() for sentence in examples["Temp_sentence"]]
        return tokenizer(examples["Temp_sentence"], truncation=True, max_length=128)

    tokenized_train = train_data.map(preprocess_function, batched=True)
    tokenized_test = test_data.map(preprocess_function, batched=True)
    tokenized_dev = dev_data.map(preprocess_function, batched=True)

    trainer = biobert_train_src.train_transformer(tokenizer, model, tokenized_train, tokenized_dev)

    prob_dev, prob_test = biobert_train_src.predict(trainer, tokenized_dev, tokenized_test)

    prob_test_cal = biobert_train_src.calibration(prob_dev, df_dev["label"].tolist(), prob_test)

    # [CLUSTER] BEFORE cleaning the scratch dirs, persist everything we want to keep:
    #   1) the training curve (loss / eval_loss / lr per logging step)
    try:
        log_hist = pd.DataFrame(trainer.state.log_history)
        log_hist.to_csv(os.path.join(TRAINLOG_DIR, f"train_curve_{split_tag}.csv"), index=False)
        logger.info(f"[split {split_tag}] saved training curve "
                    f"({len(log_hist)} log rows, best_metric={getattr(trainer.state, 'best_metric', None)})")
    except Exception as e:
        logger.warning(f"[split {split_tag}] could not save training curve: {e}")

    #   2) the learned weights for this split (best model is already loaded because
    #      load_best_model_at_end=True). save_model writes model.safetensors + config + tokenizer,
    #      i.e. a folder directly reloadable via AutoModelForSequenceClassification.from_pretrained(...).
    split_weight_dir = os.path.join(WEIGHTS_DIR, f"split_{split_tag}")
    try:
        trainer.save_model(split_weight_dir)          # safetensors weights + config
        tokenizer.save_pretrained(split_weight_dir)   # tokenizer for standalone reload
        if args.save_state_dict:                      # optional raw torch weights
            torch.save(trainer.model.state_dict(), os.path.join(split_weight_dir, "pytorch_state_dict.pt"))
        logger.info(f"[split {split_tag}] saved weights -> {split_weight_dir}")
    except Exception as e:
        logger.warning(f"[split {split_tag}] could not save weights: {e}")

     # delete results and logs dir
    # [CLUSTER] ignore_errors=True added: TrainingArguments sets logging_dir='logs',
    # but that directory is only created if a logging integration (e.g. TensorBoard)
    # actually writes to it. TensorBoard is not in our pinned env, so 'logs' never
    # exists and the original bare rmtree raised FileNotFoundError, killing the run
    # after the first split. This is cleanup-only code - training, prediction and
    # calibration logic are untouched.
    shutil.rmtree("results", ignore_errors=True)
    shutil.rmtree("logs", ignore_errors=True)
    #shutil.rmtree("mlruns")
    #shutil.rmtree("wandb")

    df_res = pd.DataFrame()
    df_res.index = df_test.index
    df_res["biobert_temp"] = prob_test
    df_res["biobert_temp_cal"] = prob_test_cal
    df_res["label"] = df_test["label"]

    # Ensure relative paths for repository portability
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(project_root, "dat", dataset, "proc", "cross_val_biobert_temp")
    os.makedirs(out_dir, exist_ok=True)
    df_res.to_csv(os.path.join(out_dir, f'df_res{split["dev"][0]}{split["test"][0]}.csv'))

    logger.info(f"[split {split_tag}] wrote predictions -> "
                f"{os.path.join(out_dir, f'df_res{split_tag}.csv')}")
    return None

possible_splits = []

# [5-FOLD] True 5-fold CV: each of the 5 splits is the TEST set exactly once,
# [5-FOLD] with a fixed dev split and the remaining three as train. This replaces
# [5-FOLD] the original 5x5 = 20-fold (all dev/test pairs) grid. Same split.csv /
# [5-FOLD] grouping is used; we simply enumerate 5 folds instead of 20 so every
# [5-FOLD] model in the study shares an identical, standard 5-fold CV.
for test in range(5):
    dev = (test + 1) % 5
    train = [x for x in range(5) if x != test and x != dev]
    possible_splits.append({"dev": [dev], "test": [test], "train": train})
for split in tqdm(possible_splits):
    full_process(split, DATASET, model, tokenizer)   # [CLUSTER] DATASET var (defaults to original)

# [CLUSTER] final summary so the end of the log is easy to find
logger.info("=== BioBERT training run finished ===")
logger.info(f"weights   -> {WEIGHTS_DIR}")
logger.info(f"curves    -> {TRAINLOG_DIR}")
logger.info(f"run log   -> {_log_path}")
logger.info(f"preds     -> {os.path.join(PROJECT_ROOT, 'dat', DATASET, 'proc', 'cross_val_biobert_temp')}")
