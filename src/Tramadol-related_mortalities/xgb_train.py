"""
Training script for the XGBoost baseline.
Performs the same 20-run cross-validation as the transformer models, tunes
hyperparameters on the validation fold with Optuna, applies Isotonic Regression
for calibration, and saves the output predictions.

Structured to mirror albert_train.py / biobert_train.py so the baseline drops
into the identical evaluation pipeline. Runs on CPU - no GPU required.

WHAT MATCHES THE PAPER
  * 5 folds, 20 runs, 3 train / 1 dev / 1 test  (section 2.2)
  * multi-hot encoded tabular features           (section 2.1)
  * dev fold used for hyperparameter tuning, early stopping and calibrator fitting
  * isotonic calibration, applied exactly as in the transformer scripts
  * output columns named "xgb" and "xgb_cal", written to cross_val_xgb/,
    which is what calc_metrics.ipynb / calc_zscores.ipynb expect

USAGE
    python xgb_train.py                          # this folder's dataset
    python xgb_train.py --dataset Tramadol-related_mortalities
    python xgb_train.py --n_trials 50            # more Optuna trials per split
    python xgb_train.py --tune_once              # tune on split 01, reuse (faster)
    python xgb_train.py --splits 01 12           # subset, for a quick smoke test
"""
import argparse
import datetime
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import xgb_train_src

MODEL_NAME = "XGBoost"
DEFAULT_DATASET = "Tramadol-related_mortalities"

parser = argparse.ArgumentParser(description="XGBoost cross-validation training.")
parser.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="Dataset folder name under dat/ (default: %(default)s)")
parser.add_argument("--n_trials", type=int, default=30,
                    help="Optuna trials per split (default: %(default)s)")
parser.add_argument("--tune_once", action="store_true",
                    help="Tune on the first split only and reuse those params for the rest.")
parser.add_argument("--splits", nargs="+", default=None,
                    help="Only run these split tags, e.g. --splits 01 12")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--project_root", default=None,
                    help="Folder containing dat/ (default: two levels above this file)")
args = parser.parse_args()

DATASET = args.dataset
PROJECT_ROOT = args.project_root or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

OUT_BASE = os.path.join(PROJECT_ROOT, "outputs", "xgboost", DATASET)
MODELS_DIR = os.path.join(OUT_BASE, "models")        # booster + vocab per split
TRAINLOG_DIR = os.path.join(OUT_BASE, "train_logs")  # training curves + best params
RUNLOG_DIR = os.path.join(OUT_BASE, "run_logs")
for _d in (MODELS_DIR, TRAINLOG_DIR, RUNLOG_DIR):
    os.makedirs(_d, exist_ok=True)

RUN_TAG = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = os.path.join(RUNLOG_DIR, f"xgboost_{DATASET}_{RUN_TAG}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(_log_path)],
)
logger = logging.getLogger("xgb_train")

logger.info("=== XGBoost training run started ===")
logger.info(f"dataset={DATASET}  project_root={PROJECT_ROOT}")
try:
    import xgboost as _xgb
    import optuna as _opt
    logger.info(f"xgboost {_xgb.__version__} | optuna {_opt.__version__}")
except ImportError as e:
    logger.error(f"missing dependency: {e}. Run: pip install xgboost optuna")
    sys.exit(1)


def set_seed(seed):
    np.random.seed(seed)
    import random
    random.seed(seed)


def full_process(split, dataset, shared_params=None):
    set_seed(args.seed)
    split_tag = f'{split["dev"][0]}{split["test"][0]}'
    logger.info(f"---- split dev={split['dev']} test={split['test']} train={split['train']} (tag {split_tag}) ----")
    t0 = time.time()

    df_train, df_dev, df_test = xgb_train_src.get_split(split, dataset, PROJECT_ROOT)
    feature_cols = [c for c in df_train.columns if c != "label"]

    # Vocabulary from the training fold only, so dev/test stay unseen.
    vocab = xgb_train_src.build_vocabulary(df_train, feature_cols)
    X_train = xgb_train_src.multi_hot_encode(df_train, vocab, feature_cols)
    X_dev = xgb_train_src.multi_hot_encode(df_dev, vocab, feature_cols)
    X_test = xgb_train_src.multi_hot_encode(df_test, vocab, feature_cols)
    y_train = df_train["label"].values
    y_dev = df_dev["label"].values

    logger.info(f"[split {split_tag}] train/dev/test = {X_train.shape[0]}/{X_dev.shape[0]}/{X_test.shape[0]}"
                f"  features = {X_train.shape[1]}  (density {X_train.nnz / np.prod(X_train.shape):.4f})")

    # ---- hyperparameter tuning on the validation fold ----
    if shared_params is not None:
        params, best_iter = shared_params
        logger.info(f"[split {split_tag}] reusing tuned params (--tune_once)")
    else:
        ts = time.time()
        params, best_iter, study = xgb_train_src.tune_hyperparameters(
            X_train, y_train, X_dev, y_dev, n_trials=args.n_trials, seed=args.seed)
        logger.info(f"[split {split_tag}] Optuna: {args.n_trials} trials in {time.time()-ts:.0f}s, "
                    f"best dev AUC={study.best_value:.4f}")
        logger.info(f"[split {split_tag}] best params: "
                    + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in study.best_params.items()))
        try:
            pd.DataFrame([{**t.params, "value": t.value, "number": t.number}
                          for t in study.trials]).to_csv(
                os.path.join(TRAINLOG_DIR, f"optuna_trials_{split_tag}.csv"), index=False)
        except Exception as e:
            logger.warning(f"[split {split_tag}] could not save Optuna trials: {e}")

    # ---- final fit with early stopping on dev ----
    booster, evals_result = xgb_train_src.train_xgb(
        X_train, y_train, X_dev, y_dev, params, seed=args.seed)
    logger.info(f"[split {split_tag}] fitted {booster.best_iteration + 1} rounds "
                f"(best dev AUC={booster.best_score:.4f})")

    # ---- predict + calibrate (calibration is the repo's verbatim function) ----
    prob_dev, prob_test = xgb_train_src.predict(booster, X_dev, X_test)
    prob_test_cal = xgb_train_src.calibration(prob_dev, df_dev["label"].tolist(), prob_test)

    # ---- persist training curve, params, model ----
    try:
        curve = pd.DataFrame({f"{k}_{m}": v for k, d in evals_result.items() for m, v in d.items()})
        curve.index.name = "round"
        curve.to_csv(os.path.join(TRAINLOG_DIR, f"train_curve_{split_tag}.csv"))
    except Exception as e:
        logger.warning(f"[split {split_tag}] could not save training curve: {e}")

    try:
        split_dir = os.path.join(MODELS_DIR, f"split_{split_tag}")
        os.makedirs(split_dir, exist_ok=True)
        booster.save_model(os.path.join(split_dir, "model.json"))
        with open(os.path.join(split_dir, "vocabulary.json"), "w") as f:
            json.dump({"feature_cols": feature_cols, "vocab": vocab}, f)
        with open(os.path.join(split_dir, "params.json"), "w") as f:
            json.dump({"params": params, "best_iteration": int(booster.best_iteration),
                       "best_dev_auc": float(booster.best_score)}, f, indent=2)
        imp = booster.get_score(importance_type="gain")
        names = xgb_train_src.feature_names(vocab, feature_cols)
        rows = [{"feature": names[int(k[1:])] if k[0] == "f" and k[1:].isdigit() else k,
                 "gain": v} for k, v in imp.items()]
        pd.DataFrame(rows).sort_values("gain", ascending=False).to_csv(
            os.path.join(split_dir, "feature_importance.csv"), index=False)
        logger.info(f"[split {split_tag}] saved model -> {split_dir}")
    except Exception as e:
        logger.warning(f"[split {split_tag}] could not save model: {e}")

    # ---- write predictions in the layout the evaluation code expects ----
    df_res = pd.DataFrame()
    df_res.index = df_test.index
    df_res["xgb"] = prob_test
    df_res["xgb_cal"] = prob_test_cal
    df_res["label"] = df_test["label"]

    out_dir = os.path.join(PROJECT_ROOT, "dat", dataset, "proc", "cross_val_xgb")
    os.makedirs(out_dir, exist_ok=True)
    df_res.to_csv(os.path.join(out_dir, f"df_res{split_tag}.csv"))

    acc = ((np.array(prob_test) >= 0.5).astype(int) == df_test["label"].values).mean()
    logger.info(f"[split {split_tag}] test accuracy={acc:.4f}  "
                f"({time.time()-t0:.0f}s total)  -> df_res{split_tag}.csv")
    return (params, best_iter)


possible_splits = []
for i in range(5):
    for j in range(5):
        if i != j:
            dicti = {"dev": [i], "test": [j], "train": [x for x in range(5) if x != i and x != j]}
            possible_splits.append(dicti)

if args.splits:
    possible_splits = [s for s in possible_splits
                       if f'{s["dev"][0]}{s["test"][0]}' in args.splits]
    logger.info(f"restricted to splits: {args.splits}")

shared = None
for split in tqdm(possible_splits):
    result = full_process(split, DATASET, shared)
    if args.tune_once and shared is None:
        shared = result

logger.info("=== XGBoost training run finished ===")
logger.info(f"models    -> {MODELS_DIR}")
logger.info(f"curves    -> {TRAINLOG_DIR}")
logger.info(f"run log   -> {_log_path}")
logger.info(f"preds     -> {os.path.join(PROJECT_ROOT, 'dat', DATASET, 'proc', 'cross_val_xgb')}")
