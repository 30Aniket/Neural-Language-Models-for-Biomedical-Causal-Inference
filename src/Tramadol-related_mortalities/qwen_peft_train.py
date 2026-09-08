"""
Qwen3.5-4B-Base / Qwen3.5-9B-Base PEFT (QLoRA) driver -- TRUE 5-FOLD CV, one job runs
all 5 folds on a SINGLE GPU. Cloned from the (working) Gemma driver: same recipe, same
_src helper, same 5-fold rotation, same outputs -> slots into run_evaluation.py.

Qwen3.5 specifics (vs the 7B text-only models):
  * MULTIMODAL checkpoints (Image-Text-to-Text) -> load native class, take the TEXT
    decoder, attach a last-token classification head (same pattern as Gemma).
  * HYBRID architecture (Gated-DeltaNet + attention). The paper config (q_proj/v_proj)
    therefore adapts ONLY the full-attention layers; all-linear covers GDN + FFN too.
  * needs transformers >= 5.2.0.

Mistakes carried over from Gemma debugging are pre-fixed here:
  * Trainer uses processing_class= (not tokenizer=) via the shared _src.
  * from_pretrained uses dtype= (not the deprecated torch_dtype=).
  * JIT/compile caches are redirected off HOME in the SLURM script.
  * --smoke runs a REAL 4-step train end-to-end (Trainer + fwd/bwd + predict) to catch
    transformers-5.2 API breakage BEFORE queuing the long jobs.

Configs (identical to the 7B/Gemma sweep):
  paper : r=8,   alpha=16,  q/v (attention layers only), lr=2e-4, dropout=0.05
  r64   : r=64,  alpha=128, all-linear,                  lr=1e-4, dropout=0.10
  r128  : r=128, alpha=256, all-linear,                  lr=5e-5, dropout=0.10

Models via --model: qwen4b (Qwen/Qwen3.5-4B-Base), qwen9b (Qwen/Qwen3.5-9B-Base).
Both Apache-2.0 (not gated). Outputs ->
  dat/<dataset>/proc/cross_val_<model>_<config>_temp/df_res<tag>.csv
  columns: <model>_<config>_temp / <model>_<config>_temp_cal / label
"""
import os, sys, gc, random, shutil, logging, datetime, argparse
import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import qwen_peft_train_src as src

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
HF_TOKEN = os.environ.get("HF_TOKEN")   # not required (Apache-2.0), passed if present
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

MODEL_IDS = {
    "qwen4b": "Qwen/Qwen3.5-4B-Base",
    "qwen9b": "Qwen/Qwen3.5-9B-Base",
}
DEFAULT_DATASET = "Analgesics-induced_acute_liver_failure"
MAX_LEN = 128
ATTN = "sdpa"      # applies to the attention layers; flip to "eager" only if it NaNs

CONFIGS = {
    "paper": dict(r=8,   alpha=16,  targets=["q_proj", "v_proj"], lr=2e-4, dropout=0.05),
    "r64":   dict(r=64,  alpha=128, targets="all-linear",         lr=1e-4, dropout=0.10),
    "r128":  dict(r=128, alpha=256, targets="all-linear",         lr=5e-5, dropout=0.10),
}

ap = argparse.ArgumentParser(description="Qwen3.5-Base PEFT -- true 5-fold, single GPU.")
ap.add_argument("--model", required=True, choices=list(MODEL_IDS.keys()))
ap.add_argument("--dataset", default=DEFAULT_DATASET)
ap.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
ap.add_argument("--per_device_train_batch_size", type=int, default=16)
ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
ap.add_argument("--learning_rate", type=float, default=None)
ap.add_argument("--dry-run-model", action="store_true",
                help="build model + LoRA, print trainable params, exit (no Trainer)")
ap.add_argument("--smoke", action="store_true",
                help="REAL 4-step train on a tiny subsample -> validates the full path")
args, _ = ap.parse_known_args()
MODEL_ID = MODEL_IDS[args.model]
DATASET = args.dataset
CFG = CONFIGS[args.config]
LR = args.learning_rate if args.learning_rate is not None else CFG["lr"]
MODEL_KEY = f"{args.model}_{args.config}"

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
logger = logging.getLogger("qwen_peft")


def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


# ---- multimodal checkpoint -> text sequence classifier -------------------
class QwenTextClassifier(torch.nn.Module):
    def __init__(self, backbone, num_labels=2):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        hidden = (getattr(self.config, "hidden_size", None)
                  or getattr(getattr(self.config, "text_config", None), "hidden_size", None))
        self.score = torch.nn.Linear(hidden, num_labels, bias=False).to(torch.bfloat16)
        self.num_labels = num_labels

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def gradient_checkpointing_enable(self, **kw):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kw)

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        h = out.last_hidden_state
        if attention_mask is not None:
            idx = attention_mask.sum(dim=1).sub(1).clamp(min=0).long()
        else:
            idx = torch.full((h.size(0),), h.size(1) - 1, device=h.device).long()
        pooled = h[torch.arange(h.size(0), device=h.device), idx]
        logits = self.score(pooled.to(self.score.weight.dtype))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.float(), labels.long())
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)


def _find_text_backbone(full):
    # try the common nesting paths across transformers VL model layouts
    for path in ("model.language_model", "language_model", "model.model.language_model",
                 "model.text_model", "text_model", "thinker.model.language_model",
                 "model.model"):
        obj, ok = full, True
        for p in path.split("."):
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False; break
        # a text backbone should expose get_input_embeddings + return last_hidden_state
        if ok and hasattr(obj, "get_input_embeddings"):
            return obj, path
    return None, None


def load_qwen_backbone_classifier(bnb_config):
    import transformers
    full = None
    for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            loader = getattr(transformers, loader_name)
            full = loader.from_pretrained(
                MODEL_ID, device_map="auto", quantization_config=bnb_config,
                attn_implementation=ATTN, dtype=torch.bfloat16,   # dtype= (not torch_dtype=)
                trust_remote_code=True, token=HF_TOKEN)
            logger.info(f"loaded checkpoint via {loader_name}")
            break
        except Exception as e:
            logger.warning(f"{loader_name} load failed: {type(e).__name__}: {e}")
            full = None
    if full is None:
        raise RuntimeError("could not load the Qwen3.5 checkpoint by any loader")

    backbone, where = _find_text_backbone(full)
    if backbone is None:
        backbone = getattr(full, "model", full); where = "fallback .model"
    logger.info(f"using text backbone at '{where}'")
    root = full.model if hasattr(full, "model") else full
    for attr in ("visual", "vision_tower", "vision_model", "multi_modal_projector"):
        if hasattr(root, attr):
            setattr(root, attr, None)   # free vision (text-only task)
    return QwenTextClassifier(backbone, num_labels=2)


def build_quantized_lora_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

    model = load_qwen_backbone_classifier(bnb_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    lora_config = LoraConfig(
        r=CFG["r"], lora_alpha=CFG["alpha"], target_modules=CFG["targets"],
        lora_dropout=CFG["dropout"], bias="none", task_type="SEQ_CLS",
        modules_to_save=["score"])
    model = get_peft_model(model, lora_config)
    return tokenizer, model


def _prep(df, tokenizer):
    def preprocess(examples):
        out = tokenizer(examples["Temp_sentence"], truncation=True, max_length=MAX_LEN)
        out["labels"] = examples["label"]
        return out
    ds = Dataset.from_pandas(df)
    return ds.map(preprocess, batched=True, remove_columns=ds.column_names)


def full_process(split, smoke=False):
    tag = f"{split['dev'][0]}{split['test'][0]}"
    set_seed(42)
    tokenizer, model = build_quantized_lora_model()
    if not getattr(full_process, "_logged", False):
        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        logger.info(f"trainable params: {tp:,} / {tot:,} ({100*tp/tot:.3f}%)  lr={LR}  dropout={CFG['dropout']}")
        full_process._logged = True

    df_train, df_dev, df_test = src.get_split(split, DATASET)
    if smoke:
        df_train, df_dev, df_test = df_train.head(64), df_dev.head(64), df_test.head(64)
    tk_train, tk_dev, tk_test = _prep(df_train, tokenizer), _prep(df_dev, tokenizer), _prep(df_test, tokenizer)

    scratch = os.path.join(os.environ.get("FOLD_SCRATCH", os.path.join(PROJECT_ROOT, "scratch")),
                           f"{MODEL_KEY}_{DATASET}_{tag}{'_smoke' if smoke else ''}")
    os.makedirs(scratch, exist_ok=True)

    kw = dict(per_device_train_batch_size=args.per_device_train_batch_size,
              per_device_eval_batch_size=args.per_device_train_batch_size,
              gradient_accumulation_steps=args.gradient_accumulation_steps, learning_rate=LR)
    if smoke:
        kw.update(max_steps=4, warmup_steps=0, eval_steps=2, save_steps=2)
    trainer = src.train_peft(tokenizer, model, tk_train, tk_dev, scratch, **kw)

    prob_dev, prob_test = src.predict(trainer, tk_dev, tk_test)
    prob_test_cal = src.calibration(prob_dev, df_dev["label"].tolist(), prob_test)

    if smoke:
        del trainer, model, tokenizer; gc.collect(); torch.cuda.empty_cache()
        shutil.rmtree(scratch, ignore_errors=True)
        logger.info("SMOKE OK -- Trainer, train step, predict and calibration all succeeded.")
        return

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
    if args.dry_run_model:
        set_seed(42); _, m = build_quantized_lora_model()
        tp = sum(p.numel() for p in m.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in m.parameters())
        logger.info(f"DRY RUN OK -- trainable params: {tp:,} / {tot:,} ({100*tp/tot:.3f}%)")
        sys.exit(0)

    possible_splits = []
    for test in range(5):
        dev = (test + 1) % 5
        train = [x for x in range(5) if x != test and x != dev]
        possible_splits.append({"dev": [dev], "test": [test], "train": train})

    if args.smoke:
        logger.info(f"=== SMOKE {args.model} [{args.config}] | {DATASET} ===")
        full_process(possible_splits[0], smoke=True)
        sys.exit(0)

    logger.info(f"=== {args.model} PEFT [{args.config}] | {DATASET} | TRUE 5-fold ===")
    logger.info(f"model={MODEL_ID} r={CFG['r']} alpha={CFG['alpha']} targets={CFG['targets']} "
                f"lr={LR} dropout={CFG['dropout']} max_len={MAX_LEN} attn={ATTN}")
    done = ran = 0
    for split in possible_splits:
        tag = f"{split['dev'][0]}{split['test'][0]}"
        if os.path.exists(os.path.join(PRED_DIR, f"df_res{tag}.csv")):
            logger.info(f"[fold {tag}] already done -> skipping"); done += 1; continue
        logger.info(f"---- fold {tag} ({ran+done+1}/5) ----")
        full_process(split); ran += 1
    logger.info(f"=== [{args.config}] {DATASET} complete: {ran} trained, {done} skipped, {ran+done}/5 ===")
