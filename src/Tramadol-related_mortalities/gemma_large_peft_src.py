"""
Shared engine for the large-Gemma QLoRA 5-fold runs (Gemma3-12B, Gemma4-12B,
Gemma4-31B). The three drivers that import this differ only in a MODEL_SPEC
dict -- checkpoint id, how the weights are placed, whether gradient
checkpointing is on, and the default batch geometry.

WHAT IS UNCHANGED FROM YOUR ORIGINAL CODE (deliberately -- these results have
to pool with every model already on disk):
  * the 5-fold rotation and split tags (10/21/32/43/04)
  * get_split, the last-token pooling head, the LoRA configs, lr, dropout
  * TrainingArguments: optim, weight_decay, max_steps, warmup, lr_scheduler,
    eval/save every 200 steps, EarlyStopping(patience=10, threshold=0.001),
    load_best_model_at_end, bf16, effective batch 32
  * predict() and the isotonic calibration
  * output paths and column names, so run_evaluation.py needs no edits

WHAT IS NEW (efficiency and robustness only -- none of it touches the maths):
  1. per_device_eval_batch_size is its own knob. It used to be pinned to the
     train batch, which made the 31B evaluate one sample at a time.
  2. pad_to_multiple_of=8 on the collator. Padded positions are masked out of
     attention and excluded from the last-token index, so this changes no
     number -- it just aligns the GEMMs.
  3. Mid-fold resume. Checkpoints can live on project storage instead of node
     TMPDIR, so a walltime kill costs <=200 steps rather than a whole fold.
  4. Per-fold GPU accounting and a hard teardown between folds.
  5. --probe runs a few real steps and prints peak memory, so you can size the
     batch before committing a 48h allocation.

Effective batch is asserted at startup: train_bs * accum must be 32.
"""
import argparse
import datetime
import gc
import glob
import logging
import os
import random
import re
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.isotonic import IsotonicRegression
from transformers import (AutoTokenizer, BitsAndBytesConfig, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

MAX_LEN = 128
ATTN = "sdpa"
EFFECTIVE_BATCH = 32          # every model in the study trains at 32
HF_TOKEN = os.environ.get("HF_TOKEN") or None

CONFIGS = {
    "paper": dict(r=8,   alpha=16,  targets=["q_proj", "v_proj"], lr=2e-4, dropout=0.05),
    "r64":   dict(r=64,  alpha=128, targets="all-linear",         lr=1e-4, dropout=0.10),
    "r128":  dict(r=128, alpha=256, targets="all-linear",         lr=5e-5, dropout=0.10),
}

logger = logging.getLogger("gemma_large")


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def log_mem(where):
    if not torch.cuda.is_available():
        return
    g = 2 ** 30
    logger.info(f"[mem] {where}: alloc={torch.cuda.memory_allocated()/g:.2f}GiB "
                f"reserved={torch.cuda.memory_reserved()/g:.2f}GiB "
                f"peak={torch.cuda.max_memory_allocated()/g:.2f}GiB")


def release_module(module):
    """Empty every tensor a module owns, without deleting the module.

    `del model` is not enough: something inside Trainer/Accelerate keeps a live
    reference, so the fp32 embedding table that prepare_model_for_kbit_training
    creates (3.77 GiB on the 12B) survives the delete and accumulates one copy
    per fold. Measured on jobs 1262603-06, fold-start allocation went
    0.00 -> 3.77 -> 7.52 -> 11.27 GiB.

    We cannot reach the holder, but we can empty the storage. Swapping .data for
    a zero-element tensor keeps the Parameter alive for whoever holds it while
    handing its memory back to the CUDA allocator. Safe here because the fold is
    finished and the model is never called again.
    """
    if module is None:
        return 0
    freed = 0
    try:
        for _, prm in module.named_parameters(recurse=True):
            if prm.grad is not None:
                freed += prm.grad.numel() * prm.grad.element_size()
                prm.grad = None
            if prm.data is not None and prm.data.numel():
                freed += prm.data.numel() * prm.data.element_size()
                prm.data = torch.empty(0, device=prm.data.device, dtype=prm.data.dtype)
        for _, buf in module.named_buffers(recurse=True):
            if buf is not None and buf.numel():
                freed += buf.numel() * buf.element_size()
                buf.data = torch.empty(0, device=buf.device, dtype=buf.dtype)
    except Exception as e:
        logger.warning(f"release_module: {type(e).__name__}: {e}")
    return freed


def teardown_fold(trainer=None, model=None):
    """Best-effort in-process release. The SLURM scripts also run one process per
    fold, which frees everything unconditionally at exit -- this is the belt to
    that pair of braces."""
    freed = 0
    if trainer is not None:
        try:
            trainer.accelerator.free_memory()      # accelerate's own teardown, first
        except Exception:
            pass
        freed += release_module(getattr(trainer, "model", None))
        freed += release_module(getattr(trainer, "model_wrapped", None))
        for attr in ("model", "model_wrapped", "optimizer", "lr_scheduler",
                     "callback_handler", "train_dataset", "eval_dataset"):
            try:
                setattr(trainer, attr, None)
            except Exception:
                pass
    freed += release_module(model)
    gc.collect()
    gc.collect()          # cycles freed on pass 1 can release more on pass 2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return freed


def _from_pretrained(loader, model_id, **kw):
    """transformers 5.x wants dtype=; the older venv_gemma stack wants torch_dtype=."""
    try:
        return loader.from_pretrained(model_id, dtype=torch.bfloat16, **kw)
    except TypeError as e:
        if "dtype" not in str(e):
            raise
        return loader.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kw)


# --------------------------------------------------------------------------
# multimodal checkpoint -> text sequence classifier  (unchanged logic)
# --------------------------------------------------------------------------
class GemmaTextClassifier(torch.nn.Module):
    def __init__(self, backbone, num_labels=2):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        hidden = (getattr(self.config, "hidden_size", None)
                  or getattr(getattr(self.config, "text_config", None), "hidden_size", None))
        self.score = torch.nn.Linear(hidden, num_labels, bias=False).to(torch.bfloat16)
        self.num_labels = num_labels
        # newer peft reads these off the top-level module during adapter injection
        self.base_model_prefix = getattr(backbone, "base_model_prefix", "model")
        self._no_split_modules = getattr(backbone, "_no_split_modules", None)
        self.main_input_name = getattr(backbone, "main_input_name", "input_ids")

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
    for path in ("model.language_model", "language_model", "model.model.language_model",
                 "model.text_model", "text_model", "thinker.model.language_model",
                 "model.model", "model"):
        obj, ok = full, True
        for p in path.split("."):
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and hasattr(obj, "get_input_embeddings"):
            return obj, path
    return None, None


def resolve_targets(model, requested):
    """Pick LoRA target modules that actually exist in this architecture.

    Gemma/Llama/Qwen expose separate q_proj and v_proj. Phi-3/Phi-3.5/Phi-4 fuse
    them into a single `qkv_proj`, so the paper config's ["q_proj","v_proj"]
    matches nothing and PEFT would raise. We fall back to the fused projection
    and say so loudly, because adapting qkv_proj also adapts K -- a real
    deviation from the paper config that belongs in the write-up.
    """
    if isinstance(requested, str):          # "all-linear"
        return requested
    names = {n.rsplit(".", 1)[-1] for n, _ in model.named_modules()}
    present = [t for t in requested if t in names]
    if present:
        missing = [t for t in requested if t not in names]
        if missing:
            logger.warning(f"LoRA targets not present in this model: {missing}")
        return present
    for fused in ("qkv_proj", "Wqkv", "query_key_value", "c_attn"):
        if fused in names:
            logger.warning(
                f"FUSED ATTENTION: none of {requested} exist here; using '{fused}' "
                f"instead. This adapts K as well as Q and V -- record as a "
                f"deviation from the paper LoRA config.")
            return [fused]
    raise RuntimeError(
        f"no usable LoRA target modules. Requested {requested}; this model has "
        f"leaf modules such as {sorted(n for n in names if 'proj' in n)[:10]}")


def build_model(spec, cfg, args):
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], trust_remote_code=True,
                                              token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_kw = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    if spec["placement"] == "auto":
        # load-time overflow (the vision tower we delete) spills to CPU instead of OOM
        bnb_kw["llm_int8_enable_fp32_cpu_offload"] = True
    bnb_config = BitsAndBytesConfig(**bnb_kw)

    if spec["placement"] == "auto":
        place_kw = dict(device_map="auto", low_cpu_mem_usage=True,
                        max_memory={0: os.environ.get("GPU_MAX_MEM", spec["gpu_max_mem"]),
                                    "cpu": "160GiB"})
    else:
        place_kw = dict(device_map={"": 0})

    full = None
    for loader_name in spec["loaders"]:
        try:
            import transformers
            loader = getattr(transformers, loader_name, None)
            if loader is None:
                continue
            full = _from_pretrained(loader, spec["model_id"],
                                    quantization_config=bnb_config,
                                    attn_implementation=ATTN,
                                    trust_remote_code=True, token=HF_TOKEN, **place_kw)
            logger.info(f"loaded checkpoint via {loader_name}")
            break
        except Exception as e:
            logger.warning(f"{loader_name} load failed: {type(e).__name__}: {e}")
            full = None
            free_gpu()
    if full is None:
        raise RuntimeError(f"could not load {spec['model_id']} by any loader")

    backbone, where = _find_text_backbone(full)
    if backbone is None:
        backbone, where = getattr(full, "model", full), "fallback .model"
    logger.info(f"using text backbone at '{where}'")

    root = full.model if hasattr(full, "model") else full
    for attr in ("visual", "vision_tower", "vision_model", "multi_modal_projector"):
        if hasattr(root, attr) and getattr(root, attr) is not None:
            setattr(root, attr, None)          # text-only task: drop the vision stack
    if hasattr(full, "lm_head"):
        full.lm_head = None                     # tied to embeddings; the head we use is .score

    model = GemmaTextClassifier(backbone, num_labels=2)
    del full, backbone
    free_gpu()

    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=spec["grad_checkpointing"])

    targets = resolve_targets(model, cfg["targets"])
    lora_config = LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], target_modules=targets,
                             lora_dropout=cfg["dropout"], bias="none", task_type="SEQ_CLS",
                             modules_to_save=["score"])
    model = get_peft_model(model, lora_config)
    return tokenizer, model


# --------------------------------------------------------------------------
# data / train / predict / calibrate  (unchanged logic)
# --------------------------------------------------------------------------
def get_split(split, dataset, project_root):
    dat_dir = os.path.join(project_root, "dat")
    df = pd.read_csv(os.path.join(dat_dir, dataset, "proc", "df_together.csv"), index_col=0)
    split_df = pd.read_csv(os.path.join(dat_dir, dataset, "proc", "split.csv"), index_col=0)
    df = df[["Temp_sentence", "label"]]
    return (df.loc[split_df[split_df["SPLIT"].isin(split["train"])].index],
            df.loc[split_df[split_df["SPLIT"].isin(split["dev"])].index],
            df.loc[split_df[split_df["SPLIT"].isin(split["test"])].index])


def tokenize(df, tokenizer):
    def preprocess(examples):
        out = tokenizer(examples["Temp_sentence"], truncation=True, max_length=MAX_LEN)
        out["labels"] = examples["label"]
        return out
    ds = Dataset.from_pandas(df)
    return ds.map(preprocess, batched=True, remove_columns=ds.column_names)


_TA_FIELDS = None
_TA_CRITICAL = {
    "output_dir", "optim", "learning_rate", "weight_decay", "max_steps", "warmup_steps",
    "lr_scheduler_type", "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "gradient_checkpointing", "eval_steps", "save_steps",
    "logging_steps", "eval_strategy", "save_strategy", "save_total_limit",
    "load_best_model_at_end", "bf16", "label_names",
}
_TA_ALIASES = {"eval_strategy": "evaluation_strategy"}   # renamed across versions


def _filter_ta_kwargs(kw):
    """venv_gemma and venv_gemma4 run different transformers majors. Drop optional
    kwargs this version does not know, but refuse to silently drop anything that
    would change the training recipe."""
    global _TA_FIELDS
    if _TA_FIELDS is None:
        try:
            import dataclasses
            _TA_FIELDS = {f.name for f in dataclasses.fields(TrainingArguments)}
        except Exception:
            import inspect
            _TA_FIELDS = set(inspect.signature(TrainingArguments.__init__).parameters)

    out, dropped = {}, []
    for k, v in kw.items():
        if k in _TA_FIELDS:
            out[k] = v
        elif k in _TA_ALIASES and _TA_ALIASES[k] in _TA_FIELDS:
            out[_TA_ALIASES[k]] = v
            logger.info(f"TrainingArguments: using '{_TA_ALIASES[k]}' for '{k}' on this version")
        else:
            dropped.append(k)

    fatal = [k for k in dropped if k in _TA_CRITICAL]
    if fatal:
        raise RuntimeError(
            f"this transformers build does not accept {fatal}, which would change the "
            f"training recipe. Refusing to run rather than produce incomparable folds.")
    for k in dropped:
        logger.warning(f"TrainingArguments: dropping unsupported optional kwarg '{k}'")
    return out


def train_peft(tokenizer, model, tokenized_train, tokenized_dev, output_dir,
               per_device_train_batch_size, per_device_eval_batch_size,
               gradient_accumulation_steps, learning_rate, grad_checkpointing,
               max_steps=30000, warmup_steps=200, eval_steps=200, save_steps=200,
               resume=True, pad_to_multiple_of=None):
    # pad_to_multiple_of aligns the GEMMs; padded positions are masked, so no
    # number changes. Everything else here matches the original helper exactly.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            pad_to_multiple_of=pad_to_multiple_of or None)

    ta_kwargs = dict(
        report_to="none",
        output_dir=output_dir,
        do_train=True, do_eval=True, do_predict=True,
        optim="paged_adamw_8bit",
        learning_rate=learning_rate,
        weight_decay=0.01,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_checkpointing else None,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        save_steps=save_steps,
        logging_steps=eval_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        bf16=True,
        label_names=["labels"],
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )
    if not grad_checkpointing:
        ta_kwargs.pop("gradient_checkpointing_kwargs", None)
    training_args = TrainingArguments(**_filter_ta_kwargs(ta_kwargs))

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized_train, eval_dataset=tokenized_dev,
        processing_class=tokenizer, data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10,
                                         early_stopping_threshold=0.001)],
    )

    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    ckpts = [c for c in ckpts if re.search(r"checkpoint-\d+$", c)]
    if resume and ckpts:
        last = max(ckpts, key=lambda p: int(p.rsplit("-", 1)[1]))
        logger.info(f"resuming mid-fold from {os.path.basename(last)}")
        trainer.train(resume_from_checkpoint=last)
    else:
        trainer.train()
    return trainer


def predict(trainer, tokenized_dev, tokenized_test):
    pd_dev = trainer.predict(tokenized_dev)
    prob_dev = F.softmax(torch.tensor(pd_dev.predictions), dim=-1)
    pd_test = trainer.predict(tokenized_test)
    prob_test = F.softmax(torch.tensor(pd_test.predictions), dim=-1)
    return prob_dev[:, 1].tolist(), prob_test[:, 1].tolist()


def calibration(prob_dev, labels_dev, prob_test):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(prob_dev, labels_dev)
    return ir.transform(prob_test)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def resolve_pad(args):
    if args.pad_to_multiple_of >= 0:
        return args.pad_to_multiple_of
    return 8 if args.per_device_train_batch_size > 1 else 0


def parse_args(spec):
    ap = argparse.ArgumentParser(description=f"{spec['key']} QLoRA -- true 5-fold, single GPU")
    ap.add_argument("--dataset", default="Analgesics-induced_acute_liver_failure")
    ap.add_argument("--config", default="paper", choices=list(CONFIGS.keys()))
    ap.add_argument("--per_device_train_batch_size", type=int, default=spec["train_bs"])
    ap.add_argument("--gradient_accumulation_steps", type=int, default=spec["accum"])
    ap.add_argument("--per_device_eval_batch_size", type=int, default=spec["eval_bs"])
    ap.add_argument("--pad_to_multiple_of", type=int, default=-1,
                    help="GEMM alignment for the collator. -1 = auto (8 when the train "
                         "batch is >1, off at batch 1 where dynamic padding already "
                         "wastes nothing). Padded positions are masked, so this changes "
                         "no number either way.")
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--folds", default="all",
                    help="comma-separated tags (e.g. 21,32) or 'all'")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore mid-fold checkpoints and start folds from scratch")
    ap.add_argument("--dry-run-model", action="store_true",
                    help="build model + LoRA, print params and memory, exit")
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="run N real training steps on a subsample and report peak memory")
    return ap.parse_known_args()[0]


def main(spec):
    spec = dict(spec)
    if os.environ.get("GRAD_CKPT") is not None:          # SLURM/env override
        spec["grad_checkpointing"] = os.environ["GRAD_CKPT"] not in ("0", "false", "False", "")
    args = parse_args(spec)
    cfg = CONFIGS[args.config]
    lr = args.learning_rate if args.learning_rate is not None else cfg["lr"]
    dataset = args.dataset
    model_key = f"{spec['key']}_{args.config}"

    eff = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if eff != EFFECTIVE_BATCH and not (args.probe or args.dry_run_model):
        sys.exit(f"effective batch is {eff}, must be {EFFECTIVE_BATCH} to stay comparable "
                 f"with the rest of the study (train_bs x accum)")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_base = os.path.join(project_root, "outputs", model_key, dataset)
    trainlog_dir = os.path.join(out_base, "train_logs")
    runlog_dir = os.path.join(out_base, "run_logs")
    pred_dir = os.path.join(project_root, "dat", dataset, "proc", f"cross_val_{model_key}_temp")
    for d in (trainlog_dir, runlog_dir, pred_dir):
        os.makedirs(d, exist_ok=True)

    run_tag = os.environ.get("SLURM_JOB_ID") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(runlog_dir,
                                                   f"{model_key}_{dataset}_{run_tag}.log"))])

    # checkpoints go to FOLD_SCRATCH; the SLURM script points that at project
    # storage for the 31B so a walltime kill costs <=200 steps, not a whole fold.
    scratch_root = os.environ.get("FOLD_SCRATCH", os.path.join(project_root, "scratch"))

    logger.info(f"=== {spec['key']} QLoRA [{args.config}] | {dataset} | true 5-fold ===")
    logger.info(f"model={spec['model_id']} r={cfg['r']} alpha={cfg['alpha']} "
                f"targets={cfg['targets']} lr={lr} dropout={cfg['dropout']} "
                f"max_len={MAX_LEN} attn={ATTN}")
    logger.info(f"batch: train={args.per_device_train_batch_size} x "
                f"accum={args.gradient_accumulation_steps} = {eff} | "
                f"eval={args.per_device_eval_batch_size} | "
                f"grad_checkpointing={spec['grad_checkpointing']} | "
                f"placement={spec['placement']} | "
                f"pad_to_multiple_of={resolve_pad(args) or 'off'}")

    if args.dry_run_model:
        set_seed(42)
        _, m = build_model(spec, cfg, args)
        tp = sum(p.numel() for p in m.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in m.parameters())
        logger.info(f"DRY RUN OK -- trainable {tp:,} / {tot:,} ({100*tp/tot:.3f}%)")
        log_mem("after build")
        return

    possible_splits = []
    for test in range(5):
        dev = (test + 1) % 5
        possible_splits.append({"dev": [dev], "test": [test],
                                "train": [x for x in range(5) if x not in (test, dev)]})
    if args.folds != "all":
        want = {f.strip() for f in args.folds.split(",")}
        possible_splits = [s for s in possible_splits
                           if f"{s['dev'][0]}{s['test'][0]}" in want]

    if args.probe:
        probe(spec, cfg, args, lr, possible_splits[0], dataset, project_root, scratch_root)
        return

    done = ran = 0
    for split in possible_splits:
        tag = f"{split['dev'][0]}{split['test'][0]}"
        if os.path.exists(os.path.join(pred_dir, f"df_res{tag}.csv")):
            logger.info(f"[fold {tag}] already done -> skipping")
            done += 1
            continue
        logger.info(f"---- fold {tag} ({ran+done+1}/{len(possible_splits)}) ----")
        run_fold(spec, cfg, args, lr, split, dataset, project_root,
                 scratch_root, model_key, pred_dir, trainlog_dir)
        ran += 1
    logger.info(f"=== [{args.config}] {dataset} complete: {ran} trained, {done} skipped, "
                f"{ran+done}/5 ===")


def run_fold(spec, cfg, args, lr, split, dataset, project_root,
             scratch_root, model_key, pred_dir, trainlog_dir):
    tag = f"{split['dev'][0]}{split['test'][0]}"
    t0 = time.time()

    free_gpu()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    log_mem(f"fold {tag} pre-build")

    set_seed(42)
    tokenizer, model = build_model(spec, cfg, args)
    log_mem(f"fold {tag} post-build")
    if not getattr(run_fold, "_logged", False):
        tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        logger.info(f"trainable params: {tp:,} / {tot:,} ({100*tp/tot:.3f}%)")
        run_fold._logged = True

    df_train, df_dev, df_test = get_split(split, dataset, project_root)
    tk_train = tokenize(df_train, tokenizer)
    tk_dev = tokenize(df_dev, tokenizer)
    tk_test = tokenize(df_test, tokenizer)

    scratch = os.path.join(scratch_root, f"{model_key}_{dataset}_{tag}")
    os.makedirs(scratch, exist_ok=True)

    trainer = train_peft(
        tokenizer, model, tk_train, tk_dev, scratch,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=lr, grad_checkpointing=spec["grad_checkpointing"],
        resume=not args.no_resume, pad_to_multiple_of=resolve_pad(args))

    prob_dev, prob_test = predict(trainer, tk_dev, tk_test)
    prob_test_cal = calibration(prob_dev, df_dev["label"].tolist(), prob_test)

    try:
        pd.DataFrame(trainer.state.log_history).to_csv(
            os.path.join(trainlog_dir, f"train_curve_{tag}.csv"), index=False)
    except Exception as e:
        logger.warning(f"[fold {tag}] curve save failed: {e}")

    df_res = pd.DataFrame(index=df_test.index)
    df_res[f"{model_key}_temp"] = prob_test
    df_res[f"{model_key}_temp_cal"] = prob_test_cal
    df_res["label"] = df_test["label"]
    df_res.to_csv(os.path.join(pred_dir, f"df_res{tag}.csv"))
    logger.info(f"[fold {tag}] wrote predictions -> df_res{tag}.csv")

    log_mem(f"fold {tag} end-of-fold")
    logger.info(f"[fold {tag}] wall time {(time.time()-t0)/3600:.2f} h")
    freed = teardown_fold(trainer, model)
    del trainer, model, tokenizer
    free_gpu()
    log_mem(f"fold {tag} post-teardown (released {freed/2**30:.2f}GiB)")
    shutil.rmtree(scratch, ignore_errors=True)   # only after df_res is safely on disk


def probe(spec, cfg, args, lr, split, dataset, project_root, scratch_root):
    """A few real steps at the requested geometry -> peak memory and s/step."""
    logger.info(f"=== PROBE: {args.probe} steps at train_bs="
                f"{args.per_device_train_batch_size}, eval_bs={args.per_device_eval_batch_size} ===")
    free_gpu()
    torch.cuda.reset_peak_memory_stats()
    set_seed(42)
    tokenizer, model = build_model(spec, cfg, args)
    log_mem("probe post-build")

    df_train, df_dev, _ = get_split(split, dataset, project_root)
    # take the longest sequences: worst case for dynamic padding
    lens = df_train["Temp_sentence"].astype(str).str.len()
    df_train = df_train.loc[lens.sort_values(ascending=False).index[:2048]]
    tk_train = tokenize(df_train, tokenizer)
    tk_dev = tokenize(df_dev.head(512), tokenizer)

    scratch = os.path.join(scratch_root, f"probe_{spec['key']}")
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, exist_ok=True)
    t0 = time.time()
    train_peft(tokenizer, model, tk_train, tk_dev, scratch,
               per_device_train_batch_size=args.per_device_train_batch_size,
               per_device_eval_batch_size=args.per_device_eval_batch_size,
               gradient_accumulation_steps=args.gradient_accumulation_steps,
               learning_rate=lr, grad_checkpointing=spec["grad_checkpointing"],
               max_steps=args.probe, warmup_steps=0,
               eval_steps=max(args.probe, 1), save_steps=10 ** 9, resume=False,
               pad_to_multiple_of=resolve_pad(args))
    dt = time.time() - t0
    log_mem("probe peak")
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    logger.info(f"PROBE: {args.probe} steps in {dt:.0f}s ({dt/max(args.probe,1):.1f}s/step, "
                f"includes one eval pass)")
    logger.info(f"PROBE: peak {peak:.2f} GiB of {total:.2f} GiB "
                f"-> {total-peak:.2f} GiB headroom on the worst-case batches")
    shutil.rmtree(scratch, ignore_errors=True)
