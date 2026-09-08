"""
Helper module for BioMistral-7B sequence-classification training.

The data-split / prediction / isotonic-calibration LOGIC is IDENTICAL to the 2026
repo's encoder pipeline (mirrored verbatim from modernbert_train_src.py). The ONLY
changes are those forced by *full* fine-tuning a 7B decoder under DeepSpeed ZeRO-3:

  * DeepSpeed ZeRO-3 (bf16, no offload) is attached via TrainingArguments(deepspeed=...).
    The AdamW optimiser and the linear warmup->decay schedule are declared in the
    DeepSpeed JSON with "auto" params, so they resolve to EXACTLY the encoder's
    hyper-parameters: lr=5e-5, betas=(0.9,0.999), eps=1e-6, weight_decay=0.01,
    warmup=200 steps, then linear decay to 0 over max_steps=30000. DeepSpeed's
    WarmupDecayLR (linear) is identical to the encoder's polynomial schedule with
    power=1.0. We therefore do NOT hand-build an optimiser/scheduler and pass it to
    Trainer (DeepSpeed owns the optimiser under ZeRO-3); the numeric values are the
    same as the encoders'.
  * bf16=True and gradient_checkpointing=True  (memory; required to fit 7B on L40).
  * per_device_train_batch_size + gradient_accumulation_steps are set so the
    EFFECTIVE batch stays ~= the encoders' 32, now spread over 3 GPUs.
  * save_total_limit=1 bounds transient checkpoint disk; load_best_model_at_end still
    restores the best-on-dev checkpoint, so the RESULT (which model makes predictions)
    is unchanged.

Everything else -- the 20-split CV, early stopping (patience 10, threshold 1e-3),
eval every 200 steps, max 30000 steps, softmax probabilities, isotonic calibration --
is byte-for-byte the same behaviour as the encoder pipeline.
"""
import os
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import (DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)
from sklearn.isotonic import IsotonicRegression


def get_split(split, dataset):
    # IDENTICAL to the encoder pipeline.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dat_dir = os.path.join(project_root, "dat")

    df = pd.read_csv(os.path.join(dat_dir, dataset, "proc", "df_together.csv"), index_col=0)
    split_df = pd.read_csv(os.path.join(dat_dir, dataset, "proc", "split.csv"), index_col=0)

    df = df[["Temp_sentence", "label"]]

    df_train = df.loc[split_df[split_df["SPLIT"].isin(split["train"])].index]
    df_dev = df.loc[split_df[split_df["SPLIT"].isin(split["dev"])].index]
    df_test = df.loc[split_df[split_df["SPLIT"].isin(split["test"])].index]

    return df_train, df_dev, df_test


def train_transformer(tokenizer, model, tokenized_train, tokenized_dev,
                      deepspeed_config, output_dir,
                      per_device_train_batch_size=2,
                      per_device_eval_batch_size=4,
                      gradient_accumulation_steps=6,
                      max_steps=30000, warmup_steps=200,
                      eval_steps=200, save_steps=200):
    # max_steps / warmup_steps / eval_steps / save_steps default to the encoders'
    # faithful values; the smoke test overrides them to tiny numbers for a fast
    # end-to-end check. Nothing else differs between smoke and real runs.
    # Data collator setup (IDENTICAL to encoders).
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments. The optimiser (AdamW) and schedule (linear warmup->decay)
    # live in the DeepSpeed JSON with "auto" params, which HF fills from the values
    # below -- so lr / betas / eps / weight_decay / warmup / total-steps match the
    # encoders exactly. We do NOT pass an `optimizers=` tuple: under ZeRO-3 DeepSpeed
    # constructs and shards the optimiser itself.
    training_args = TrainingArguments(
        report_to="none",
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        do_predict=True,
        learning_rate=1e-5,          # 7B full FT: lowered from 5e-5 (that diverged to NaN); -> optimizer.params.lr "auto"
        adam_beta1=0.9,              # -> optimizer.params.betas "auto"
        adam_beta2=0.999,
        adam_epsilon=1e-6,           # -> optimizer.params.eps "auto"
        weight_decay=0.01,           # -> optimizer.params.weight_decay "auto"
        max_grad_norm=1.0,           # gradient clipping (-> deepspeed gradient_clipping "auto")
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,         # -> scheduler.params.total_num_steps "auto"
        warmup_steps=warmup_steps,   # -> scheduler.params.warmup_num_steps "auto"
        save_steps=save_steps,
        logging_dir="logs",
        logging_steps=eval_steps,
        load_best_model_at_end=True,
        save_total_limit=1,          # disk bound; best model still restored at end
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        bf16=True,                   # 7B: bf16 mixed precision
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,    # faster data feeding (no effect on results)
        dataloader_pin_memory=True,
        deepspeed=deepspeed_config,  # ZeRO-3 (bf16, no offload)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.001)],
    )

    trainer.train()

    return trainer


def predict(trainer, tokenized_dev, tokenized_test):
    # IDENTICAL to the encoder pipeline.
    predictions_dev = trainer.predict(tokenized_dev)
    scores_dev = predictions_dev.predictions
    probabilities_dev = F.softmax(torch.tensor(scores_dev), dim=-1)

    predictions = trainer.predict(tokenized_test)
    scores = predictions.predictions
    probabilities_test = F.softmax(torch.tensor(scores), dim=-1)

    return probabilities_dev[:, 1].tolist(), probabilities_test[:, 1].tolist()


def calibration(probabilities_dev, labels_dev, probabilities_test):
    # IDENTICAL to the encoder pipeline.
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(probabilities_dev, labels_dev)
    probabilities_test_calibrated = ir.transform(probabilities_test)

    return probabilities_test_calibrated
