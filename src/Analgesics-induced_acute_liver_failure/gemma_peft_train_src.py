"""
Helper for the 5-fold PEFT (QLoRA) sequence-classification runs (Gemma3-4B /
MedGemma-1.5-4B). IDENTICAL to the BioMistral/Med-LLaMA helper -- reuses the base
paper's recipe (4-bit nf4 + LoRA + paged_adamw_8bit) and keeps get_split / predict /
isotonic-calibration identical to the rest of the study, so the joint statistics stay
valid across all models.

SINGLE GPU: frozen 4-bit base + small LoRA adapters fit on one L40 -> no DeepSpeed /
ZeRO-3 / NCCL / P2P. Efficient defaults: SDPA attention + real batch 16 (set in the
driver), dataloader workers here.
"""
import os
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import (DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments)
from sklearn.isotonic import IsotonicRegression


def get_split(split, dataset):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dat_dir = os.path.join(project_root, "dat")
    df = pd.read_csv(os.path.join(dat_dir, dataset, "proc", "df_together.csv"), index_col=0)
    split_df = pd.read_csv(os.path.join(dat_dir, dataset, "proc", "split.csv"), index_col=0)
    df = df[["Temp_sentence", "label"]]
    df_train = df.loc[split_df[split_df["SPLIT"].isin(split["train"])].index]
    df_dev = df.loc[split_df[split_df["SPLIT"].isin(split["dev"])].index]
    df_test = df.loc[split_df[split_df["SPLIT"].isin(split["test"])].index]
    return df_train, df_dev, df_test


def train_peft(tokenizer, model, tokenized_train, tokenized_dev, output_dir,
               per_device_train_batch_size=16, per_device_eval_batch_size=16,
               gradient_accumulation_steps=2, learning_rate=2e-4,
               max_steps=30000, warmup_steps=200, eval_steps=200, save_steps=200):
    # effective batch = 16 x 2 = 32 (same as the encoders); paged_adamw_8bit optimizer.
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        report_to="none",
        output_dir=output_dir,
        do_train=True, do_eval=True, do_predict=True,
        optim="paged_adamw_8bit",
        learning_rate=learning_rate,        # per-config LoRA lr (passed by the driver)
        weight_decay=0.01,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
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

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized_train, eval_dataset=tokenized_dev,
        processing_class=tokenizer, data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.001)],
    )
    trainer.train()
    return trainer


def predict(trainer, tokenized_dev, tokenized_test):
    pd_dev = trainer.predict(tokenized_dev)
    prob_dev = F.softmax(torch.tensor(pd_dev.predictions), dim=-1)
    pd_test = trainer.predict(tokenized_test)
    prob_test = F.softmax(torch.tensor(pd_test.predictions), dim=-1)
    return prob_dev[:, 1].tolist(), prob_test[:, 1].tolist()


def calibration(prob_dev, labels_dev, prob_test):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(prob_dev, labels_dev)
    return ir.transform(prob_test)
