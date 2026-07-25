"""
Helper module containing functions to set up data splits and train the BioBERT transformer.
"""
import numpy as np
import torch
from transformers import TrainingArguments, Trainer
import pandas as pd
import os
from transformers import DataCollatorWithPadding, AdamW, EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

def get_split(split, dataset):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dat_dir = os.path.join(project_root, "dat")
    
    df = pd.read_csv(os.path.join(dat_dir, dataset, "proc", "df_together.csv"), index_col=0)
    split_df = pd.read_csv(os.path.join(dat_dir, dataset, "proc", "split.csv"), index_col=0)
    
    df = df[["Temp_sentence", "label"]]
    
    # Correct way to filter rows where SPLIT value is in the list
    df_train = df.loc[split_df[split_df["SPLIT"].isin(split["train"])].index]
    df_dev = df.loc[split_df[split_df["SPLIT"].isin(split["dev"])].index]
    df_test = df.loc[split_df[split_df["SPLIT"].isin(split["test"])].index]
    
    return df_train, df_dev, df_test


def train_transformer(tokenizer, model, tokenized_train, tokenized_dev):
    # Data collator setup
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Custom optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=5e-5,  # Learning rate
        weight_decay=0.01,  # Weight decay rate
        betas=(0.9, 0.999),  # Beta1, Beta2
        eps=1e-6  # Epsilon value
    )

    # Function to exclude specific parameters from weight decay
    def exclude_from_weight_decay(param_name):
        excluded_layers = ["LayerNorm", "layer_norm", "bias"]
        return any(layer in param_name for layer in excluded_layers)

    # Training arguments without `power`
    training_args = TrainingArguments(
        report_to=None,
        output_dir="results",
        do_train=True,
        do_eval=True,
        do_predict=True,
        learning_rate=5e-5,  # This is the initial learning rate; it'll be modified by the scheduler
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        max_steps=30000,
        warmup_steps=200,
        save_steps=200,
        weight_decay=0.01,
        logging_dir='logs',
        logging_steps=200,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=200,
        lr_scheduler_type="polynomial",  # Removed 'power'
    )

    # Custom optimizer setup in Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),  # Pass custom optimizer (scheduler set manually below)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.001)],
    )

    # Apply weight decay only to parameters that aren't in the exclude list
    for name, param in model.named_parameters():
        if not exclude_from_weight_decay(name):
            param.requires_grad = True

    # Scheduler: Polynomial decay with warmup
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,  # Number of warmup steps
        num_training_steps=training_args.max_steps,   # Total number of training steps
        lr_end=0.0,  # End learning rate after decay
        power=1.0     # The polynomial decay power
    )

    # Set the scheduler in the Trainer's optimizer tuple
    trainer.optimizer, trainer.lr_scheduler = optimizer, scheduler

    # Start training
    trainer.train()
    
    return trainer

def predict(trainer, tokenized_dev, tokenized_test):
    # predict on dev data
    predictions_dev = trainer.predict(tokenized_dev)
    scores_dev = predictions_dev.predictions
    probabilities_dev = F.softmax(torch.tensor(scores_dev), dim=-1)
    
    # Predict on test data
    predictions = trainer.predict(tokenized_test)
    # Extract predicted probabilities
    scores = predictions.predictions
    probabilities_test = F.softmax(torch.tensor(scores), dim=-1)
    
    return probabilities_dev[:,1].tolist(), probabilities_test[:,1].tolist()


def calibration(probabilities_dev, labels_dev, probabilities_test):
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(probabilities_dev, labels_dev)
    probabilities_test_calibrated = ir.transform(probabilities_test)
    
    return probabilities_test_calibrated