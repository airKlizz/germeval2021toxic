import random

import numpy as np
import torch
import typer
from datasets import load_metric
from loguru import logger
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from dataset import load_test, load_train

app = typer.Typer()


@app.command()
def train(
    seed: int = 42,
    train_csv: str = "data/train.csv",
    train_train_csv: str = "data/train.train.csv",
    train_test_csv: str = "data/train.test.csv",
    train_ratio: int = 0.8,
    num_labels: int = 3,
    model_checkpoint: str = "deepset/gbert-base",
    output_dir: str = "models/",
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    nb_epoch: int = 5,
):
    random.seed(seed)
    output_dir += model_checkpoint
    logger.info(f"Load the model: {model_checkpoint}.")
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=nb_epoch,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        logging_steps=10,
    )

    metric = load_metric("metrics/singleclass.py")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.sign(predictions)
        return metric.compute(predictions=predictions.astype("int32"), references=labels.astype("int32"))

    logger.info("Load and preprocess the dataset.")
    train_dataset = load_train(train_train_csv, model_checkpoint, preprocess=True)
    test_dataset = load_test(train_test_csv, model_checkpoint, preprocess=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Start the training.")
    trainer.train()

    logger.info("Start the evaluation.")
    trainer.evaluate()

if __name__ == "__main__":
    app()