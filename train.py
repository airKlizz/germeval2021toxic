import random
from typing import List, Union

import numpy as np
import torch
import typer
from datasets import load_metric
from loguru import logger
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from dataset import load

app = typer.Typer()


@app.command()
def multiclass(
    train_csv: List[str] = ["data/train.train.csv"],
    test_csv: str = "data/train.test.csv",
    model_checkpoint: str = "deepset/gbert-base",
    output_dir: str = "models/multiclass/",
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    nb_epoch: int = 5,
    max_length: str = None,
):
    logger.info(f"Start multiclass training.")
    output_dir += (
        model_checkpoint.replace("/", "_")
        + "_bs="
        + str(batch_size)
        + "_lr="
        + str(learning_rate)
        + "_epoch="
        + str(nb_epoch)
    )
    logger.info(f"Load the model: {model_checkpoint}.")
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

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

    metric = load_metric("metrics/flat_multiclass.py")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.sign(predictions)
        return metric.compute(predictions=predictions.astype("int32"), references=labels.astype("int32"))

    logger.info("Load and preprocess the dataset.")
    train_dataset = load(train_csv, model_checkpoint, preprocess=True, max_length=max_length)
    test_dataset = load(test_csv, model_checkpoint, preprocess=True, max_length=max_length)
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


@app.command()
def singleclass(
    train_csv: List[str] = ["data/train.train.csv"],
    test_csv: str = "data/train.test.csv",
    label: int = 0,
    model_checkpoint: str = "deepset/gbert-base",
    output_dir: str = "models/singleclass/",
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    nb_epoch: int = 5,
    max_length: str = None,
):
    logger.info(f"Start singleclass training.")
    output_dir += (
        model_checkpoint.replace("/", "_")
        + "_label="
        + str(label)
        + "_languages="
        + "+".join(train_csv).replace("data/", "").replace("/", "_")
        + "_bs="
        + str(batch_size)
        + "_lr="
        + str(learning_rate)
        + "_epoch="
        + str(nb_epoch)
    )
    logger.info(f"Load the model: {model_checkpoint}.")
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
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
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    logger.info("Load and preprocess the dataset.")
    logger.debug(f"train_csv: {train_csv}")
    logger.debug(f"test_csv: {test_csv}")
    train_dataset = load(
        train_csv, model_checkpoint, preprocess=True, num_labels=1, label=label, max_length=max_length
    )
    test_dataset = load(test_csv, model_checkpoint, preprocess=True, num_labels=1, label=label, max_length=max_length)
    logger.info(f"Dataset sample: {train_dataset[0]}")
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


@app.command()
def hyperparameter_search_singleclass(
    train_csv: List[str] = ["data/train.train.csv"],
    test_csv: str = "data/train.test.csv",
    label: int = 0,
    model_checkpoint: str = "deepset/gbert-base",
    output_dir: str = "models/hyperparameter_search_singleclass/",
    max_length: str = None,
):
    logger.info(f"Start singleclass training.")
    output_dir += (
        model_checkpoint.replace("/", "_")
        + "_label="
        + str(label)
        + "_languages="
        + "+".join(train_csv).replace("data/", "").replace("/", "_")
    )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir="./logs",
        logging_steps=10,
    )

    metric = load_metric("metrics/singleclass.py")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    logger.info("Load and preprocess the dataset.")
    logger.debug(f"train_csv: {train_csv}")
    logger.debug(f"test_csv: {test_csv}")
    train_dataset = load(
        train_csv, model_checkpoint, preprocess=True, num_labels=1, label=label, max_length=max_length
    )
    test_dataset = load(test_csv, model_checkpoint, preprocess=True, num_labels=1, label=label, max_length=max_length)
    logger.info(f"Dataset sample: {train_dataset[0]}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Start the hyperparameter search.")
    best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")

    logger.info(f"Best run: {best_run}")

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    logger.info("Start final training.")
    trainer.train()

    logger.info("Start the evaluation.")
    trainer.evaluate()


if __name__ == "__main__":
    app()
