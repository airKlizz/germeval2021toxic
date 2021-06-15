import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import typer
from datasets import load_metric
from loguru import logger
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification,
                          MT5ForConditionalGeneration)
from pathlib import Path

from dataset import balance_evaluation, load
import json

app = typer.Typer()


@app.command()
def best_checkpoint(folder: str):
    states_files = list(Path(folder).glob("**/*state.json"))
    for f in states_files:
        logger.info(f"{f} - {find_best_checkpoint_from_states(f)}")
    
def find_best_checkpoint_from_states(states_file):
    with open(states_file) as f:
        data = json.load(f)
    return data["best_model_checkpoint"]
    

@app.command()
def predict(
    test_csv: str = "data/train.test.csv",
    labels: List[str] = ["Sub1_Toxic"],
    model_checkpoint: str = "deepset/gbert-base",
    model_type: str = "auto",
    batch_size: int = 16,
    max_length: int = 256,
    balanced: bool = False,
):
    logger.info(f"Start singleclass prediction.")
    logger.info(f"Load the model: {model_checkpoint}.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == "auto":
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(device)
    elif model_type == "t5":
        model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint).to(device)
    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")

    metric = load_metric("metrics/singleclass.py")

    if model_type == "auto":

        def get_predictions(outputs):
            return np.argmax(outputs.logits.tolist(), axis=1).tolist()

        def get_labels(labels):
            labels = labels.cpu()
            labels = np.where(labels == -1.0, 0, labels)
            labels = np.where(labels == 1.0, 1, labels)
            return labels.tolist()

    elif model_type == "t5":

        def get_predictions(outputs):
            logits = outputs.logits.squeeze(1)
            selected_logits = logits[:, [59006, 112560]]
            probs = F.softmax(selected_logits, dim=1)
            return np.argmax(probs.tolist(), axis=1).tolist()

        def get_labels(labels):
            labels = labels.cpu()
            labels = np.where(labels == 59006, 0, labels)
            labels = np.where(labels == 112560, 1, labels)
            return labels.tolist()

    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")

    logger.info("Load and preprocess the dataset.")
    logger.debug(f"test_csv: {test_csv}")
    dataset = load(test_csv, model_checkpoint, model_type, preprocess=True, labels=labels, max_length=max_length)
    if model_type == "auto":
        columns = ["input_ids", "token_type_ids", "attention_mask", "labels"]
    elif model_type == "t5":
        columns = ["input_ids", "attention_mask", "decoder_input_ids", "labels"]
    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")
    dataset.set_format(type="torch", columns=columns)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    all_labels = []
    all_predictions = []
    for batch in tqdm(dataloader, desc="In progress..."):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = get_labels(batch.pop("labels"))
        outputs = model(**batch)
        predictions = get_predictions(outputs)
        assert len(predictions) == len(labels)
        all_labels += labels
        all_predictions += predictions

    if balanced:
        all_labels, all_predictions = balance_evaluation(all_labels, all_predictions)
    stats = metric.compute(predictions=all_predictions, references=all_labels)
    print(stats)
    return stats


@app.command()
def create_submission(
    test_csv: str = "data/toxic_kaggle/test.csv",
    model_checkpoint: str = "deepset/gbert-base",
    model_type: str = "auto",
    batch_size: int = 16,
    max_length: int = 256,
    output_file: str = "submission.csv",
):
    logger.info(f"Start singleclass prediction.")
    logger.info(f"Load the model: {model_checkpoint}.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_type == "auto":
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(device)
    elif model_type == "t5":
        model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint).to(device)
    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")

    if model_type == "auto":

        def get_predictions(outputs):
            return np.argmax(outputs.logits.tolist(), axis=1).tolist()

    elif model_type == "t5":

        def get_predictions(outputs):
            logits = outputs.logits.squeeze(1)
            selected_logits = logits[:, [59006, 112560]]
            probs = F.softmax(selected_logits, dim=1)
            return np.argmax(probs.tolist(), axis=1).tolist()

    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")

    logger.info("Load and preprocess the dataset.")
    logger.debug(f"test_csv: {test_csv}")
    dataset = load(test_csv, model_checkpoint, model_type, preprocess=True, labels=[], max_length=max_length)
    if model_type == "auto":
        columns = ["input_ids", "token_type_ids", "attention_mask"]
    elif model_type == "t5":
        columns = ["input_ids", "attention_mask", "decoder_input_ids"]
    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")
    dataset.set_format(type="torch", columns=columns)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    all_predictions = []
    for batch in tqdm(dataloader, desc="In progress..."):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = get_predictions(outputs)
        all_predictions += predictions

    ids = dataset["id"]
    df = pd.DataFrame(columns=["id", "prediction"], data=zip(*[ids, all_predictions]))
    df.to_csv(output_file)


if __name__ == "__main__":
    app()
