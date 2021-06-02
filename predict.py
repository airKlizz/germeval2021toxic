import random
from typing import List

import numpy as np
import torch
import typer
from datasets import load_metric
from loguru import logger
from transformers import (AutoModelForSequenceClassification,
                          MT5ForConditionalGeneration)

from dataset import load

app = typer.Typer()


@app.command()
def predict(
    test_csv: str = "data/train.test.csv",
    labels: List[str] = ["Sub1_Toxic"],
    model_checkpoint: str = "deepset/gbert-base",
    model_type: str = "auto",
    batch_size: int = 16,
    max_length: int = 256,
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
            return np.argmax(outputs.logits.tolist(), axis=1)

        def get_labels(labels):
            return labels

    elif model_type == "t5":

        def get_predictions(outputs):
            logits = outputs.logits.squeeze(1)
            selected_logits = logits[:, [59006, 112560]]
            probs = F.softmax(selected_logits, dim=1)
            return np.argmax(probs.tolist(), axis=1).tolist()

        def get_labels(labels):
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
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = get_labels(batch.pop("labels"))
        outputs = model(**batch)
        predictions = get_predictions(outputs)
        assert len(predictions) == len(labels)
        all_labels += labels
        all_predictions += predictions

    stats = metric.compute(predictions=all_predictions, references=all_labels)
    print(stats)
    return stats


if __name__ == "__main__":
    app()
