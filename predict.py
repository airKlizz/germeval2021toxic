import json
import random
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import typer
from datasets import load_metric
from loguru import logger
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification,
                          MT5ForConditionalGeneration)

from dataset import balance_evaluation, load

app = typer.Typer()


@app.command()
def best_checkpoint(folder: str):
    states_files = list(Path(folder).glob("**/*state.json"))
    buckets = {}
    for f in states_files:
        parent = f.parent.parent
        if parent in buckets.keys():
            buckets[parent].append(f)
        else:
            buckets[parent] = [f]
    final_fs = []
    for bucket_files in buckets.values():
        final_f = None
        best_checkpoint = 0
        pattern = r"checkpoint-[0-9]+/trainer_state.json"
        for f in bucket_files:
            checkpoint = int(re.findall(pattern, str(f))[0][11:-19])
            if checkpoint > best_checkpoint:
                final_f = f
        final_fs.append(final_f)
    for f in final_fs:
        logger.info(f"\n{f}\n{find_best_checkpoint_from_states(f)}\n")


def find_best_checkpoint_from_states(states_file):
    with open(states_file) as f:
        data = json.load(f)
    logger.debug(data["best_metric"])
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
    final_columns = []
    for column in columns:
        if column in dataset.column_names:
            final_columns.append(column)
    columns = final_columns

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
def joint_predict(
    test_csv: str = "data/train.test.csv",
    labels: List[str] = ["Sub1_Toxic"],
    model_checkpoints: List[str] = ["deepset/gbert-base"],
    model_types: List[str] = ["auto"],
    batch_size: int = 16,
    max_length: int = 256,
    binary: bool = True,
    output_file: str = "joint_prediction.csv",
    to_file: bool = False,
):
    logger.info(f"Start singleclass joint prediction.")
    metric = load_metric("metrics/singleclass.py")
    dataset_labels = labels
    all_models_predictions = []
    for model_checkpoint, model_type in zip(model_checkpoints, model_types):
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
                if binary:
                    return np.argmax(outputs.logits.tolist(), axis=1).tolist()
                return outputs.logits.tolist()[1]

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
                if binary:
                    return np.argmax(probs.tolist(), axis=1).tolist()
                return probs.tolist()[1]

            def get_labels(labels):
                labels = labels.cpu()
                labels = np.where(labels == 59006, 0, labels)
                labels = np.where(labels == 112560, 1, labels)
                return labels.tolist()

        else:
            raise NotImplementedError("Model type available: 'auto' or 't5'")

        logger.info("Load and preprocess the dataset.")
        logger.debug(f"test_csv: {test_csv}")
        dataset = load(
            test_csv, model_checkpoint, model_type, preprocess=True, labels=dataset_labels, max_length=max_length
        )
        if model_type == "auto":
            columns = ["input_ids", "token_type_ids", "attention_mask", "labels"]
        elif model_type == "t5":
            columns = ["input_ids", "attention_mask", "decoder_input_ids", "labels"]
        else:
            raise NotImplementedError("Model type available: 'auto' or 't5'")
        final_columns = []
        for column in columns:
            if column in dataset.column_names:
                final_columns.append(column)
        columns = final_columns

        dataset.set_format(type="torch", columns=columns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        all_labels = []
        all_predictions = []
        for batch in tqdm(dataloader, desc="In progress..."):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = get_labels(batch.pop("labels"))
            outputs = model(**batch)
            predictions = get_predictions(outputs)
            all_labels += labels
            all_predictions += predictions

        all_models_predictions.append(all_predictions)

    final_predictions = []
    for models_prediction in zip(*all_models_predictions):
        assert len(models_prediction) == len(model_checkpoints)
        final_predictions.append(round(sum(models_prediction) / len(models_prediction)))

    stats = metric.compute(predictions=final_predictions, references=all_labels)
    print(stats)

    try:
        ids = dataset["id"]
    except:
        try:
            ids = dataset["comment_id"]
        except:
            ids = dataset["comment_text"]
    df = pd.DataFrame(columns=["id"] + list(model_checkpoints), data=zip(*([ids] + list(all_models_predictions))))
    print(df.head())
    if to_file:
        df.to_csv(output_file)


@app.command()
def create_submission(
    test_csv: str = "data/test.csv",
    model_checkpoint: str = "deepset/gbert-base",
    model_type: str = "auto",
    batch_size: int = 16,
    max_length: int = 256,
    output_file: str = "submission.csv",
    binary: bool = True,
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
            if binary:
                return np.argmax(outputs.logits.tolist(), axis=1).tolist()
            return outputs.logits.tolist()

    elif model_type == "t5":

        def get_predictions(outputs):
            logits = outputs.logits.squeeze(1)
            selected_logits = logits[:, [59006, 112560]]
            probs = F.softmax(selected_logits, dim=1)
            if binary:
                return np.argmax(probs.tolist(), axis=1).tolist()
            return probs.tolist()

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
    final_columns = []
    for column in columns:
        if column in dataset.column_names:
            final_columns.append(column)
    columns = final_columns

    dataset.set_format(type="torch", columns=columns)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    all_predictions = []
    for batch in tqdm(dataloader, desc="In progress..."):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = get_predictions(outputs)
        all_predictions += predictions

    try:
        ids = dataset["id"]
    except:
        try:
            ids = dataset["comment_id"]
        except:
            ids = dataset["comment_text"]
    if binary:
        df = pd.DataFrame(columns=["id", "prediction"], data=zip(*[ids, all_predictions]))
    else:
        predictions0 = list(list(zip(*all_predictions))[0])
        predictions1 = list(list(zip(*all_predictions))[1])
        df = pd.DataFrame(columns=["id", "prediction0", "prediction1"], data=zip(*[ids, predictions0, predictions1]))
    df.to_csv(output_file)


@app.command()
def create_random_submission(
    test_csv: str = "data/test.csv",
    output_file: str = "submission.csv",
):
    dataset = load(test_csv, preprocess=False)
    try:
        ids = dataset["id"]
    except:
        try:
            ids = dataset["comment_id"]
        except:
            ids = dataset["comment_text"]
    all_predictions = [random.randint(0, 1) for _ in range(len(ids))]
    df = pd.DataFrame(columns=["id", "prediction"], data=zip(*[ids, all_predictions]))
    df.to_csv(output_file)


@app.command()
def predict_official(
    test_csv: str = "data/test.csv",
    truth_csv: str = "data/truth.csv",
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
    dataset = load(test_csv, model_checkpoint, model_type, preprocess=True, labels=[], max_length=max_length)
    if model_type == "auto":
        columns = ["input_ids", "token_type_ids", "attention_mask", "comment_id"]
    elif model_type == "t5":
        columns = ["input_ids", "attention_mask", "decoder_input_ids", "comment_id"]
    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")
    final_columns = []
    for column in columns:
        if column in dataset.column_names:
            final_columns.append(column)
    columns = final_columns

    dataset.set_format(type="torch", columns=columns)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    all_ids = []
    all_predictions = []
    for batch in tqdm(dataloader, desc="In progress..."):
        batch = {k: v.to(device) for k, v in batch.items()}
        ids = get_labels(batch.pop("comment_id"))
        outputs = model(**batch)
        predictions = get_predictions(outputs)
        assert len(predictions) == len(ids)
        all_ids += ids
        all_predictions += predictions

    labels_df = pd.read_csv(truth_csv)
    labels_df = labels_df.set_index("comment_id")
    all_labels = [labels_df.loc[i]["Sub1_Toxic"] for i in all_ids]

    if balanced:
        all_labels, all_predictions = balance_evaluation(all_labels, all_predictions)

    report = classification_report(all_labels, all_predictions, output_dict=True)
    precision_score_1 = report["macro avg"]["precision"]
    recall_score_1 = report["macro avg"]["recall"]
    f1_score_1 = 0
    if precision_score_1 + recall_score_1 > 0:
        f1_score_1 = 2 * precision_score_1 * recall_score_1 / (precision_score_1 + recall_score_1)
    stats = {
        "f1": f1_score_1,
        "recall": recall_score_1,
        "precision": precision_score_1,
    }

    print(stats)
    return stats


if __name__ == "__main__":
    app()
