import random

import numpy as np
import pandas as pd
import torch
import typer
from datasets import load_dataset, load_metric
from tqdm import tqdm
from transformers import AutoTokenizer, T5Tokenizer

app = typer.Typer()


def describe(l):
    l = np.array(l)
    return {
        "min": l.min(),
        "max": l.max(),
        "mean": l.mean(),
        "10th percentile": np.percentile(l, 10),
        "30th percentile": np.percentile(l, 30),
        "50th percentile": np.percentile(l, 50),
        "70th percentile": np.percentile(l, 70),
        "90th percentile": np.percentile(l, 90),
    }


def split(train_ratio, train_csv, train_train_csv, train_test_csv):
    df = pd.read_csv(train_csv)
    df = df.sample(frac=1)
    train_data = df[: int(len(df) * train_ratio)]
    test_data = df[int(len(df) * train_ratio) :]
    train_data.to_csv(train_train_csv)
    test_data.to_csv(train_test_csv)


def oversampling(train_train_csv, train_train_oversampling_csv, label):
    df = pd.read_csv(train_train_csv)
    columns = df.columns
    label_idx = columns.to_list().index(label)
    data = df.values.tolist()
    data_label_0 = []
    data_label_1 = []
    for d in tqdm(data):
        if d[label_idx] == 0:
            data_label_0.append(d)
        else:
            assert d[label_idx] == 1
            data_label_1.append(d)
    assert len(data_label_0) >= len(data_label_1), "Not implemented when label_0 < label_1"
    print("Number of label 0: ", len(data_label_0))
    print("Number of label 1: ", len(data_label_1))
    new_data_label_1 = []
    for _ in range(int(len(data_label_0) / len(data_label_1))):
        new_data_label_1 += data_label_1.copy()
    # new_data_label_1 += data_label_1.copy()[:len(data_label_0)%len(data_label_1)]
    new_data = data_label_0 + new_data_label_1
    new_df = pd.DataFrame(columns=columns, data=new_data).sample(frac=1)
    new_df.to_csv(train_train_oversampling_csv)


def undersampling(train_train_csv, train_train_undersampling_csv, label):
    df = pd.read_csv(train_train_csv)
    columns = df.columns
    label_idx = columns.to_list().index(label)
    data = df.values.tolist()
    data_label_0 = []
    data_label_1 = []
    for d in tqdm(data):
        if d[label_idx] == 0:
            data_label_0.append(d)
        else:
            assert d[label_idx] == 1
            data_label_1.append(d)
    assert len(data_label_0) >= len(data_label_1), "Not implemented when label_0 < label_1"
    print("Number of label 0: ", len(data_label_0))
    print("Number of label 1: ", len(data_label_1))
    new_data_label_0 = random.sample(data_label_0, len(data_label_1))
    new_data = new_data_label_0 + data_label_1
    new_df = pd.DataFrame(columns=columns, data=new_data).sample(frac=1)
    new_df.to_csv(train_train_undersampling_csv)


def load(csv, model_checkpoint=None, model_type="auto", preprocess=False, labels=None, max_length=None):
    if isinstance(csv, str):
        csv = [csv]
    csv = list(csv)
    data = load_dataset("csv", data_files=csv)
    dataset = data["train"]
    if preprocess:
        return preprocess_dataset(dataset, model_checkpoint, model_type, labels, max_length).shuffle(seed=42)
    return dataset.shuffle(seed=42)


def preprocess_dataset(dataset, model_checkpoint, model_type, labels=None, max_length=None):

    if model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def preprocess_function(examples, labels=labels):
        # input
        if model_type == "t5":
            output = tokenizer(
                ["speech review: " + t for t in examples["comment_text"]],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            output["decoder_input_ids"] = [[tokenizer.pad_token_id] for _ in range(len(output["input_ids"]))]
        else:
            output = tokenizer(examples["comment_text"], max_length=max_length, padding="max_length", truncation=True)

        # label
        def get_label(label):
            if label == "Sub1_Toxic" or label == "toxic":
                return [59006, 112560]
            if label == "Sub2_Engaging":
                return [59006, 46151]
            if label == "Sub3_FactClaiming":
                return [59006, 12558]
            raise ValueError(f"label [{label}] not found.")

        idx_to_label = {idx: get_label(label) for idx, label in enumerate(labels)}

        labels_values = []
        for label in labels:
            labels_values.append(examples[label])

        labels = []
        for values in zip(*labels_values):
            if model_type == "t5":
                values = [idx_to_label[idx][v] for idx, v in enumerate(values)]
                labels.append(values if len(values) > 1 else values[0])
            else:
                values = [1.0 if v == 1 else -1.0 for v in values]
                labels.append(values if len(values) > 1 else values[0])

        output["labels"] = labels
        return output

    return dataset.map(preprocess_function, batched=True)


@app.command()
def stats(csv: str, model_checkpoint: str):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    if isinstance(csv, str):
        csv = [csv]
    csv = list(csv)
    data = load_dataset("csv", data_files=csv)
    dataset = data["train"]
    lengths = []
    toxic_labels = []
    engaging_labels = []
    factclaiming_labels = []
    for entry in dataset:
        lengths.append(len(tokenizer(entry["comment_text"])["input_ids"]))
        toxic_labels.append(entry["Sub1_Toxic"])
        engaging_labels.append(entry["Sub2_Engaging"])
        factclaiming_labels.append(entry["Sub3_FactClaiming"])

    print(
        "Lengths: ",
        describe(lengths),
        "\nToxic: ",
        np.array(toxic_labels).mean(),
        "\nEngaging: ",
        np.array(engaging_labels).mean(),
        "\nFactclaiming: ",
        np.array(factclaiming_labels).mean(),
    )


@app.command()
def random_baseline(csv: str):
    if isinstance(csv, str):
        csv = [csv]
    csv = list(csv)
    data = load_dataset("csv", data_files=csv)
    dataset = data["train"]
    toxic_labels = []
    engaging_labels = []
    factclaiming_labels = []
    random_predictions = []
    for entry in dataset:
        toxic_labels.append(entry["Sub1_Toxic"])
        engaging_labels.append(entry["Sub2_Engaging"])
        factclaiming_labels.append(entry["Sub3_FactClaiming"])
        random_predictions.append(random.randint(0, 1))
        # random_predictions.append(0)
    metric = load_metric("metrics/singleclass.py")
    print(
        "Toxic: ",
        metric.compute(predictions=random_predictions, references=toxic_labels),
        "\nEngaging: ",
        metric.compute(predictions=random_predictions, references=engaging_labels),
        "\nFactclaiming: ",
        metric.compute(predictions=random_predictions, references=factclaiming_labels),
    )


if __name__ == "__main__":
    app()
