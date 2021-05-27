import random

import numpy as np
import torch
import typer
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer

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


def split(train_ratio, train_csv, train_train_csv, train_test_csv, seed):
    random.seed(seed)
    with open(train_csv) as f:
        data = f.read().split("\n")
    header = data[0]
    data = data[1:]
    random.shuffle(data)
    train_data = data[: int(len(data) * train_ratio)]
    test_data = data[int(len(data) * train_ratio) :]
    assert len(train_data) + len(test_data) == len(data)
    with open(train_train_csv, "w") as f:
        f.write(header)
        f.write("\n")
        for elem in train_data:
            f.write(elem)
            f.write("\n")
    with open(train_test_csv, "w") as f:
        f.write(header)
        f.write("\n")
        for elem in test_data:
            f.write(elem)
            f.write("\n")


def load(csv, model_checkpoint=None, model_type="auto", preprocess=False, num_labels=3, label=None, max_length=None):
    if isinstance(csv, str):
        csv = [csv]
    csv = list(csv)
    data = load_dataset("csv", data_files=csv)
    dataset = data["train"]
    if preprocess:
        return preprocess_dataset(dataset, model_checkpoint, model_type, num_labels, label, max_length).shuffle(seed=42)
    return dataset.shuffle(seed=42)


def preprocess_dataset(dataset, model_checkpoint, model_type, num_labels=3, label=None, max_length=None):

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def preprocess_function(examples):
        output = tokenizer(examples["comment_text"], max_length=max_length, padding="max_length", truncation=True)
        if model_type == "t5":
            output["decoder_input_ids"] = torch.ones(output["input_ids"].size(0), 1) * tokenizer.pad_token_id
        toxic = examples["Sub1_Toxic"]
        engaging = examples["Sub2_Engaging"]
        factclaiming = examples["Sub3_FactClaiming"]
        labels = []
        for t, e, f in zip(toxic, engaging, factclaiming):
            if num_labels == 3:
                t = 1.0 if t == 1 else -1.0
                e = 1.0 if e == 1 else -1.0
                f = 1.0 if f == 1 else -1.0
                labels.append([t, e, f])
            elif num_labels == 1:
                assert label is not None and (label == 0 or label == 1 or label == 2)
                if label == 0:
                    labels.append(t)
                elif label == 1:
                    labels.append(e)
                else:
                    labels.append(f)

            else:
                raise NotImplementedError("Preprocessing method implemented only for 1 or 3 labels.")
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
