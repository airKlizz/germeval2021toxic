import random
import re
from typing import List

import emoji
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


def oversampling(train_train_csv, train_train_oversampling_csv, label, precise=False):
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
    if precise:
        new_data_label_1 += data_label_1.copy()[: len(data_label_0) % len(data_label_1)]
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


@app.command()
def combine_two_datasets(output_csv, csv1, csv2, comment_text1, comment_text2, label1, label2):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    texts = df1[comment_text1].values.tolist() + df2[comment_text2].values.tolist()
    labels = df1[label1].values.tolist() + df2[label2].values.tolist()
    data = list(zip(texts, labels))
    df = pd.DataFrame(data=data, columns=[comment_text1, label1])
    df.to_csv(output_csv)


def load(csv, model_checkpoint=None, model_type="auto", preprocess=False, labels=None, max_length=None, shuffle=True):
    if isinstance(csv, str):
        csv = [csv]
    csv = list(csv)
    data = load_dataset("csv", data_files=csv)
    dataset = data["train"]
    if preprocess:
        return preprocess_dataset(dataset, model_checkpoint, model_type, labels, max_length).shuffle(seed=42)
    if shuffle:
        return dataset.shuffle(seed=42)
    return dataset


def remove_in_word_whitespaces(comment):
    find = re.findall(r"(^| )(([a-zA-z????????] ){1,}[a-zA-z????????!?,.]([^a-zA-z????????]|$))", comment)
    if len(find) > 0:
        for match in find:
            found = match[0] + match[1]
            replacement = " " + re.sub(r" ", "", found) + " "
            comment = comment.replace(found, replacement, 1)
    return comment


def demojize(comment):
    return emoji.demojize(comment, delimiters=(" <", "> "))


def preprocess_dataset(dataset, model_checkpoint, model_type, labels=None, max_length=None):

    if model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def preprocess_function(examples, labels=labels):

        # pre-process the input text
        ## insert whitespaces before and after emojis so they are tokenized as separate tokens
        examples["comment_text_preprocessed"] = list(map(lambda t: demojize(t), examples["comment_text"]))
        examples["comment_text_preprocessed"] = list(map(lambda t: emoji.emojize(t, delimiters=("<", ">")), examples["comment_text_preprocessed"]))
        ## convert terms like "a k t u e l l" to "aktuell"
        examples["comment_text_preprocessed"] = list(map(lambda t: remove_in_word_whitespaces(t), examples["comment_text_preprocessed"]))
        ## trim mutliple whitespace characters
        examples["comment_text_preprocessed"] = list(map(lambda t: re.sub(r" {2,}", " ", t), examples["comment_text_preprocessed"]))
        ## strip outer whitespaces
        examples["comment_text_preprocessed"] = list(map(lambda t: t.strip(), examples["comment_text_preprocessed"]))

        # input
        if model_type == "t5":
            output = tokenizer(
                ["speech review: " + t for t in examples["comment_text_preprocessed"]],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            output["decoder_input_ids"] = [[tokenizer.pad_token_id] for _ in range(len(output["input_ids"]))]
        else:
            output = tokenizer(examples["comment_text_preprocessed"], max_length=max_length, padding="max_length", truncation=True)

        # label
        def get_label(label):
            if label == "Sub1_Toxic" or label == "toxic" or label == "hf":
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
                values = [1 if v == 1 else 0 for v in values]
                labels.append(values if len(values) > 1 else values[0])
        if len(labels) != 0:
            output["labels"] = labels
        return output

    return dataset.map(preprocess_function, batched=True)


@app.command()
def stats(csv: List[str], label_column: str = "hf"):
    if isinstance(csv, str):
        csv = [csv]
    csv = list(csv)
    data = load_dataset("csv", data_files=csv)
    dataset = data["train"]
    print(dataset)
    lengths = []
    labels = []
    for i, entry in enumerate(dataset):
        if not isinstance(entry["comment_text"], str):
            print(i, entry["Unnamed: 0"], entry["comment_text"], entry["hf"])
            continue
        lengths.append(len(entry["comment_text"].split()))
        labels.append(entry[label_column])
    print(
        "Lengths: ",
        describe(lengths),
        "\nLabel: ",
        np.array(labels).mean(),
    )


def balance_evaluation(labels, predictions):
    idx_0 = []
    idx_1 = []
    for i, label in enumerate(labels):
        if label == 0:
            idx_0.append(i)
        elif label == 1:
            idx_1.append(i)
        else:
            raise ValueError
    new_idx_1 = []
    for _ in range(int(len(idx_0) / len(idx_1))):
        new_idx_1 += idx_1.copy()
    new_idx_1 += idx_1.copy()[: len(idx_0) % len(idx_1)]
    new_labels = []
    new_predictions = []
    for i in idx_0 + new_idx_1:
        new_labels.append(labels[i])
        new_predictions.append(predictions[i])
    return new_labels, new_predictions


@app.command()
def random_baseline(csv: str, balanced: bool = False):
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
    if balanced:
        toxic_labels, random_toxic_predictions = balance_evaluation(toxic_labels, random_predictions)
        engaging_labels, random_engaging_predictions = balance_evaluation(engaging_labels, random_predictions)
        factclaiming_labels, random_factclaiming_predictions = balance_evaluation(
            factclaiming_labels, random_predictions
        )
    else:
        random_toxic_predictions = random_predictions
        random_engaging_predictions = random_predictions
        random_factclaiming_predictions = random_predictions
    metric = load_metric("metrics/singleclass.py")
    print(
        "Toxic: ",
        metric.compute(predictions=random_toxic_predictions, references=toxic_labels),
        "\nEngaging: ",
        metric.compute(predictions=random_engaging_predictions, references=engaging_labels),
        "\nFactclaiming: ",
        metric.compute(predictions=random_factclaiming_predictions, references=factclaiming_labels),
    )


@app.command()
def true_baseline(csv: str, balanced: bool = False):
    if isinstance(csv, str):
        csv = [csv]
    csv = list(csv)
    data = load_dataset("csv", data_files=csv)
    dataset = data["train"]
    toxic_labels = []
    engaging_labels = []
    factclaiming_labels = []
    predictions = []
    for entry in dataset:
        toxic_labels.append(entry["Sub1_Toxic"])
        engaging_labels.append(entry["Sub2_Engaging"])
        factclaiming_labels.append(entry["Sub3_FactClaiming"])
        predictions.append(1)
    if balanced:
        toxic_labels, toxic_predictions = balance_evaluation(toxic_labels, predictions)
        engaging_labels, engaging_predictions = balance_evaluation(engaging_labels, predictions)
        factclaiming_labels, factclaiming_predictions = balance_evaluation(factclaiming_labels, predictions)
    else:
        toxic_predictions = predictions
        engaging_predictions = predictions
        factclaiming_predictions = predictions
    metric = load_metric("metrics/singleclass.py")
    print(
        "Toxic: ",
        metric.compute(predictions=toxic_predictions, references=toxic_labels),
        "\nEngaging: ",
        metric.compute(predictions=engaging_predictions, references=engaging_labels),
        "\nFactclaiming: ",
        metric.compute(predictions=factclaiming_predictions, references=factclaiming_labels),
    )


@app.command()
def false_baseline(csv: str, balanced: bool = False):
    if isinstance(csv, str):
        csv = [csv]
    csv = list(csv)
    data = load_dataset("csv", data_files=csv)
    dataset = data["train"]
    toxic_labels = []
    engaging_labels = []
    factclaiming_labels = []
    predictions = []
    for entry in dataset:
        toxic_labels.append(entry["Sub1_Toxic"])
        engaging_labels.append(entry["Sub2_Engaging"])
        factclaiming_labels.append(entry["Sub3_FactClaiming"])
        predictions.append(0)
    if balanced:
        toxic_labels, toxic_predictions = balance_evaluation(toxic_labels, predictions)
        engaging_labels, engaging_predictions = balance_evaluation(engaging_labels, predictions)
        factclaiming_labels, factclaiming_predictions = balance_evaluation(factclaiming_labels, predictions)
    else:
        toxic_predictions = predictions
        engaging_predictions = predictions
        factclaiming_predictions = predictions
    metric = load_metric("metrics/singleclass.py")
    print(
        "Toxic: ",
        metric.compute(predictions=toxic_predictions, references=toxic_labels),
        "\nEngaging: ",
        metric.compute(predictions=engaging_predictions, references=engaging_labels),
        "\nFactclaiming: ",
        metric.compute(predictions=factclaiming_predictions, references=factclaiming_labels),
    )


@app.command()
def combine_train_csvs(
    csvs: List[str] = [
        "data/conan/train.csv",
        "data/fox-news/train.csv",
        "data/germeval18/train.csv",
        "data/HASOC/train.csv",
        "data/hate_speech_mlma/train.csv",
        "data/hate-speech-dataset/train.csv",
        "data/hateoffensive/train.csv",
        "data/idhsd_rio/train.csv",
        "data/IWG_hatespeech/train.csv",
        "data/L-HSAB/train.csv",
        "data/LaCAfe/train.csv",
        "data/okkyibrohim/train.csv",
    ],
):
    df = pd.concat([pd.read_csv(csv) for csv in csvs]).drop(columns=["Unnamed: 0"])
    df.to_csv("data/combination/train.csv")


if __name__ == "__main__":
    app()
