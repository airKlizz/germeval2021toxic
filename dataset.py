import random

from datasets import load_dataset
from transformers import AutoTokenizer


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


def load(csv, model_checkpoint=None, preprocess=False, num_labels=3, label=None):
    if isinstance(csv, str):
        csv = [csv]
    data = load_dataset("csv", data_files=csv)
    dataset = data["train"]
    if preprocess:
        return preprocess_dataset(dataset, model_checkpoint, num_labels, label)
    return dataset


def preprocess_dataset(dataset, model_checkpoint, num_labels=3, label=None):

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def preprocess_function(examples):
        output = tokenizer(examples["comment_text"], padding="max_length", truncation=True)
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
