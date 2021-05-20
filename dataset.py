import random

from datasets import load_dataset
from transformers import AutoTokenizer

from parameters import (MODEL_CHECKPOINT, SEED, TRAIN_CSV, TRAIN_RATIO,
                        TRAIN_TEST_CSV, TRAIN_TRAIN_CSV)

random.seed(SEED)


def split(train_ratio=TRAIN_RATIO):
    with open(TRAIN_CSV) as f:
        data = f.read().split("\n")
    header = data[0]
    data = data[1:]
    random.shuffle(data)
    train_data = data[: int(len(data) * train_ratio)]
    test_data = data[int(len(data) * train_ratio) :]
    assert len(train_data) + len(test_data) == len(data)
    with open(TRAIN_TRAIN_CSV, "w") as f:
        f.write(header)
        f.write("\n")
        for elem in train_data:
            f.write(elem)
            f.write("\n")
    with open(TRAIN_TEST_CSV, "w") as f:
        f.write(header)
        f.write("\n")
        for elem in test_data:
            f.write(elem)
            f.write("\n")


def load_train(preprocess=False):
    data = load_dataset("csv", data_files=[TRAIN_TRAIN_CSV])
    dataset = data["train"]
    if preprocess:
        return preprocess_dataset(dataset)
    return dataset


def load_test(preprocess=False):
    data = load_dataset("csv", data_files=[TRAIN_TEST_CSV])
    dataset = data["train"]
    if preprocess:
        return preprocess_dataset(dataset)
    return dataset


def preprocess_dataset(dataset):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

    def preprocess_function(examples):
        output = tokenizer(examples["comment_text"], padding="max_length", truncation=True)
        toxic = examples["Sub1_Toxic"]
        engaging = examples["Sub2_Engaging"]
        factclaiming = examples["Sub3_FactClaiming"]
        labels = []
        for t, e, f in zip(toxic, engaging, factclaiming):
            t = 1. if t == 1 else -1.
            e = 1. if e == 1 else -1.
            f = 1. if f == 1 else -1.
            labels.append([t, e, f])
        output["labels"] = labels
        return output

    return dataset.map(preprocess_function, batched=True)
