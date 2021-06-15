from pprint import pprint

import numpy as np
import pandas as pd
import typer
from datasets import load_from_disk, load_metric
from elg import Service
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


from dataset import load
from german_stop_words import GERMAN_STOP_WORDS

app = typer.Typer()


class Patterns:

    toxic_emojis = ["⛔", "⚠", "🔻", "🙉", "🙈", "🙊", "❓", "🤬", "😂", "🤦‍♀️", "🤦‍", "🤮", "💥", "💩"]
    spell = SpellChecker(language="de")
    service = Service.from_docker_image(
        "registry.gitlab.com/european-language-grid/cuni/srv-udpipe:ud-2.4-190531",
        "http://localhost:8080/udpipe/de",
        8384,
    )
    toxic_lemmas = pd.read_csv("german_toxic_words.tsv", sep="\t")["lemma"].values.tolist()

    @classmethod
    def number_of_toxic_emojis(cls, example):
        for emoji in cls.toxic_emojis:
            example[emoji] = int(emoji in example["comment_text"])
            # example[emoji] = example["comment_text"].count(emoji)
        return example

    @classmethod
    def number_of_misspelled_words(cls, example):
        misspelled = cls.spell.unknown(example["comment_text"].split())
        example["number_of_misspelled_words"] = len([w for w in misspelled if "@user" not in w])
        return example

    @classmethod
    def percent_of_misspelled_words(cls, example):
        misspelled = cls.spell.unknown(example["comment_text"].split())
        example["percent_of_misspelled_words"] = len([w for w in misspelled if "@user" not in w]) / len(
            example["comment_text"].split()
        )
        return example

    @classmethod
    def wie_and_adj(cls, example):
        try:
            r = cls.service(example["comment_text"], sync_mode=True)
            example["wie_and_adj"] = 0
            wie = False
            for tok in r.annotations["udpipe/tokens"]:
                if wie == True:
                    if tok.features["words"][0]["upos"] == "ADJ":
                        # example["wie_and_adj"] += 1
                        example["wie_and_adj"] = 1
                    wie = False
                if tok.features["words"][0]["form"].lower() == "wie":
                    wie = True
            return example
        except:
            example["wie_and_adj"] = 0
            logger.error(f"Service call error with input: {example['comment_text']}")
            return example

    @classmethod
    def ppron_and_np(cls, example):
        try:
            r = cls.service(example["comment_text"], sync_mode=True)
            example["ppron_and_np"] = 0
            ppron = False
            for tok in r.annotations["udpipe/tokens"]:
                if ppron == True:
                    if tok.features["words"][0]["upos"] == "NOUN":
                        # example["ppron_and_np"] += 1
                        example["ppron_and_np"] = 1
                    ppron = False
                if (
                    tok.features["words"][0]["form"].lower() == "du"
                    or tok.features["words"][0]["form"].lower() == "ihr"
                ):
                    ppron = True
            return example
        except:
            example["ppron_and_np"] = 0
            logger.error(f"Service call error with input: {example['comment_text']}")
            return example

    @classmethod
    def toxic_words(cls, example):
        try:
            r = cls.service(example["comment_text"], sync_mode=True)
            example["toxic_words"] = 0
            for tok in r.annotations["udpipe/tokens"]:
                if tok.features["words"][0]["lemma"] in cls.toxic_lemmas:
                    # example["toxic_words"] += 1
                    example["toxic_words"] = 1
            return example
        except:
            example["toxic_words"] = 0
            logger.error(f"Service call error with input: {example['comment_text']}")
            return example

    @classmethod
    def all(cls, example):
        example["all"] = []
        for emoji in cls.toxic_emojis:
            example["all"].append(example["comment_text"].count(emoji))
        misspelled = cls.spell.unknown(example["comment_text"].split())
        example["all"].append(len([w for w in misspelled if "@user" not in w]))
        misspelled = cls.spell.unknown(example["comment_text"].split())
        example["all"].append(len([w for w in misspelled if "@user" not in w]) / len(example["comment_text"].split()))
        try:
            r = cls.service(example["comment_text"], sync_mode=True)
            example["wie_and_adj"] = 0
            example["ppron_and_np"] = 0
            example["toxic_words"] = 0
            wie = False
            ppron = False
            for tok in r.annotations["udpipe/tokens"]:
                if wie == True:
                    if tok.features["words"][0]["upos"] == "ADJ":
                        example["wie_and_adj"] += 1
                    wie = False
                if tok.features["words"][0]["form"].lower() == "wie":
                    wie = True
                if ppron == True:
                    if tok.features["words"][0]["upos"] == "NOUN":
                        example["ppron_and_np"] += 1
                    ppron = False
                if (
                    tok.features["words"][0]["form"].lower() == "du"
                    or tok.features["words"][0]["form"].lower() == "ihr"
                ):
                    ppron = True
                if tok.features["words"][0]["lemma"] in cls.toxic_lemmas:
                    example["toxic_words"] += 1
        except:
            example["wie_and_adj"] = 0
            example["ppron_and_np"] = 0
            example["toxic_words"] = 0
            logger.error(f"Service call error with input: {example['comment_text']}")
        example["all"] += [example["wie_and_adj"], example["ppron_and_np"], example["toxic_words"]]
        example.pop("wie_and_adj")
        example.pop("ppron_and_np")
        example.pop("toxic_words")
        return example


@app.command()
def test_correlation_number_of_toxic_emojis(test_csv: str = "data/train.csv"):
    dataset = load(test_csv)
    dataset = dataset.map(Patterns.number_of_toxic_emojis)
    metric = load_metric("metrics/singleclass.py")
    results = {}
    for emoji in Patterns.toxic_emojis:
        predictions = dataset[emoji]
        references = dataset["Sub1_Toxic"]
        results[emoji] = metric.compute(predictions=predictions, references=references)
        results[emoji]["number_of_occurence"] = f"{sum(predictions)}/{len(predictions)}"
    logger.info(pprint(results, indent=2))


@app.command()
def test_correlation_number_of_misspelled_words(test_csv: str = "data/train.csv", threashold: int = 5):
    dataset = load(test_csv)
    dataset = dataset.map(Patterns.number_of_misspelled_words)
    metric = load_metric("metrics/singleclass.py")
    results = {}
    predictions = [0 if nb < threashold else 1 for nb in dataset["number_of_misspelled_words"]]
    references = dataset["Sub1_Toxic"]
    results["number_of_misspelled_words"] = metric.compute(predictions=predictions, references=references)
    results["number_of_misspelled_words"]["number_of_occurence"] = f"{sum(predictions)}/{len(predictions)}"
    logger.info(pprint(results, indent=2))


@app.command()
def test_correlation_wie_and_adj(test_csv: str = "data/train.csv"):
    dataset = load(test_csv)
    dataset = dataset.map(Patterns.wie_and_adj)
    metric = load_metric("metrics/singleclass.py")
    results = {}
    predictions = dataset["wie_and_adj"]
    references = dataset["Sub1_Toxic"]
    results["wie_and_adj"] = metric.compute(predictions=predictions, references=references)
    results["wie_and_adj"]["number_of_occurence"] = f"{sum(predictions)}/{len(predictions)}"
    logger.info(pprint(results, indent=2))


@app.command()
def test_correlation_ppron_and_np(test_csv: str = "data/train.csv"):
    dataset = load(test_csv)
    dataset = dataset.map(Patterns.ppron_and_np)
    metric = load_metric("metrics/singleclass.py")
    results = {}
    predictions = dataset["ppron_and_np"]
    references = dataset["Sub1_Toxic"]
    results["ppron_and_np"] = metric.compute(predictions=predictions, references=references)
    results["ppron_and_np"]["number_of_occurence"] = f"{sum(predictions)}/{len(predictions)}"
    logger.info(pprint(results, indent=2))


@app.command()
def test_toxic_words(test_csv: str = "data/train.csv"):
    dataset = load(test_csv)
    dataset = dataset.map(Patterns.toxic_words)
    metric = load_metric("metrics/singleclass.py")
    results = {}
    predictions = dataset["toxic_words"]
    references = dataset["Sub1_Toxic"]
    results["toxic_words"] = metric.compute(predictions=predictions, references=references)
    results["toxic_words"]["number_of_occurence"] = f"{sum(predictions)}/{len(predictions)}"
    logger.info(pprint(results, indent=2))


@app.command()
def test_correlation_percent_of_misspelled_words(test_csv: str = "data/train.csv", threashold: float = 0.5):
    dataset = load(test_csv)
    dataset = dataset.map(Patterns.percent_of_misspelled_words)
    metric = load_metric("metrics/singleclass.py")
    results = {}
    predictions = [0 if nb < threashold else 1 for nb in dataset["percent_of_misspelled_words"]]
    references = dataset["Sub1_Toxic"]
    results["percent_of_misspelled_words"] = metric.compute(predictions=predictions, references=references)
    results["percent_of_misspelled_words"]["number_of_occurence"] = f"{sum(predictions)}/{len(predictions)}"
    logger.info(pprint(results, indent=2))


@app.command()
def test_all(
    train_csv: str = "data/train.train.csv",
    test_csv: str = "data/train.test.csv",
    train_disk: str = "rules_based_train_dataset",
    test_disk: str = "rules_based_test_dataset",
    from_disk: bool = False,
):
    if from_disk:
        train_dataset = load_from_disk(train_disk)
        test_dataset = load_from_disk(test_disk)
    else:
        train_dataset = load(train_csv)
        train_dataset = train_dataset.map(Patterns.all)
        train_dataset.save_to_disk(train_disk)
        test_dataset = load(test_csv)
        test_dataset = test_dataset.map(Patterns.all)
        test_dataset.save_to_disk(test_disk)
    metric = load_metric("metrics/singleclass.py")
    results = {}
    X = train_dataset["all"]
    y = train_dataset["Sub1_Toxic"]
    class_weights = len(y) / (2 * np.bincount(y))
    sample_weight = [class_weights[label] for label in y]
    #clf = RandomForestClassifier(max_depth=10, random_state=42)
    clf = LogisticRegression(C=24.0)
    clf.fit(X, y)
    X = test_dataset["all"]
    references = test_dataset["Sub1_Toxic"]
    predictions = clf.predict(X)
    results[f"all"] = metric.compute(predictions=predictions, references=references)
    results[f"all"]["number_of_occurence"] = f"{sum(predictions)}/{len(predictions)}"
    logger.info(pprint(results, indent=2))

@app.command()
def test_tfidf_logistic_regression(
    train_csv: str = "data/train.train.csv",
    test_csv: str = "data/train.test.csv",
    train_disk: str = "rules_based_train_dataset",
    test_disk: str = "rules_based_test_dataset",
    from_disk: bool = False,
):
    if from_disk:
        train_dataset = load_from_disk(train_disk)
        test_dataset = load_from_disk(test_disk)
    else:
        train_dataset = load(train_csv)
        train_dataset = train_dataset.map(Patterns.all)
        train_dataset.save_to_disk(train_disk)
        test_dataset = load(test_csv)
        test_dataset = test_dataset.map(Patterns.all)
        test_dataset.save_to_disk(test_disk)
    metric = load_metric("metrics/singleclass.py")
    results = {}   
    X = np.array(train_dataset["comment_text"]).astype('U')
    vect = TfidfVectorizer(max_features=5000,stop_words=GERMAN_STOP_WORDS)
    X_dtm = vect.fit_transform(X)
    y = train_dataset["Sub1_Toxic"]
    logreg = LogisticRegression(C=24.0)
    logger.debug(X_dtm.shape)
    logreg.fit(X_dtm, y)
    predictions = logreg.predict(X_dtm)
    results[f"tfidf_logistic_regression_training"] = metric.compute(predictions=predictions, references=y)
    results[f"tfidf_logistic_regression_training"]["number_of_occurence"] = f"{sum(predictions)}/{len(predictions)}"
    X = np.array(test_dataset["comment_text"]).astype('U')
    X_dtm = vect.transform(X)
    predictions = logreg.predict(X_dtm)
    references = test_dataset["Sub1_Toxic"]
    results[f"tfidf_logistic_regression"] = metric.compute(predictions=predictions, references=references)
    results[f"tfidf_logistic_regression"]["number_of_occurence"] = f"{sum(predictions)}/{len(predictions)}"
    logger.info(pprint(results, indent=2))


@app.command()
def test_all_and_tfidf_logistic_regression(
    train_csv: str = "data/train.train.csv",
    test_csv: str = "data/train.test.csv",
    train_disk: str = "rules_based_train_dataset",
    test_disk: str = "rules_based_test_dataset",
    from_disk: bool = False,
):
    if from_disk:
        train_dataset = load_from_disk(train_disk)
        test_dataset = load_from_disk(test_disk)
    else:
        train_dataset = load(train_csv)
        train_dataset = train_dataset.map(Patterns.all)
        train_dataset.save_to_disk(train_disk)
        test_dataset = load(test_csv)
        test_dataset = test_dataset.map(Patterns.all)
        test_dataset.save_to_disk(test_disk)
    metric = load_metric("metrics/singleclass.py")
    results = {}
    X_all = np.array(train_dataset["all"])
    X_comment_text = np.array(train_dataset["comment_text"]).astype('U')
    vect = TfidfVectorizer(max_features=5000,stop_words=GERMAN_STOP_WORDS)
    X_tfidf_logistic_regression = vect.fit_transform(X_comment_text).toarray()
    X = np.concatenate((X_all, X_tfidf_logistic_regression), axis=1)
    y = train_dataset["Sub1_Toxic"]
    clf = LogisticRegression(C=24.0)
    clf.fit(X, y)
    X_all = np.array(test_dataset["all"])
    X_comment_text = np.array(test_dataset["comment_text"]).astype('U')
    X_tfidf_logistic_regression = vect.transform(X_comment_text).toarray()
    X = np.concatenate((X_all, X_tfidf_logistic_regression), axis=1)
    references = test_dataset["Sub1_Toxic"]
    predictions = clf.predict(X)
    results[f"all"] = metric.compute(predictions=predictions, references=references)
    results[f"all"]["number_of_occurence"] = f"{sum(predictions)}/{len(predictions)}"
    logger.info(pprint(results, indent=2))

if __name__ == "__main__":
    app()
