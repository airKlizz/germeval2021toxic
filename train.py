import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import typer
from datasets import load_metric
from loguru import logger
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          MT5ForConditionalGeneration, T5Tokenizer, Trainer,
                          TrainingArguments)

from dataset import load

app = typer.Typer()


# class MT5Trainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
#         logits = logits.squeeze(1)
#         logits = logits[:, [375, 36339]]  # no=375 yes=36339
#         loss_fct = torch.nn.CrossEntropyLoss()
#         loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         return (loss, outputs) if return_outputs else loss


class TrainerWithClassWeightsToxic(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Sub1_Toxic
        # loss_fct = torch.nn.CrossEntropyLoss(
        #     weight=torch.Tensor([0.34586929716399506, 0.6541307028360049]).to(logits.device)
        # )
        # Sub1_Toxic with labelled data
        # loss_fct = torch.nn.CrossEntropyLoss(
        #     weight=torch.Tensor([0.4087087734425523, 0.5912912265574477]).to(logits.device)
        # )
        # Sub2_Engaging
        # loss_fct = torch.nn.CrossEntropyLoss(
        #     weight=torch.Tensor([0.2658959537572254, 0.7341040462427746]).to(logits.device)
        # )
        # Sub3_FactClaiming
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.Tensor([0.3421965317919075, 0.6578034682080924]).to(logits.device)
        )

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class MT5TrainerWithClassWeightsToxic(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = torch.zeros(logits.size(-1)).to(logits.device)
        weight[375] = 0.34586929716399506
        weight[36339] = 0.6541307028360049
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# @app.command()
# def multiclass(
#     train_csv: List[str] = ["data/train.train.csv"],
#     test_csv: str = "data/train.test.csv",
#     model_checkpoint: str = "deepset/gbert-base",
#     output_dir: str = "models/multiclass/",
#     batch_size: int = 16,
#     learning_rate: float = 2e-5,
#     nb_epoch: int = 5,
#     max_length: int = None,
# ):
#     logger.info(f"Start multiclass training.")
#     output_dir += (
#         model_checkpoint.replace("/", "_")
#         + "_bs="
#         + str(batch_size)
#         + "_lr="
#         + str(learning_rate)
#         + "_epoch="
#         + str(nb_epoch)
#     )
#     logger.info(f"Load the model: {model_checkpoint}.")
#     model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

#     args = TrainingArguments(
#         output_dir=output_dir,
#         evaluation_strategy="epoch",
#         learning_rate=learning_rate,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         num_train_epochs=nb_epoch,
#         weight_decay=0.01,
#         load_best_model_at_end=True,
#         metric_for_best_model="f1",
#         logging_dir="./logs",
#         logging_steps=10,
#     )

#     metric = load_metric("metrics/flat_multiclass.py")

#     def compute_metrics(eval_pred):
#         predictions, labels = eval_pred
#         predictions = np.sign(predictions)
#         return metric.compute(predictions=predictions.astype("int32"), references=labels.astype("int32"))

#     logger.info("Load and preprocess the dataset.")
#     train_dataset = load(train_csv, model_checkpoint, preprocess=True, max_length=max_length)
#     test_dataset = load(test_csv, model_checkpoint, preprocess=True, max_length=max_length)
#     tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

#     trainer = Trainer(
#         model,
#         args,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#     )

#     logger.info("Start the training.")
#     trainer.train()

#     logger.info("Start the evaluation.")
#     trainer.evaluate()


@app.command()
def singleclass(
    train_csv: List[str] = ["data/train.train.csv"],
    test_csv: str = "data/train.test.csv",
    train_labels: List[str] = ["Sub1_Toxic"],
    test_labels: List[str] = ["Sub1_Toxic"],
    class_weights: bool = False,
    model_checkpoint: str = "deepset/gbert-base",
    model_type: str = "auto",
    output_dir: str = "models/singleclass/",
    strategy: str = "epoch",
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    eval_accumulation_steps: int = 100,
    learning_rate: float = 5e-5,
    nb_epoch: int = 3,
    max_length: int = 256,
    eval_steps: int = 250,
    save_steps: int = 500,
):
    logger.info(f"Start singleclass training.")
    output_dir += (
        model_checkpoint.replace("/", "_")
        + "_class_weights="
        + str(class_weights)
        + "_labels="
        + "_".join(train_labels)
        + "_languages="
        + "+".join(train_csv).replace("data/", "").replace("/", "_")
        + "_bs="
        + str(batch_size)
        + "_lr="
        + str(learning_rate)
        + "_epoch="
        + str(nb_epoch)
    )
    output_dir = output_dir[:256]
    logger.info(f"Load the model: {model_checkpoint}.")

    if model_type == "auto":
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    elif model_type == "t5":
        model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint)
    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")

    args = TrainingArguments(
        output_dir=output_dir,
        save_strategy=strategy,
        save_steps=save_steps,
        evaluation_strategy=strategy,
        eval_steps=eval_steps,
        eval_accumulation_steps=eval_accumulation_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=nb_epoch,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=10,
    )

    metric = load_metric("metrics/singleclass.py")

    if model_type == "auto":

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)

    elif model_type == "t5":

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            # print("LOGITS")
            # print(type(logits))
            # print(len(logits))
            # print(type(logits[0]))
            # print(logits[0].shape)
            # print(np.argmax(logits[0], axis=2))
            labels = np.where(labels == 59006, 0, labels)
            labels = np.where(labels == 112560, 1, labels)
            logits = torch.tensor(logits[0]).squeeze(1)
            selected_logits = logits[:, [59006, 112560]]
            probs = F.softmax(selected_logits, dim=1)
            predictions = np.argmax(probs.tolist(), axis=1)
            return metric.compute(predictions=predictions, references=labels)

    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")

    logger.info("Load and preprocess the dataset.")
    logger.debug(f"train_csv: {train_csv}")
    logger.debug(f"test_csv: {test_csv}")
    train_dataset = load(
        train_csv, model_checkpoint, model_type, preprocess=True, labels=train_labels, max_length=max_length
    )
    test_dataset = load(
        test_csv, model_checkpoint, model_type, preprocess=True, labels=test_labels, max_length=max_length
    )
    logger.info(f"Dataset sample: {train_dataset[0]}")
    if model_type == "auto":
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    elif model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, use_fast=True)
    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")

    if model_type == "auto":
        if class_weights == True:
            if len(train_labels) == 1 and train_labels[0] == "Sub1_Toxic":
                logger.info("Using TrainerWithClassWeightsToxic")
                trainer = TrainerWithClassWeightsToxic(
                    model,
                    args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                )
            else:
                raise NotImplementedError()
        else:
            logger.info("Using Trainer")
            trainer = Trainer(
                model,
                args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
    elif model_type == "t5":
        if class_weights == True:
            if len(train_labels) == 1 and train_labels[0] == "Sub1_Toxic":
                logger.info("Using MT5TrainerWithClassWeightsToxic")
                trainer = MT5TrainerWithClassWeightsToxic(
                    model,
                    args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                )
            else:
                raise NotImplementedError()
        else:
            logger.info("Using MT5Trainer")
            trainer = Trainer(
                model,
                args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")

    logger.info("Start the training.")
    trainer.train()

    logger.info("Start the evaluation.")
    metrics = trainer.evaluate()
    logger.info(metrics)
    trainer.save_model()


@app.command()
def hyperparameter_search_singleclass(
    train_csv: List[str] = ["data/train.train.csv"],
    test_csv: str = "data/train.test.csv",
    train_labels: List[str] = ["Sub1_Toxic"],
    test_labels: List[str] = ["Sub1_Toxic"],
    class_weights: bool = False,
    model_checkpoint: str = "deepset/gbert-base",
    model_type: str = "auto",
    output_dir: str = "models/hyperparameter_search_singleclass/",
    strategy: str = "epoch",
    eval_accumulation_steps: int = 100,
    max_length: int = 256,
    eval_steps: int = 250,
    save_steps: int = 500,
    n_trials: int = 10,
):
    logger.info(f"Start singleclass training.")
    output_dir += (
        model_checkpoint.replace("/", "_")
        + "_class_weights="
        + str(class_weights)
        + "_label="
        + str(train_labels)
        + "_languages="
        + "+".join(train_csv).replace("data/", "").replace("/", "_")
    )
    output_dir = output_dir[:256]
    logger.info(f"Load the model: {model_checkpoint}.")

    if model_type == "auto":

        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    elif model_type == "t5":

        def model_init():
            return MT5ForConditionalGeneration.from_pretrained(model_checkpoint)

    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")

    args = TrainingArguments(
        output_dir=output_dir,
        save_strategy=strategy,
        save_steps=save_steps,
        evaluation_strategy=strategy,
        eval_steps=eval_steps,
        eval_accumulation_steps=eval_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=10,
    )

    metric = load_metric("metrics/singleclass.py")

    if model_type == "auto":

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)

    elif model_type == "t5":

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            labels = np.where(labels == 59006, 0, labels)
            labels = np.where(labels == 112560, 1, labels)
            logits = torch.tensor(logits[0]).squeeze(1)
            selected_logits = logits[:, [59006, 112560]]
            probs = F.softmax(selected_logits, dim=1)
            predictions = np.argmax(probs.tolist(), axis=1)
            return metric.compute(predictions=predictions, references=labels)

    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")

    logger.info("Load and preprocess the dataset.")
    logger.debug(f"train_csv: {train_csv}")
    logger.debug(f"test_csv: {test_csv}")
    train_dataset = load(
        train_csv, model_checkpoint, model_type, preprocess=True, labels=train_labels, max_length=max_length
    )
    test_dataset = load(
        test_csv, model_checkpoint, model_type, preprocess=True, labels=test_labels, max_length=max_length
    )
    logger.info(f"Dataset sample: {train_dataset[0]}")
    if model_type == "auto":
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    elif model_type == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, use_fast=True)
    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")

    if model_type == "auto":
        if class_weights == True:
            if len(train_labels) == 1:
                logger.info("Using TrainerWithClassWeightsToxic")
                trainer = TrainerWithClassWeightsToxic(
                    model_init=model_init,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                )
            else:
                raise NotImplementedError()
        else:
            logger.info("Using Trainer")
            trainer = Trainer(
                model_init=model_init,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
    elif model_type == "t5":
        if class_weights == True:
            if len(train_labels) == 1:
                logger.info("Using MT5TrainerWithClassWeightsToxic")
                trainer = MT5TrainerWithClassWeightsToxic(
                    model_init=model_init,
                    args=args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                )
            else:
                raise NotImplementedError()
        else:
            logger.info("Using MT5Trainer")
            trainer = Trainer(
                model_init=model_init,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )
    else:
        raise NotImplementedError("Model type available: 'auto' or 't5'")

    def my_objective(metrics):
        try:
            return metrics["eval_f1"]
        except:
            logger.debug(metrics.keys())
            return metrics["f1"]

    logger.info("Start the hyperparameter search.")
    best_run = trainer.hyperparameter_search(n_trials=n_trials, direction="maximize", compute_objective=my_objective)

    logger.info(f"Best run: {best_run}")

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    logger.info("Start final training.")
    trainer.train()

    logger.info("Start the evaluation.")
    metrics = trainer.evaluate()
    logger.info(metrics)
    trainer.save_model()


if __name__ == "__main__":
    app()
