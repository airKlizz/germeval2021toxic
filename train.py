import numpy as np
import torch
from datasets import load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from parameters import (BATCH_SIZE, LEARNING_RATE, MODEL_CHECKPOINT, NB_EPOCH,
                        NUM_LABELS, OUTPUT_DIR)
from dataset import load_train, load_test
from loguru import logger


logger.info(f"Load the model: {MODEL_CHECKPOINT}.")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=NUM_LABELS)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NB_EPOCH,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='./logs',
    logging_steps=10,
)

metric = load_metric("metrics/singleclass.py")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.sign(predictions)
    return metric.compute(predictions=predictions.astype("int32"), references=labels.astype("int32"))

logger.info("Load and preprocess the dataset.")
train_dataset = load_train(preprocess=True)
test_dataset = load_test(preprocess=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

logger.info("Start the training.")
trainer.train()

logger.info("Start the evaluation.")
trainer.evaluate()