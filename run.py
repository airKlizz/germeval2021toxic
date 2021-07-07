import torch
import typer
from torch import nn
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          MT5ForConditionalGeneration)

app = typer.Typer()


@app.command()
def t5(comment: str, model_checkpoint: str, cuda: bool = True):
    device = "cuda" if torch.cuda.is_available() and cuda else "cpu"
    tok = AutoTokenizer.from_pretrained(model_checkpoint)
    model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint).to(device)
    model.eval()

    inputs = tok("speech review: " + comment, return_tensors="pt")
    print(inputs)
    outputs = model(**inputs)
    selected_logits = outputs.logits.squeeze(1)[:, [59006, 112560]]
    score = nn.functional.softmax(selected_logits, dim=-1)
    print(score)
    return score


@app.command()
def auto(comment: str, model_checkpoint: str, cuda: bool = True):
    device = "cuda" if torch.cuda.is_available() and cuda else "cpu"
    tok = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(device)
    model.eval()

    inputs = tok(comment, return_tensors="pt")
    outputs = model(**inputs)
    score = nn.functional.softmax(outputs.logits, dim=-1)
    print(score)
    return score


if __name__ == "__main__":
    app()
