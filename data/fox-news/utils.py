import json

import pandas as pd

DATA_FILE = "data/fox-news/fox-news-comments.json"

with open(DATA_FILE) as f:
    data = f.read().split("\n")

data = [json.loads(l) for l in data[:-1]]
labels = [d["label"] for d in data]
texts = [d["text"] for d in data]

data = list(zip(texts, labels))

df = pd.DataFrame(data=data, columns=["comment_text", "hf"])
df.to_csv("data/fox-news/train.csv")