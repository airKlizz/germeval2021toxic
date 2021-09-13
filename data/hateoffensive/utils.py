import pandas as pd

df = pd.read_csv("data/hateoffensive/labeled_data.csv")

labels = [0 if l == 2 else 1 for l in df["class"].values.tolist()]
texts = df["tweet"].values.tolist()

data = list(zip(texts, labels))

df = pd.DataFrame(data=data, columns=["comment_text", "hf"])
df.to_csv("data/hateoffensive/train.csv")
