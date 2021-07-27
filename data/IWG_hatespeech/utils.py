import pandas as pd

df = pd.read_csv("data/IWG_hatespeech/german hatespeech refugees.csv")

labels = [0 if l1  == "NO" and l2 == "NO" else 1 for l1, l2 in zip(df["HatespeechOrNot (Expert 1)"].values.tolist(), df["HatespeechOrNot (Expert 2)"].values.tolist())]
texts = df["Tweet"].values.tolist()

data = list(zip(texts, labels))

df = pd.DataFrame(data=data, columns=["comment_text", "hf"])
df.to_csv("data/IWG_hatespeech/train.csv")