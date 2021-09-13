import pandas as pd

DATA_FILES = [
    "data/germeval18/germeval2018.training.txt",
]


df = pd.concat([pd.read_csv(DATA_FILE, header=None, sep="\t") for DATA_FILE in DATA_FILES])
print(df)

TEXT_COLUMN = 0
LABEL_COLUMN = 1

texts = df[TEXT_COLUMN].values.tolist()
labels = [0 if l.replace(" ", "").lower() == "other" else 1 for l in df[LABEL_COLUMN].values.tolist()]

data = list(zip(texts, labels))

df = pd.DataFrame(data=data, columns=["comment_text", "hf"])
print(df)
df.to_csv("data/germeval18/train.csv")
