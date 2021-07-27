import pandas as pd

DATA_FILES = [
    "data/HASOC/english_dataset.tsv",
    "data/HASOC/german_dataset.tsv",
    "data/HASOC/hindi_dataset.tsv",
]


df = pd.concat([pd.read_csv(DATA_FILE, sep="\t") for DATA_FILE in DATA_FILES])
print(df)

TEXT_COLUMN = "text"
LABEL_COLUMN = "task_1"

texts = df[TEXT_COLUMN].values.tolist()
labels = [0 if l.replace(" ", "").lower() == "not" else 1 for l in df[LABEL_COLUMN].values.tolist()]

data = list(zip(texts, labels))

df = pd.DataFrame(data=data, columns=["comment_text", "hf"])
print(df)
df.to_csv("data/HASOC/train.csv")