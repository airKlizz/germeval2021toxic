import pandas as pd

DATA_FILE = "data/okkyibrohim/re_dataset.csv"

df = pd.read_csv(DATA_FILE)

TEXT_COLUMN = "Tweet"
LABEL_COLUMN = "HS"

texts = df[TEXT_COLUMN].values.tolist()
labels = df[LABEL_COLUMN].values.tolist()

data = list(zip(texts, labels))

df = pd.DataFrame(data=data, columns=["comment_text", "hf"])
df.to_csv("data/okkyibrohim/train.csv")
