import pandas as pd

DATA_FILE = "data/L-HSAB/L-HSAB"

df = pd.read_csv(DATA_FILE, delimiter="\t")
print(df)

TEXT_COLUMN = "Tweet"
LABEL_COLUMN = "Class"

texts = df[TEXT_COLUMN].values.tolist()
labels = [0 if l.replace(" ", "").lower() == "normal" else 1 for l in df[LABEL_COLUMN].values.tolist()]

data = list(zip(texts, labels))

df = pd.DataFrame(data=data, columns=["comment_text", "hf"])
df.to_csv("data/L-HSAB/train.csv")
