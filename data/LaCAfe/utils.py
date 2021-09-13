import pandas as pd

DATA_FILE = "data/LaCAfe/df_dataset.xlsx"

df = pd.read_excel(DATA_FILE)
TEXT_COLUMN = "txt"
LABEL_COLUMN = "has_anger"

texts = df[TEXT_COLUMN].values.tolist()
labels = [1 if l == "S" else 0 for l in df[LABEL_COLUMN].values.tolist()]
data = list(zip(texts, labels))

df = pd.DataFrame(data=data, columns=["comment_text", "hf"])
print(df)
df.to_csv("data/LaCAfe/train.csv")
