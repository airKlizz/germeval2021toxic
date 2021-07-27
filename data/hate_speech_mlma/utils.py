import pandas as pd

DATA_FILES = [
    "data/hate_speech_mlma/fr_dataset.csv",
    "data/hate_speech_mlma/en_dataset.csv",
    "data/hate_speech_mlma/ar_dataset.csv",
]


df = pd.concat([pd.read_csv(DATA_FILE) for DATA_FILE in DATA_FILES])
print(df)

TEXT_COLUMN = "tweet"
LABEL_COLUMN = "sentiment"

texts = df[TEXT_COLUMN].values.tolist()
labels = [0 if l.replace(" ", "").lower() == "normal" else 1 for l in df[LABEL_COLUMN].values.tolist()]

data = list(zip(texts, labels))

df = pd.DataFrame(data=data, columns=["comment_text", "hf"])
df.to_csv("data/hate_speech_mlma/train.csv")