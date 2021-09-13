import pandas as pd

df = pd.read_csv("data/conan/CONAN.csv")

hate_texts = list(set(df["hateSpeech"].values.tolist()))
non_hate_texts = list(set(df["counterSpeech"].values.tolist()))

data = [[t, 1] for t in hate_texts]
data += [[t, 0] for t in non_hate_texts]

df = pd.DataFrame(data=data, columns=["comment_text", "hf"])
df.to_csv("data/conan/train.csv")
