import pandas as pd

df = pd.read_csv("data/hate-speech-dataset/annotations_metadata.csv")

def read_file(filename):
    path = "data/hate-speech-dataset/all_files/" + filename + ".txt"
    with open(path) as f:
        return f.read()

filenames = df["file_id"].values.tolist()
texts = [read_file(f) for f in filenames]

labels = [0 if l == "noHate" else 1 for l in df["label"].values.tolist()]

data = list(zip(texts, labels))

df = pd.DataFrame(data=data, columns=["comment_text", "hf"])
df.to_csv("data/hate-speech-dataset/train.csv")

