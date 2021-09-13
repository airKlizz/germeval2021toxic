import pandas as pd

DATA_FILE = "data/idhsd_rio/IDHSD_RIO_unbalanced_713_2017.txt"

with open(DATA_FILE) as f:
    data = f.read().split("\n")[1:]

values = []
for d in data:
    elems = d.split(" ~ ")
    assert len(elems) == 2
    new_elems = [elems[1]]
    if elems[0] == "Non_HS":
        new_elems.append(0)
    elif elems[0] == "HS":
        new_elems.append(1)
    else:
        raise ValueError
    values.append(new_elems)

df = pd.DataFrame(data=values, columns=["comment_text", "hf"])
df.to_csv("data/idhsd_rio/train.csv")
