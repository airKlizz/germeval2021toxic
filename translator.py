import pandas as pd
import typer
from tqdm import tqdm
from transformers import pipeline

app = typer.Typer()


@app.command()
def translate(csv: str, output_csv: str, lang_iso: str, batch_size: int = 16):
    translator = pipeline(f"translation_de_to_{lang_iso}", model=f"Helsinki-NLP/opus-mt-de-{lang_iso}")
    df = pd.read_csv(csv)
    translated_df = pd.read_csv(csv)
    for i in tqdm(range(0, len(df), batch_size), desc="Translation in progress..."):
        j = min(i + batch_size, len(df))
        texts = [df.iloc[k]["comment_text"] for k in range(i, j)]
        responses = translator(texts)
        for k, r in enumerate(responses):
            translated_df.at[i + k, "comment_text"] = r["translation_text"]
    translated_df.to_csv(output_csv)


if __name__ == "__main__":
    app()
