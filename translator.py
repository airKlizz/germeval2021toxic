import pandas as pd
import typer
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

app = typer.Typer()


@app.command()
def translate(csv: str, output_csv: str, lang_iso: str, batch_size: int = 16):
    tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-de-{lang_iso}", use_fast=True)
    translator = pipeline(f"translation_de_to_{lang_iso}", model=f"Helsinki-NLP/opus-mt-de-{lang_iso}")
    df = pd.read_csv(csv)
    translated_df = pd.read_csv(csv)
    too_long_samples = []
    for i in tqdm(range(0, len(df), batch_size), desc="Translation in progress..."):
        j = min(i + batch_size, len(df))
        texts = [df.iloc[k]["comment_text"] for k in range(i, j)]
        for k, text in enumerate(texts):
            if len(tokenizer(text)["input_ids"]) > tokenizer.max_len_single_sentence:
                too_long_samples.append(i + k)
                texts[k] = "xxx"
        responses = translator(texts)
        for k, r in enumerate(responses):
            translated_df.at[i + k, "comment_text"] = r["translation_text"]
    print(f"Number of samples too long: {len(too_long_samples)}")
    translated_df = translated_df.drop(translated_df.index[too_long_samples])
    translated_df.to_csv(output_csv)


if __name__ == "__main__":
    app()
