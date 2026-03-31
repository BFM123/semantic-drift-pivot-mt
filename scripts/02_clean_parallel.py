import pandas as pd
import re
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_path = os.path.join(BASE_DIR, "data", "raw", "chichewa_parallel_corpus.csv")

output_path = os.path.join(BASE_DIR, "data", "processed", "chichewa_parallel_corpus_clean.csv")


if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ File not found: {input_path}")

print(f"🧩 Loading dataset from: {input_path}")
df = pd.read_csv(input_path, dtype=str, encoding="utf-8")

# Remove NaNs and clean whitespace
df.dropna(inplace=True)
df["chichewa"] = df["chichewa"].astype(str).str.strip()
df["english"] = df["english"].astype(str).str.strip()

def clean_text(text):
    # Remove leading line numbers, commas, quotes
    text = re.sub(r'^[\"\']?\d+\s*,\s*', '', text)
    text = re.sub(r'^[\"\']+|[\"\']+$', '', text)  # remove surrounding quotes
    text = text.replace(',,', ',')  # collapse multiple commas
    text = text.strip()
    return text

df["chichewa"] = df["chichewa"].apply(clean_text)
df["english"] = df["english"].apply(clean_text)

# Remove any leftover numeric-only or malformed entries
df = df[~df["chichewa"].str.match(r"^\d+$")]
df = df[~df["english"].str.match(r"^\d+$")]

# Drop empty rows
df = df[(df["chichewa"].str.len() > 1) & (df["english"].str.len() > 1)]

# Normalize text (lowercase + space cleanup)
df["chichewa"] = df["chichewa"].str.lower().str.replace(r"\s+", " ", regex=True)
df["english"] = df["english"].str.lower().str.replace(r"\s+", " ", regex=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Save clean version
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"✅ Cleaned parallel corpus saved to: {output_path}")
print(f"📊 Final sentence pairs: {len(df)}")
print(df.sample(5))