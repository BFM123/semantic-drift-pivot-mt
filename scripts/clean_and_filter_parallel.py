"""
clean_and_filter_parallel.py
----------------------------
Enhanced cleaning for the parallel corpus + basic quality filtering.

Reads processed/chichewa_parallel_corpus_clean.csv (or LT_INPUT_DATA override) and outputs
processed/chichewa_parallel_corpus_clean_filtered.csv (or LT_OUTPUT_DATA override).

Filters:
- Remove rows where source == target (English duplicated or same text).
- Remove rows where either side is very short (< 3 chars) or excessively long (> 300 chars).
- Remove rows dominated by ASCII if Chichewa side looks English (heuristic: >85% letters in A-Za-z and >=3 common English stopwords).
- Deduplicate after normalization.
- Optional length ratio filter (drop if len(src)/len(tgt) > 3 or < 1/3).

Set LT_KEEP_RATIO=0 to disable ratio filtering.
Usage:
  .\.venv\Scripts\python.exe scripts\clean_and_filter_parallel.py
"""
import os
import re
import pandas as pd
from collections import Counter

INPUT_DEFAULT = r"C:\Users\Brian\AI_Lingua_Triad\data\processed\chichewa_parallel_corpus_clean.csv"
OUTPUT_DEFAULT = r"C:\Users\Brian\AI_Lingua_Triad\data\processed\chichewa_parallel_corpus_clean_filtered.csv"

input_path = os.environ.get("LT_INPUT_DATA", INPUT_DEFAULT)
output_path = os.environ.get("LT_OUTPUT_DATA", OUTPUT_DEFAULT)
ratio_enabled = os.environ.get("LT_KEEP_RATIO", "1") != "0"

if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ Input dataset not found: {input_path}")

print(f"🔍 Loading dataset: {input_path}")
df = pd.read_csv(input_path).dropna()

# Basic strip
for col in ["chichewa", "english"]:
    df[col] = df[col].astype(str).str.strip()

# Lowercase normalization
df["chichewa"] = df["chichewa"].str.lower().str.replace(r"\s+", " ", regex=True)
df["english"] = df["english"].str.lower().str.replace(r"\s+", " ", regex=True)

initial = len(df)

# Remove identical lines (often English duplicated)
df = df[df["chichewa"] != df["english"]]

# Length based filtering
df = df[(df["chichewa"].str.len() >= 3) & (df["english"].str.len() >= 3)]
df = df[(df["chichewa"].str.len() <= 300) & (df["english"].str.len() <= 300)]

# Heuristic: detect likely-English in chichewa column
english_stop = {"the","and","for","that","you","with","this","are","was","have","not","can","will"}

def looks_english(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    alpha_ratio = sum(c in 'abcdefghijklmnopqrstuvwxyz' for c in letters) / len(letters)
    stops = sum(1 for w in text.split() if w in english_stop)
    return alpha_ratio > 0.85 and stops >= 3

mask_eng_like = df["chichewa"].apply(looks_english)
removed_eng_like = int(mask_eng_like.sum())
df = df[~mask_eng_like]

# Length ratio filter
if ratio_enabled:
    ratios = df["english"].str.len() / df["chichewa"].str.len().clip(lower=1)
    df = df[(ratios <= 3.0) & (ratios >= 1/3)]

# Deduplicate
before_dup = len(df)
df = df.drop_duplicates(subset=["chichewa","english"])
removed_dup = before_dup - len(df)

final = len(df)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False, encoding="utf-8")

print("✅ Cleaning & filtering complete")
print(f"📊 Initial rows: {initial}")
print(f"🚫 Removed identical src==tgt rows: {initial - len(df) + removed_dup - removed_eng_like}")
print(f"🚫 Removed English-like Chichewa rows: {removed_eng_like}")
print(f"🚫 Removed duplicates: {removed_dup}")
print(f"✅ Final rows: {final}")
print("🔎 Sample:")
print(df.sample(min(5, final)))
