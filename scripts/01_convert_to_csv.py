import os
import pandas as pd
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "data", "raw", "SpokenChichewaCorpus-master", "SpokenChichewaCorpus-master", "ChichewaParallelCorpus"))
output_file = os.path.normpath(os.path.join(BASE_DIR, "data", "raw", "chichewa_parallel_corpus.csv"))

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"❌ Folder not found: {DATA_DIR}")

# Collect all Chichewa-English file pairs
ny_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_ny.txt")))
en_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_en.txt")))

if not ny_files or not en_files:
    raise FileNotFoundError("❌ Could not find *_ny.txt or *_en.txt files in folder.")

paired_files = [(ny, ny.replace("_ny.txt", "_en.txt")) for ny in ny_files if os.path.exists(ny.replace("_ny.txt", "_en.txt"))]

all_pairs = []

for ny_path, en_path in paired_files:
    with open(ny_path, "r", encoding="utf-8") as f_ny, open(en_path, "r", encoding="utf-8") as f_en:
        ny_lines = [l.strip() for l in f_ny.readlines()]
        en_lines = [l.strip() for l in f_en.readlines()]

        # Align shorter length
        min_len = min(len(ny_lines), len(en_lines))
        ny_lines = ny_lines[:min_len]
        en_lines = en_lines[:min_len]

        pairs = list(zip(ny_lines, en_lines))
        all_pairs.extend(pairs)

print(f"🧩 Found {len(paired_files)} paired files")
print(f"🧾 Combined {len(all_pairs)} sentence pairs before cleaning")

# Create DataFrame
df = pd.DataFrame(all_pairs, columns=["chichewa", "english"])
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Clean and normalize
df["chichewa"] = df["chichewa"].str.strip().str.lower()
df["english"] = df["english"].str.strip().str.lower()

# Save CSV
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"✅ Saved clean parallel corpus to: {output_file}")
print(f"📊 Final rows after cleaning: {len(df)}")