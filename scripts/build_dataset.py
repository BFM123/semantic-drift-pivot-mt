import pandas as pd

# Input file (downloaded from OPUS/Tatoeba)
INPUT_FILE = "raw_data/tatoeba_ny_en.txt"
OUTPUT_FILE = "data/dataset_412.csv"

MAX_SENTENCES = 412
MAX_LENGTH = 15

def clean_sentence(s):
    return s.strip()

def is_valid(s):
    return len(s.split()) <= MAX_LENGTH and len(s) > 3

def main():
    pairs = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ny, en = line.strip().split("\t")
            except:
                continue

            ny = clean_sentence(ny)
            en = clean_sentence(en)

            if is_valid(ny) and is_valid(en):
                pairs.append((ny, en))

            if len(pairs) >= MAX_SENTENCES:
                break

    df = pd.DataFrame(pairs, columns=["source_chichewa", "reference_english"])
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} sentence pairs to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()