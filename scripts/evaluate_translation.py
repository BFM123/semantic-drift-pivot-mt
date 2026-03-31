"""
Evaluate pivot-based translation quality using pretrained MarianMT models.

Pipeline:
Chichewa (ny) → English (en) → Hindi (hi)

Metrics:
- BLEU (surface-level baseline)
- BERTScore (semantic similarity)

This script is fully reproducible and does NOT require any trained models.
"""

import os
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from sacrebleu import corpus_bleu
from bert_score import score as bert_score


# ==============================
# 1. Setup paths (reproducible)
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "chichewa_parallel_corpus_clean_filtered.csv"
)


# ==============================
# 2. Load dataset
# ==============================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Adjust column names if needed
chichewa_col = "chichewa"
english_col = "english"

chichewa_sentences = df[chichewa_col].astype(str).tolist()
reference_english = df[english_col].astype(str).tolist()

# Optional: limit for faster testing
MAX_SAMPLES = 100
chichewa_sentences = chichewa_sentences[:MAX_SAMPLES]
reference_english = reference_english[:MAX_SAMPLES]


# ==============================
# 3. Load pretrained models
# ==============================
print("🔄 Loading pretrained MarianMT models...")

model_ny_en_name = "Helsinki-NLP/opus-mt-ny-en"
model_en_hi_name = "Helsinki-NLP/opus-mt-en-hi"

tokenizer_ny_en = MarianTokenizer.from_pretrained(model_ny_en_name)
model_ny_en = MarianMTModel.from_pretrained(model_ny_en_name)

tokenizer_en_hi = MarianTokenizer.from_pretrained(model_en_hi_name)
model_en_hi = MarianMTModel.from_pretrained(model_en_hi_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ny_en.to(device)
model_en_hi.to(device)


# ==============================
# 4. Translation function
# ==============================
def translate_batch(texts, tokenizer, model, batch_size=16):
    translations = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            generated = model.generate(**inputs)

        decoded = [
            tokenizer.decode(t, skip_special_tokens=True)
            for t in generated
        ]

        translations.extend(decoded)

    return translations


# ==============================
# 5. Pivot Translation
# ==============================
print("🌐 Translating Chichewa → English...")
predicted_english = translate_batch(
    chichewa_sentences,
    tokenizer_ny_en,
    model_ny_en
)

print("🌐 Translating English → Hindi...")
predicted_hindi = translate_batch(
    predicted_english,
    tokenizer_en_hi,
    model_en_hi
)


# ==============================
# 6. Evaluation
# ==============================

# ---- BLEU (English level for reference)
bleu = corpus_bleu(predicted_english, [reference_english])
print(f"\n📊 BLEU Score (ny→en): {bleu.score:.4f}")

# ---- BERTScore (semantic similarity)
print("🧠 Computing BERTScore...")
P, R, F1 = bert_score(
    predicted_english,
    reference_english,
    lang="en",
    model_type="xlm-roberta-base"
)

print(f"BERTScore F1 (mean): {F1.mean().item():.4f}")


# ==============================
# 7. Sample outputs
# ==============================
print("\n🔍 Sample Translations:\n")

for i in range(min(5, len(chichewa_sentences))):
    print(f"Chichewa: {chichewa_sentences[i]}")
    print(f"Reference EN: {reference_english[i]}")
    print(f"Predicted EN: {predicted_english[i]}")
    print(f"Predicted HI: {predicted_hindi[i]}")
    print("-" * 50)


# ==============================
# 8. Save results (optional)
# ==============================
results_path = os.path.join(BASE_DIR, "results", "metrics")
os.makedirs(results_path, exist_ok=True)

output_file = os.path.join(results_path, "translation_results.csv")

output_df = pd.DataFrame({
    "chichewa": chichewa_sentences,
    "reference_en": reference_english,
    "predicted_en": predicted_english,
    "predicted_hi": predicted_hindi
})

output_df.to_csv(output_file, index=False)

print(f"\n✅ Results saved to: {output_file}")