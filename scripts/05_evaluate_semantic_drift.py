"""
Semantic Drift Evaluation with Augmentation

Pipeline:
1. Chichewa → English → Hindi (baseline)
2. Compute semantic drift
3. Apply augmentation (English pivot)
4. Re-translate English → Hindi
5. Compute drift again
6. Compare results

Metrics:
- Cosine similarity (drift = 1 - similarity)
- BERTScore (semantic validation)

This script is fully reproducible and uses pretrained models only.
"""
import random
import numpy as np
import os
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score as bert_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import re
import unicodedata

def clean_hindi_text(text):
    # Normalize unicode (VERY IMPORTANT)
    text = unicodedata.normalize("NFC", text)

    # Remove weird spacing between Devanagari characters
    text = re.sub(r"(?<=[\u0900-\u097F])\s+(?=[\u0900-\u097F])", "", text)

    # Fix spacing around punctuation
    text = re.sub(r"\s+([।,.!?])", r"\1", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Fix common broken patterns
    text = text.replace(" ्", "्")   # halant
    text = text.replace(" ि", "ि")
    text = text.replace(" ा", "ा")
    text = text.replace(" ी", "ी")
    text = text.replace(" ु", "ु")
    text = text.replace(" े", "े")
    text = text.replace(" ै", "ै")

    return text.strip()

# 1. Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "chichewa_parallel_corpus_clean_filtered.csv"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results", "metrics")
os.makedirs(RESULTS_DIR, exist_ok=True)


# 2. Load dataset
# Limit for testing (remove later)
MAX_SAMPLES = 100

df = pd.read_csv(DATA_PATH)

chichewa_sentences = df["chichewa"].astype(str).tolist()
reference_english = df["english"].astype(str).tolist()

# Apply same slicing consistently
chichewa_sentences = chichewa_sentences[:MAX_SAMPLES]
reference_english = reference_english[:MAX_SAMPLES]

# 3. Load models
print("🔄 Loading models...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Translation models
model_ny_en_name = "Helsinki-NLP/opus-mt-ny-en"
model_en_hi_name = "Helsinki-NLP/opus-mt-en-hi"

tokenizer_ny_en = MarianTokenizer.from_pretrained(model_ny_en_name)
model_ny_en = MarianMTModel.from_pretrained(model_ny_en_name).to(device)

tokenizer_en_hi = MarianTokenizer.from_pretrained(model_en_hi_name)
model_en_hi = MarianMTModel.from_pretrained(model_en_hi_name).to(device)

# Embedding model (IMPORTANT — explicitly defined)
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# 4. Translation function
def translate_batch(texts, tokenizer, model, batch_size=16):
    outputs = []

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

        outputs.extend(decoded)

    return outputs


# 5. Augmentation (EXPLICIT)
def augment_english(sentences, tokenizer_en_hi, model_en_hi,
                    tokenizer_ny_en, model_ny_en):
    """
    Augmentation = Back-translation + Phrase stabilization
    """

    print("🔁 Running back-translation (EN → NY → EN)...")

    # Step 1: English → Chichewa (reverse direction)
    en_ny_model_name = "Helsinki-NLP/opus-mt-en-ny"
    tokenizer_en_ny = MarianTokenizer.from_pretrained(en_ny_model_name)
    model_en_ny = MarianMTModel.from_pretrained(en_ny_model_name).to(device)

    # Translate EN → NY
    ny_back = translate_batch(sentences, tokenizer_en_ny, model_en_ny)

    # Step 2: Chichewa → English
    en_reconstructed = translate_batch(ny_back, tokenizer_ny_en, model_ny_en)

    print("🔧 Applying phrase stabilization...")

    # Step 3: Phrase stabilization
    replacements = {
        "make a business": "conduct business",
        "make business": "conduct business",
        "customers wants": "customers want",
        "a customers": "a customer",
        "top - paying facilities": "key opportunities",
        "full - time business": "business operations",
        "we are throughout the world": "we operate globally",
        "pursuing what a customers wants": "meeting customer needs",
    }

    stabilized = []

    for s in en_reconstructed:
        s_aug = s
        for k, v in replacements.items():
            s_aug = s_aug.replace(k, v)
        stabilized.append(s_aug)

    return stabilized

# 6. Drift computation
def compute_drift(source, target):
    emb_src = embedding_model.encode(source)
    emb_tgt = embedding_model.encode(target)

    sims = [
        cosine_similarity([s], [t])[0][0]
        for s, t in zip(emb_src, emb_tgt)
    ]

    drift = [1 - sim for sim in sims]
    return sims, drift


# 7. Baseline pipeline
print("🌐 Running baseline translation...")

en_baseline = translate_batch(chichewa_sentences, tokenizer_ny_en, model_ny_en)
hi_baseline = translate_batch(en_baseline, tokenizer_en_hi, model_en_hi)
hi_baseline = [clean_hindi_text(s) for s in hi_baseline]

# 8. Drift (baseline)
print("📉 Computing baseline drift...")

sim_base, drift_base = compute_drift(chichewa_sentences, hi_baseline)


# 9. Augmentation pipeline
print("🔧 Applying augmentation...")

en_augmented = augment_english(en_baseline, tokenizer_en_hi, model_en_hi, tokenizer_ny_en, model_ny_en)
hi_augmented = translate_batch(en_augmented, tokenizer_en_hi, model_en_hi)
hi_augmented = [clean_hindi_text(s) for s in hi_augmented]

# 10. Drift (augmented)
print("📉 Computing augmented drift...")

sim_aug, drift_aug = compute_drift(chichewa_sentences, hi_augmented)


# 11. BERTScore comparison
print("🧠 Computing BERTScore comparison...")

_, _, F1_base = bert_score(en_baseline,reference_english,lang="en",model_type="xlm-roberta-base")

_, _, F1_aug = bert_score(en_augmented,reference_english,lang="en",model_type="xlm-roberta-base")

bert_base = F1_base.mean().item()
bert_aug = F1_aug.mean().item()

# 12. Results summary
mean_drift_base = sum(drift_base) / len(drift_base)
mean_drift_aug = sum(drift_aug) / len(drift_aug)

improvement = (mean_drift_base - mean_drift_aug) / mean_drift_base * 100

print("\n📊 RESULTS")
print(f"Baseline Drift: {mean_drift_base:.4f}")
print(f"Augmented Drift: {mean_drift_aug:.4f}")
print(f"Drift Reduction: {improvement:.2f}%")

print(f"\nBERTScore (baseline): {bert_base:.4f}")
print(f"BERTScore (augmented): {bert_aug:.4f}")


# 13. Save results
output = pd.DataFrame({
    "chichewa": chichewa_sentences,
    "en_baseline": en_baseline,
    "hi_baseline": hi_baseline,
    "en_augmented": en_augmented,
    "hi_augmented": hi_augmented,
    "drift_baseline": drift_base,
    "drift_augmented": drift_aug
})

output_file = os.path.join(RESULTS_DIR, "semantic_drift_results.csv")
output.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n✅ Saved results to: {output_file}")

# 13B. Save summary metrics
metrics_summary = pd.DataFrame([{
    "baseline_drift": mean_drift_base,
    "augmented_drift": mean_drift_aug,
    "drift_reduction_percent": improvement,
    "bert_baseline": bert_base,
    "bert_augmented": bert_aug
}])

# 13C. Generate human evaluation sheet

eval_df = pd.DataFrame({
    "id": range(1, len(chichewa_sentences)+1),
    "chichewa": chichewa_sentences,
    "reference_en": reference_english,
    "baseline_hi": hi_baseline,
    "augmented_hi": hi_augmented,
    "choice": ["" for _ in chichewa_sentences],
    "notes": ["" for _ in chichewa_sentences]
})

eval_df = eval_df.sample(n=50, random_state=42)

eval_file = os.path.join(RESULTS_DIR, "human_evaluation.csv")
eval_df.to_csv(eval_file, index=False, encoding="utf-8-sig")

print(f"🧾 Human evaluation file saved to: {eval_file}")

metrics_file = os.path.join(RESULTS_DIR, "metrics_summary.csv")
metrics_summary.to_csv(metrics_file, index=False)

print(f"📊 Metrics summary saved to: {metrics_file}")


# 14. Sample output
print("\n🔍 SAMPLE COMPARISON:\n")

for i in range(5):
    print(f"Chichewa: {chichewa_sentences[i]}")
    print(f"Baseline HI: {hi_baseline[i]}")
    print(f"Augmented HI: {hi_augmented[i]}")
    print(f"Drift ↓: {drift_base[i]:.3f} → {drift_aug[i]:.3f}")
    print("-" * 50)