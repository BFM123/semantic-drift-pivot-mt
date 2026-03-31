# Semantic Drift in Pivot-Based Machine Translation

This repository contains the implementation for the study:

"Quantifying and Mitigating Semantic Drift in Low-Resource Pivot-Based Machine Translation"

## Overview

This project investigates semantic drift in a Chichewa → English → Hindi pivot-based translation pipeline and evaluates lightweight mitigation strategies.

## Pipeline

1. Chichewa → English (MarianMT)
2. Semantic augmentation (back-translation + synonym substitution)
3. English → Hindi (MarianMT)
4. Evaluation using:
   - BERTScore
   - Cosine similarity
   - Human evaluation

## Models Used

- Helsinki-NLP/opus-mt-ny-en
- Helsinki-NLP/opus-mt-en-hi
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- bert-base-multilingual-cased

## Dataset

The dataset consists of 412 sentence pairs derived from publicly available bilingual corpora. It is used for evaluation only.

## Reproducibility

All experiments use pretrained models. No training or fine-tuning is performed.

## Disclaimer

This repository contains a simplified and clean version of the experimental pipeline aligned with the published study. Earlier exploratory experiments (including model training) are not included.

## License

This project is licensed under the MIT License.