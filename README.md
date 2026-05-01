# Sentiment Analysis — Recurrent Architectures vs. Embedding Paradigms

A comparative study investigating how embedding paradigms (Static vs. Contextual) and recurrent architectures (Bi-LSTM vs. Bi-GRU) jointly influence sentiment classification performance — with a focus on data efficiency, computational cost, and representational quality.

---

## Motivation

A central question in applied NLP is whether the representational richness of contextual embeddings (BERT) justifies their computational overhead compared to static embeddings (GloVe) — and whether this trade-off shifts depending on how much training data is available.

This project systematically answers that question through controlled experiments across two data regimes.

---

## Key Results

| Model | Val. Acc (2k samples) | Val. Acc (15k samples) | Training Time (15k) |
|---|---|---|---|
| GloVe + Bi-LSTM | 60.20% | 82.50% | ~106s |
| GloVe + Bi-GRU | 67.60% | 86.25% | ~81s |
| BERT + Bi-LSTM | 80.80% | 87.00% | ~1579s |
| **BERT + Bi-GRU** | **83.40%** | **91.45%** | ~1705s |

**Main finding:** BERT dominates in low-resource settings, but GloVe + Bi-GRU closes most of the gap at scale — at 20x lower computational cost.

---

## Project Structure

```
├── sentiment_analysis.ipynb     # Full experiment notebook
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── .gitignore                   # Excludes large embedding files
```

---

## Methodology

### Task
Binary sentiment classification on the **IMDb dataset** (positive / negative reviews).

### Architectures
- **Bi-LSTM** — Bidirectional Long Short-Term Memory
- **Bi-GRU** — Bidirectional Gated Recurrent Unit

A shared projection layer maps all embedding dimensions to 256d, ensuring identical downstream parameter counts for a fair comparison.

### Embeddings
- **Static:** GloVe (6B tokens, 300d)
- **Contextual:** DistilBERT (`distilbert-base-uncased`), fine-tuned end-to-end

### Experimental Design
- **Phase 1 (Low-Resource):** 2,000 training samples, 3 epochs
- **Phase 2 (Scale-Up):** 15,000 training samples, 5 epochs

### Evaluation
- Validation Accuracy & Macro-F1
- Training time (convergence efficiency)
- t-SNE visualization of learned representations

---

## Findings

**1. Contextual embeddings excel in low-resource settings.**
BERT-based models immediately reach >80% accuracy with only 2k samples, while GloVe models struggle around 60–67%. BERT's dynamic context-aware representations prevent feature collapse where static vectors fail.

**2. Static embeddings are competitive at scale.**
With 15k samples, GloVe + Bi-GRU reaches 86.25% — just 5 points behind the best BERT model — while training nearly 20x faster (81s vs. 1705s).

**3. Bi-GRU consistently outperforms Bi-LSTM.**
Across all embedding types and data regimes, GRU's simpler gating mechanism generalizes better on this task, suggesting LSTM's added complexity introduces overfitting risk without proportional gain.

---

## Latent Space Visualization

t-SNE plots reveal that BERT representations form distinct class clusters even in Phase 1, while GloVe representations remain scattered until sufficient data is provided.

| Phase 1 (Low-Resource) | Phase 2 (Scale-Up) |
|---|---|
| BERT clusters clearly, GloVe overlaps | Both improve; BERT maintains sharper separation |

---

## Setup & Reproduction

### 1. Clone the repository

```bash
git clone https://github.com/DenizhanOngun/Sentiment-Analysis-RNN-vs-Embeddings.git
cd Sentiment-Analysis-RNN-vs-Embeddings
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `sentiment_analysis.ipynb` and run all cells sequentially.

> **Note:** GloVe embeddings (~800MB) are downloaded automatically in Cell 2. No manual setup required.

---

## Requirements

```
torch
datasets
transformers
scikit-learn
matplotlib
numpy
```

---

## Live Notebook

[Open in Google Colab](https://colab.research.google.com/drive/1N8FKebe1owCpvStBKH9XiwyJCFIk1HOg#scrollTo=3VYFL_X4vWbP)
