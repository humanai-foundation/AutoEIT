# Spanish EIT Automated Scoring

> **GSoC 2026 project** — Automated scoring for the Spanish Elicited Imitation Task using ordinal machine learning + LLM arbitration.
>
> **Author:** Shreya Khanna | **Organization:** HumanAI Foundation | **Mentor:** Mandy Faretta-Stutenberg

---

## What is this?

The **Spanish Elicited Imitation Task (EIT)** is a spoken language proficiency assessment where learners hear a sentence in Spanish and repeat it from memory. Trained human raters score responses on a 0-4 scale. This project automates that scoring using a linguistically-motivated ordinal machine learning pipeline.

**Key result:** QWK ≈ 0.90 on held-out test data — approaching the inter-rater reliability of human raters.

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/humanai-foundation/AutoEIT
cd AutoEIT

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

**For scoring new responses** (lightweight, ~50 MB):
```bash
pip install -r requirements_production.txt
```

**For research & model development** (full dependencies with transformers/LLM APIs):
```bash
pip install -r requirements_research.txt
```

### 2. Score new responses

```bash
# Score a single sheet in an EIT workbook:
python score_responses.py \
  --input data/my_responses.xlsx \
  --output data/scored_output.xlsx

# Score with LLM arbitration for uncertain cases (requires API key):
python score_responses.py \
  --input data/my_responses.xlsx \
  --output data/scored_output.xlsx \
  --llm gemini  # or --llm openai
```

**Required input columns:** `Stimulus`, `Final Transcription`

The output Excel preserves all original columns and adds:
| Column | Description |
|---|---|
| `AutoScore` | Predicted score (0–4) from ordinal SVM |
| `Confidence` | Model confidence (higher = more certain) |
| `Final_AutoScore` | Final score (LLM-adjusted if arbitrated) |
| `Arbitrated` | Whether LLM was invoked for this row |

### 3. Re-train the model

If you have your own training data:

```bash
python score_responses.py --input your_data.xlsx --output scored.xlsx --retrain
```

Or to fully reproduce the original training:

```bash
# Place combined.xlsx and idea_units_Spanish_AutoEIT.xlsx in data/
python score_responses.py --input data/combined.xlsx --output scored.xlsx --retrain
```

---

## How It Works

### Feature Engineering

The model uses **11 linguistically-motivated features** computed from the stimulus and learner response — no pretrained neural networks required:

| Feature | Description |
|---|---|
| `coverage_fraction` | % of rubric idea units present in response |
| `coverage_count` | Absolute number of idea units covered |
| `wer` | Word error rate (custom alignment) |
| `hit_rate` | Fraction of stimulus words correctly reproduced |
| `substitutions / insertions / deletions` | Word-level edit operations |
| `syllable_ratio` | Response syllables / stimulus syllables |
| `response_length_ratio` | Response tokens / stimulus tokens |
| `negation_preserved` | Whether Spanish negation words are preserved |
| `plural_preserved` | Whether plural markers are preserved |

### Model: Ordinal Threshold Ensemble

Instead of treating scoring as a 5-class classification problem, the model uses an **ordinal decomposition**: it trains one binary classifier per threshold (P(score ≥ 1), P(score ≥ 2), P(score ≥ 3), P(score ≥ 4)), then sums the votes. This respects the **ordered structure** of EIT scores.

```
Input response → Feature engineering → 4 binary classifiers → Sum votes → Final score
```

### LLM Arbitration (Optional)

For responses where the model is uncertain (confidence < 0.70), an LLM (GPT-4o-mini or Gemini 1.5 Flash) is invoked to adjudicate. At the default threshold of 0.70, arbitration triggers selectively on SOME input items (the hardest boundary decisions), keeping API costs low while improving inter-rater agreement and reducing error margins.

---

## Results

### Model Comparison & LLM Arbitration Impact (note: this is for a single experiment with a specific confidence threshold)

| Model Architecture | LLM Arbitration | QWK ↑ | Exact Agreement | Within-1 Point Agreement | MAE ↓ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ordinal Decomposition (Threshold Ensemble)** | ❌ No | **0.898** *(0.8854 test)* | **69.6%** | **96.8%** | **0.314** |
| **Ordinal Decomposition (Threshold Ensemble)** | ✅ Yes (Gemini/GPT-4o) | **0.899** *(0.8885 test)* | **70.0%** | **96.8%** | **0.312** |
| Multi-Class Logistic Regression | ❌ No | 0.893 *(0.8712 test)* | 68.9% | 95.8% | 0.324 |
| Multi-Class Logistic Regression | ✅ Yes (Gemini/GPT-4o) | 0.895 *(0.8745 test)* | 69.2% | 95.8% | 0.321 |

### Full Experiment Baseline Leaderboard

| Model | Accuracy | Weighted F1 | QWK ↑ | MAE ↓ |
|---|---|---|---|---|
| **Engineered + Ordinal LinearSVM** | **0.696** | **0.696** | **0.898** | **0.314** |
| Threshold ensemble (ordinal decomp.) | 0.696 | 0.695 | 0.898 | 0.315 |
| Hybrid: BETO cosine-sim + Engineered | 0.692 | 0.691 | 0.897 | 0.318 |
| Engineered + BERTScore (F1) | 0.686 | 0.684 | 0.895 | 0.324 |
| BETO mean pooling + LogisticAT | 0.602 | 0.602 | 0.855 | 0.419 |
| BETO CLS + LogisticRegression | 0.600 | 0.598 | 0.838 | 0.444 |
| TF-IDF + LinearSVC (baseline) | 0.513 | 0.511 | 0.749 | 0.594 |

**Key insights:**
1. **Ordinal Decomposition vs. Multi-Class Logistic Regression:** Ordinal decomposition outperforms standard multi-class logistic regression because it respects the natural ordering ($0 < 1 < 2 < 3 < 4$) of human rubric ratings rather than treating classes as nominal categories.
2. **Global Pooled QWK vs. Per-Sheet Macro Average:** Global pooled QWK across all test items reaches **0.898 (0.8885 arbitrated)**. When calculated macro-averaged per 30-sentence participant sheet, mean sheet QWK is **0.750**, reflecting smaller per-participant score variance.
3. **Linguistic Features beat Embeddings:** Pure engineered linguistic features beat transformer embeddings (BETO) — and are 10× cheaper to run.



## Reproducing Results

1. Install dependencies: `pip install -r requirements.txt`
2. Place `combined.xlsx` and `idea_units_Spanish_AutoEIT.xlsx` in `data/`
3. Run the notebook: open `notebooks/ordinal.ipynb` and run all cells
4. To evaluate sentence-level agreement and LLM arbitration:
   ```bash
   python evaluation_scripts/test_sentence_agreement.py --llm gemini
   ```
5. All random seeds are fixed at `random_state=42`

For the full experiment suite (BETO, BERTScore, hybrid variants), see `notebooks/multinomial_log_reg.ipynb`.

> **Note:** BETO experiments require downloading `dccuchile/bert-base-spanish-wwm-cased` from HuggingFace (~400MB). BERTScore uses `bert-base-multilingual-cased`.

---

## Limitations & Next Steps

- **Dataset size:** n=1964 with some small score bins (especially score=4, n=295). Larger datasets would improve robustness.
- **Idea-unit dependency:** The `coverage_fraction` feature requires a manually curated rubric. Expanding to new stimuli requires rubric annotation.
- **LLM arbitration:** Systematically evaluated — LLM arbitration selectively resolves low-confidence boundary decisions (8.1% trigger rate), boosting QWK to 0.899.
- **Paper:** These results are being prepared for submission to a language assessment journal.

---

## Citation

If you use this work, please cite:

```
Khanna, S. and Faretta-Stutenberg, M. (2026). Automated Ordinal Scoring for the Spanish Elicited Imitation Task.
Google Summer of Code 2026, HumanAI Foundation.
https://github.com/humanai-foundation/SpanishEIT
```

---

## Contact

- **Author:** Shreya Khanna
- **Email:** shreyyabiz@gmail.com
- **Organization:** HumanAI Foundation — https://humanai.foundation
- **Blog:** https://medium.com/@khanna.shreya19/automating-spanish-proficiency-scoring-my-gsoc-2026-journey-5115c5798985
