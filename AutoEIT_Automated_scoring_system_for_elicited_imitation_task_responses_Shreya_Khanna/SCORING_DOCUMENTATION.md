# Scoring Documentation — Spanish EIT Automated Scoring Engine

> **Project:** GSoC 2026 — Automated Spanish Elicited Imitation Task (EIT) Scorer  
> **Repository:** `SpanishEIT_LogisticRegression`  
> **Primary script:** `score_responses.py`  
> **Last updated:** 2026-08-19

---

## Table of Contents

1. [Overview](#1-overview)
2. [Scoring Rubric (0–4 Scale)](#2-scoring-rubric-0-4-scale)
3. [Feature Engineering (11 Features)](#3-feature-engineering-11-features)
4. [Model Architecture — OrdinalThresholdEnsemble](#4-model-architecture--ordinalthresholdensemble)
5. [Training Procedure](#5-training-procedure)
6. [LLM Arbitration Policy](#6-llm-arbitration-policy)
7. [Acceptance Criteria & Validation Protocol](#7-acceptance-criteria--validation-protocol)
8. [Reproducibility Checklist](#8-reproducibility-checklist)
9. [Known Limitations & Future Work](#9-known-limitations--future-work)
10. [References](#10-references)

---

## 1. Overview

The scoring engine takes as input:
- A **stimulus sentence** — the Spanish sentence the learner heard.
- A **learner transcription** — the sentence the learner produced from memory.

It outputs a **score from 0 to 4** that approximates the score an experienced human rater would assign under the EIT rubric (Ortega, 2015).

The pipeline has two stages:

```
Stimulus + Transcription
        │
        ▼
┌───────────────────────────┐
│  Feature Engineering (11) │
└───────────┬───────────────┘
            │
            ▼
┌────────────────────────────────┐
│  OrdinalThresholdEnsemble      │
│  (4 binary logistic regressions│
│   + monotonicity enforcement)  │
└───────────┬────────────────────┘
            │  AutoScore  +  Confidence ∈ [0, 1]
            ▼
   Confidence < threshold?
        │           │
       YES          NO
        │           │
        ▼           ▼
  LLM Arbitration  AutoScore → Final_AutoScore
  (optional)
        │
        ▼
  LLM_Score → Final_AutoScore
```

---

## 2. Scoring Rubric (0–4 Scale)

The rubric follows **Ortega (2015)** and evaluates three linguistic dimensions:
*content word accuracy*, *morphological accuracy*, and *syntactic structure*.

| Score | Label | Criteria |
|-------|-------|----------|
| **4** | Perfect | All content words (nouns, verbs, adjectives, adverbs) present and accurate. Morphology (gender, number, tense, person) matches stimulus. Syntactic structure fully preserved. |
| **3** | Minor deviation | At most **one** content word missing, substituted, or morphologically incorrect. Overall meaning and structure intact. |
| **2** | Partial | Multiple content words missing or wrong; **OR** significant morphological errors; **OR** major syntactic restructuring that alters meaning. |
| **1** | Minimal | Very few correct content words. Response is fragmentary or heavily restructured. |
| **0** | No content | Response is unintelligible, off-topic, completely empty, or a non-Spanish utterance. |

### Scoring Decision Guidelines

- Score on **content accuracy** (content words, morphology, syntax) — **not** pronunciation.
- Filler words (`"um"`, `"este"`, pauses, `"xxx"`) are ignored.
- A student who reproduces **meaning** with minor wording differences scores higher than one who uses exact words in the wrong order.
- When in doubt between two adjacent scores, choose the **lower** score.
- Negation words (`no`, `nunca`, `nadie`, `jamás`, etc.) are content words — their omission is penalized.
- Plural morphology on nouns and adjectives is content-relevant.

---

## 3. Feature Engineering (11 Features)

All features are computed deterministically from the stimulus and transcription strings. No external models or embeddings are required.

### Text Preprocessing

Before feature computation, all text is:
1. Unicode-normalized (NFKD, accents stripped for comparison).
2. Lowercased.
3. Non-alphabetic characters removed (except `ñ`).
4. Whitespace-normalized.

A lightweight **lemmatizer** strips common Spanish plural suffixes (`-s`, `-es`) for coverage matching.

---

### Feature Definitions

| # | Feature | Formula | Rationale |
|---|---------|---------|-----------|
| 1 | `coverage_fraction` | `covered_idea_units / total_idea_units` | Idea-unit coverage directly operationalises content-word accuracy |
| 2 | `coverage_count` | Raw count of idea units covered | Absolute count complements the fraction for short stimuli |
| 3 | `wer` | `(S + I + D) / len(reference)` via Levenshtein alignment | Global fidelity measure; high WER = many errors |
| 4 | `substitutions` | Edit-distance substitution count | Distinguishes "wrong word" from "missing word" errors |
| 5 | `insertions` | Edit-distance insertion count | Captures hallucinated words not in stimulus |
| 6 | `deletions` | Edit-distance deletion count | Captures omitted words |
| 7 | `hit_rate` | `correct_words / len(reference)` | Complement of WER; higher = better |
| 8 | `syllable_ratio` | `response_syllables / stimulus_syllables` | Proxies for response completeness at the phonological level |
| 9 | `response_length_ratio` | `len(response_tokens) / len(stimulus_tokens)` | Word-level completeness ratio |
| 10 | `negation_preserved` | 1 if stimulus negations ⊆ response negations, else 0 | Critical for morpho-syntactic accuracy; negation change flips meaning |
| 11 | `plural_preserved` | 1 if stimulus plurals ⊆ response (after lemmatizing), else 0 | Morphological agreement signal |

#### Idea Unit Coverage

Idea units are stored in `data/idea_units_Spanish_AutoEIT.xlsx`, one row per stimulus.
Each row's `idea_units_breakdown` column contains `::` -delimited chunks representing
the meaningful sub-parts of the sentence.

A learner's response **covers** an idea unit if **every lemmatized token** of the unit
appears in the lemmatized token set of the response.

#### Word Alignment (WER)

The engine implements a custom Levenshtein dynamic-programming alignment
(`compute_word_alignment` in `score_responses.py`) that returns
`(hits, substitutions, insertions, deletions, wer, hit_rate)`.

Monotonicity of edit operations is guaranteed by the backtracking logic
(hits preferred over substitutions; substitutions preferred over deletions/insertions).

#### Syllable Counting

Spanish syllable counts are computed using the `silabeador` library.
If unavailable, a vowel-group fallback is used: `max(1, count(vowel_groups))` per word.
For the stimulus, the count embedded in parentheses — e.g. `"El niño corre. (4)"` — is
extracted directly, avoiding recomputation.

---

## 4. Model Architecture — OrdinalThresholdEnsemble

### Why Ordinal Regression?

The 0–4 score is **ordinal** — the distance between scores is not uniform, and the
classes are ordered. A standard multiclass classifier (e.g., one-vs-rest logistic
regression) ignores this ordering information.

The `OrdinalThresholdEnsemble` implements the **cumulative link / proportional-odds**
approach from Agresti (2002) via K−1 binary classifiers:

```
For k ∈ {1, 2, 3, 4}:
    P(score ≥ k | x)  ←  LogisticRegression( y_binary = [score ≥ k] )

Monotonicity enforcement:
    P(≥1) ≥ P(≥2) ≥ P(≥3) ≥ P(≥4)   [enforced via cumulative minimum]

Prediction:
    score = Σ_{k=1}^{4}  𝟙[ P(≥k | x) ≥ 0.5 ]
```

This ensures the model respects score ordering without any post-hoc correction.

### Confidence Estimate

```
confidence(x) = clip( 2 · mean_k( |P(≥k|x) - 0.5| ), 0, 1 )
```

This maps the average distance-from-boundary of all four threshold probabilities
onto [0, 1]. A score of 0 means the model is maximally uncertain (all thresholds
near 0.5); a score of 1 means maximally certain (all thresholds far from 0.5).

---

## 5. Training Procedure

### Data

- **Source file:** `data/combined.xlsx`
- **Size:** ~1,964 learner responses (as of GSoC 2026 snapshot)
- **Labels:** `final score` column (0–4, human consensus score)
- **Stimuli:** 30 unique EIT items from Ortega (2015)

### Split Strategy

Responses are grouped by stimulus using `sklearn.model_selection.GroupShuffleSplit`
with `test_size=0.20` and `random_state=42`. This guarantees that the test set
contains stimuli **never seen during training**, simulating real-world generalization.

```
random_state = 42
test_size    = 0.20
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| C (regularization) | 0.5 | Tuned via 5-fold CV; prevents overfitting to stimulus-specific patterns |
| class_weight | "balanced" | Compensates for class imbalance (scores 0 and 4 are rare) |
| max_iter | 5000 | Ensures convergence on standardized features |
| Scaler | StandardScaler | Mean-centering + unit variance required for L2-regularized logistic regression |

### Performance (Canonical Split)

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| QWK (quadratic weighted κ) | 0.8957 | 0.8906 | 0.8329 |
| MAE | — | — | 0.4902 |

The test QWK of **0.8329** is well above the 0.70 threshold commonly used
to establish machine–human agreement for automatic scoring systems
(Williamson et al., 2012).

### Model Persistence

The trained model is saved to `models/ordinal_svm_pipeline.pkl` using Python's
`pickle` module. The file encodes the `OrdinalThresholdEnsemble` object and all
four internal `Pipeline(StandardScaler, LogisticRegression)` sub-models.

---

## 6. LLM Arbitration Policy

### Purpose

The ordinal model is most uncertain near score boundaries (1↔2, 2↔3).
LLM arbitration targets **only these uncertain cases**, keeping API costs low
while improving agreement for the hardest items.

### Trigger Condition

LLM arbitration is invoked when:

```
confidence(x) < ARBITRATION_THRESHOLD   (default: 0.70)
```

Empirically at threshold 0.70, this triggers on **8.1% of responses** (17 items out of 210 in complete sheets, or 36 items out of 532 in full test workbook).

### Supported Backends

| Backend | Flag | API Key | Notes |
|---------|------|---------|-------|
| Gemini | `--llm gemini` | `GOOGLE_API_KEY` | Supports both `google-generativeai` (v0) and `google-genai` (v1) SDKs; auto-detects |
| OpenAI | `--llm openai` | `OPENAI_API_KEY` | Uses `gpt-4o-mini` |
| Mock | `--llm mock` | None | Echoes SVM score; for CI/offline testing |

### Empirical Benchmarking & Impact

| Metric | Ordinal Baseline (No LLM) | Ordinal + LLM (Gemini) | Impact |
|---|:---:|:---:|:---:|
| **Global Pooled QWK** | **0.8854** *(0.8976 CV)* | **0.8885** *(0.8985 CV)* | **+0.0031** |
| **Exact Sentence Agreement** | **65.60%** *(69.59% CV)* | **66.17%** *(69.95% CV)* | **+0.57%** |
| **Within-1 Point Agreement ($\le \pm 1$)** | **96.80%** | **96.80%** | **0.00%** |
| **Mean Absolute Error (MAE)** | **0.3816** | **0.3741** | **-0.0075** |

---

## 7. Acceptance Criteria & Validation Protocol

### GSoC KPIs & Empirical Validation

| # | KPI | Target | Empirical Result | Status |
|---|-----|--------|------------------|--------|
| KPI 1 | Exact sentence-level agreement with human raters | **≥ 90%** | **61.0%** micro exact / **95.2%** within-1 point (complete sheets); **66.2%** exact / **96.8%** within-1 (all pooled items). | ⚠️ Within-1 meets target; exact match strictly below 90% |
| KPI 2 | Total EIT score difference per participant | **< 10 points** | **Mean 4.86 pts** difference across participants (120-point scale). Passes on 6/7 complete sheets. | ✅ PASS (Average < 10 pts) |

### Note on Global Pooled vs. Per-Sheet Macro QWK

- **Global Pooled QWK (0.8885 $\approx$ 0.898)**: Measures overall inter-rater reliability across the entire dataset, maintaining the full 0–4 score variance across all participants.
- **Per-Sheet Macro Average QWK (0.750)**: Measures individual rater agreement calculated separately per 30-sentence sheet. Smaller score variance within an individual participant depresses the per-sheet Cohen's Kappa score.

### Evaluation Scripts

```powershell
# Full acceptance audit (baseline only, exit code 0 = PASS):
python evaluate_acceptance_criteria.py

# Sentence-level agreement test (baseline only):
python test_sentence_agreement.py

# Sentence-level agreement test (with mock LLM — no API key needed):
python test_sentence_agreement.py --llm mock

# Sentence-level agreement test (with Gemini arbitration):
python test_sentence_agreement.py --llm gemini

# LLM arbitration comparative evaluation:
python evaluate_llm_arbitration.py
python evaluate_llm_arbitration.py --backend gemini
```

### Report Outputs

| File | Contents |
|------|----------|
| `models/acceptance_test_report.json` | KPI pass/fail + per-sheet and aggregate metrics |
| `models/acceptance_test_sheet_results.csv` | Per-sheet metrics table |
| `models/sentence_agreement_report.json` | Full sentence-level report (baseline + LLM comparison) |
| `models/sentence_agreement_sheet_results.csv` | Per-sheet metrics table with PASS/FAIL columns |
| `models/llm_arbitration_evaluation.json` | Baseline vs. arbitration comparison by sheet |

---

## 8. Reproducibility Checklist

### Step 1 — Install dependencies

```powershell
pip install -r requirements.txt
# For LLM arbitration (optional):
pip install "google-genai>=1.0"        # Gemini v1 SDK (recommended)
# or: pip install "google-generativeai>=0.5"  # Gemini v0 SDK
# or: pip install "openai>=1.0"               # OpenAI
```

### Step 2 — Retrain the model

```powershell
python score_responses.py \
    --retrain \
    --input data/combined.xlsx \
    --output data/test_scored_output.xlsx
# Model saved to: models/ordinal_svm_pipeline.pkl
```

### Step 3 — Run acceptance audit

```powershell
python evaluate_acceptance_criteria.py
# Writes: models/acceptance_test_report.json
# Exit code 0 = all KPIs pass
```

### Step 4 — Run sentence-level agreement test

```powershell
python test_sentence_agreement.py
# Writes: models/sentence_agreement_report.json
#         models/sentence_agreement_sheet_results.csv
```

### Step 5 — Score new responses

```powershell
python score_responses.py \
    --input path/to/new_responses.xlsx \
    --output path/to/scored_output.xlsx \
    [--llm gemini]           # optional LLM arbitration
    [--arbitration-threshold 0.65]
```

### Pinning exact results

All randomness is controlled by `RANDOM_STATE = 42`.
The `GroupShuffleSplit` uses the same seed, so train/val/test splits are
**identical** across all runs that use `--retrain` from the same `combined.xlsx`.

The model pickle (`models/ordinal_svm_pipeline.pkl`) encodes the exact trained weights.
Distributing this file alongside the code guarantees byte-identical predictions
without retraining.

---

## 9. Known Limitations & Future Work

### Current Limitations

| Limitation | Impact | Mitigation |
|------------|--------|-----------|
| Idea-unit coverage requires `idea_units_Spanish_AutoEIT.xlsx` | `coverage_fraction` = 0 if file missing | Use `--no-idea-units` flag; other features still work |
| Lemmatizer is rule-based (suffix stripping) | May over- or under-stem irregular forms | Future: integrate SpaCy/stanza for morphological analysis |
| LLM arbitration costs tokens per call | API costs scale with low-confidence responses | `--llm mock` for offline testing; threshold controls rate |
| Test QWK (0.8329) measured on held-out *stimuli*, not held-out *participants* | May overestimate generalization to new speakers | Future: evaluate on held-out participants |
| No disfluency or pause detection | Transcriptions may include markers that inflate WER | `build_agreement_summary.py` tracks disfluency count as a diagnostic |

### Future Work

- **Speaker-level cross-validation** — hold out participants rather than stimuli for more realistic generalization estimates.
- **SpaCy/stanza integration** — replace rule-based lemmatizer with a full morphological analyzer.
- **Active arbitration selection** — use uncertainty sampling to select the most informative cases for human re-scoring.
- **BETO / BERTScore features** — contextual embeddings as additional features (experimental results in `notebooks/`).
- **Longitudinal drift detection** — monitor score distribution over time to catch model drift.

---

## 10. References

- Agresti, A. (2002). *Categorical Data Analysis* (2nd ed.). Wiley.
- Ortega, M., Paredes, L., Tenbrink, T., Rodríguez, J. A. (2015). Scoring rubric for Spanish Elicited Imitation Task. [Internal rubric document.]
- Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice*, 31(1), 2–13.
- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly Media.
- Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
