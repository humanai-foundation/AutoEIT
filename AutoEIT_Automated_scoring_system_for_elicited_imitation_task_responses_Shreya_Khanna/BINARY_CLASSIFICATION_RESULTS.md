# Overall Model Rankings

| Rank | Model | F1 Score | Features | Key Insight |
|------|-------|----------|----------|-------------|
| 🥇 1 | SVM + TF-IDF (Optimized) | 84.23% | 500 | Best statistical approach |
| 🥈 2 | SVM + TF-IDF (Default) | 84.19% | 500 | Barely beats hypertuned model |
| 🥉 3 | Coverage Fraction Only | ~83% | 1 | Best value: 500× fewer features |
| 4 | SVM + Sentence-BERT | 83.16% | 384 | Semantic < Lexical |
| 5 | Rule-Based (2 Features) | ~81–82% | 2 | Rubric-grounded baseline |
| 6 | Hybrid | ~82% | 502 | Additional features provide little benefit |
| 7 | Logistic Regression | 80.25% | 500 | Linear < Non-linear |

---

# "main.ipynb" — Statistical & Neural Approaches

## Model 1: Logistic Regression + TF-IDF

- **Features:** 500 TF-IDF terms from transcriptions
- **Algorithm:** Logistic Regression (`lbfgs` solver, `max_iter=1000`)
- **Goal:** Baseline linear classifier

### Test Results
- Accuracy: ~76%
- F1 Score: **80.25%**

### Key Finding
Simple and interpretable, but struggles to capture non-linear feature interactions.

---

## Model 2: SVM + TF-IDF (Default)

- **Features:** 500 TF-IDF terms
- **Algorithm:** SVM with RBF kernel (`C=1.0`, `gamma='scale'`)
- **Goal:** Non-linear classifier with default hyperparameters

### Test Results
- Accuracy: ~82%
- F1 Score: **84.19%**

### Key Finding
The RBF kernel captures non-linear patterns and substantially outperforms logistic regression.

---

## Model 3: SVM + TF-IDF (Optimized)

- **Features:** 500 TF-IDF terms
- **Algorithm:** SVM with RBF kernel tuned via `GridSearchCV`

### Hyperparameter Search

- `C ∈ [0.1, 1, 10, 100]`
- `γ ∈ ['scale', 'auto', 0.001, 0.01, 0.1, 1]`

### Best Parameters

- `C = 10`
- `γ = 'scale'`

### Cross-Validation Performance

- Best 5-fold CV F1: **0.8329**

### Test Results

- Accuracy: ~82%
- F1 Score: **84.23%**

### Key Finding

Hyperparameter tuning provides only a marginal improvement (+0.04 F1), suggesting the default configuration is already near-optimal.

---

## Model 4: SVM + Sentence-BERT Embeddings

- **Features:** 384-dimensional semantic embeddings generated using `all-MiniLM-L6-v2`
- **Algorithm:** SVM with RBF kernel (`C=1.0`, `gamma='scale'`)
- **Goal:** Evaluate whether semantic representations outperform lexical features

### Test Results

- Accuracy: ~80%
- F1 Score: **83.16%**

### Key Finding

Semantic embeddings underperform TF-IDF features, suggesting lexical overlap is more predictive than semantic similarity for this task.

---

# Idea Unit Coverage Notebook — Rule-Based & Hybrid Approaches

## Model 5: Rule-Based Features + SVM

### Features

- `coverage_fraction`: percentage of rubric idea units recovered
- `coverage_count`: absolute number of idea units recovered

Idea units are matched using lemmatized lexical overlap.

- **Algorithm:** SVM with RBF kernel (`C=1.0`, `gamma='scale'`)
- **Goal:** Evaluate linguistically motivated rubric-grounded features

### Test Results

- Accuracy: ~81–82%

### Key Finding

Rubric-derived features perform competitively with high-dimensional statistical approaches while remaining highly interpretable.

---


---

## Model 6: Hybrid (TF-IDF + Idea Units) + SVM

### Features

- 500 TF-IDF features
- 2 idea-unit coverage features

**Total:** 502 features

- **Algorithm:** GridSearchCV-tuned SVM with RBF kernel

### Hyperparameter Search

- `C ∈ [1, 10, 100]`
- `γ ∈ ['scale', 'auto', 0.01, 0.1]`

### Test Results

- Accuracy: ~82%
- F1 Score: ~82%

### Key Finding

Combining statistical and rule-based features provides little additional benefit, indicating that TF-IDF already captures much of the predictive signal.

---

## Model 7: Coverage Fraction Only (Single-Feature Model)

### Features

- `coverage_fraction` only

A single lemmatized idea-unit coverage metric.

- **Algorithm:** SVM with RBF kernel (`C=10`, `gamma='scale'`)

### Test Results

- Accuracy: ~81–82%
- F1 Score: ~83%

### Key Findings

- Uses **1 feature** instead of 500 TF-IDF features.
- Uses **1 feature** instead of 502 hybrid features.
- Achieves performance competitive with all other models.

### Interpretability Advantage

The model is extremely easy to explain to non-technical stakeholders:

> "The score depends primarily on the percentage of rubric content preserved by the learner."

---

