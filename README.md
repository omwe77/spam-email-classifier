# Spam Email Classifier

An introductory machine learning project exploring text feature extraction and probabilistic classification for spam email detection using Python and `scikit-learn`.

---

## Overview

- **Author:** Om Dangol
- **Type:** Foundational Machine Learning Project
- **Algorithm:** Multinomial Naive Bayes (`MultinomialNB`)
- **Feature Extraction:** Bag-of-Words via `CountVectorizer`
- **Language:** Python 3.10+
- **Primary Library:** `scikit-learn`

---

## Technical Concept

The classifier implements the classic Bag-of-Words text classification pipeline:

```
Raw Text Inputs
      │
      ▼
Tokenization & Counting (CountVectorizer)
      │
      ▼
Document-Term Matrix (Sparse Feature Vectors)
      │
      ▼
Multinomial Naive Bayes Fitting (P(Class | Features))
      │
      ▼
Inference / Prediction (Spam vs. Ham)
```

1. **Vectorization:** Converts raw email text into token-frequency feature representations.
2. **Probabilistic Modeling:** Applies Bayes' theorem with the assumption of conditional feature independence across vocabulary tokens:
   $$P(\text{Spam} \mid \mathbf{x}) \propto P(\text{Spam}) \prod_{i=1}^n P(x_i \mid \text{Spam})$$
3. **Inference:** Evaluates arbitrary user-provided input strings and returns classification labels (`spam` or `ham`).

---

## Project Structure

```
spam-email-classifier/
├── email_classifier.py   # Primary vectorization and inference script
├── requirements.txt      # Dependency specification
├── .gitignore            # Ignores bytecode and virtual environments
└── README.md             # Project documentation
```

---

## Implementation Status

### Implemented (Verified in Active Codebase)
- **Text Feature Extraction:** Converts sample email strings into bag-of-words token-count matrices via `CountVectorizer`.
- **Probabilistic Modeling:** Fits a `MultinomialNB` model against sample label matrices.
- **Interactive Console Inference:** Accepts arbitrary user text via terminal prompt and outputs predicted `spam` or `ham` classification.

### In Progress
- *None (Exploratory Proof-of-Concept Complete).*

### Planned (Future Enhancements)
- **Benchmark Corpus Training:** Training and evaluating against standardized NLP corpora (e.g. Enron Spam or SMS Spam Collection).
- **Evaluation Telemetry:** Generating confusion matrices, ROC-AUC curves, and F1-score benchmarks.
- **TF-IDF Weighting:** Replacing basic word counts with Term Frequency-Inverse Document Frequency scaling.

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/omwe77/spam-email-classifier.git
cd spam-email-classifier

# Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Classifier

```bash
python email_classifier.py
```

When prompted, enter a message to receive real-time classification:
```text
Enter an email message: Claim your free prize
Prediction: spam
```

---

## Scope & Limitations

- **Dataset Scale:** Built as a minimal proof-of-concept with inline synthetic samples to test feature extraction mechanics rather than a fully generalized production detector.
- **Model Evaluation:** Does not include train/test splits, cross-validation, or precision-recall curves on large public corpora (e.g. Enron or SMS Spam Collection).
- **Future Directions:** Incorporating TF-IDF weighting, subword tokenization, and evaluating on formal NLP benchmark datasets.

---

## License

Open-source under the MIT License for educational reference.
