# Uncovering Deceptive Reviews Using ML and NLP with Power BI

**MBA Final Project — Amity University Online**

| | |
|---|---|
| **Student** | Keshav Kant Singh |
| **Enrollment** | A9920124010871 |
| **Program** | MBA (Data Science) |
| **Guide** | Kunwar Saurabh Bisen |
| **Institution** | Amity University Online, Noida |

## Project Overview

This project develops an end-to-end pipeline for detecting deceptive (fake) reviews using machine learning and natural language processing, with Power BI dashboards for strategic insights.

**Key Result:** A Linear SVM classifier trained on TF-IDF bigram features achieved **86.6% accuracy** on a held-out test set (precision 85.9%, recall 87.5%, F1 86.7%).

## Dataset

- **Corpus:** Deceptive Opinion Spam Corpus v1.4 (Ott et al., 2011, 2013)
- **Size:** 1,600 labeled hotel reviews (800 deceptive + 800 genuine)
- **Domain:** 20 most-reviewed Chicago hotels
- **Source:** [myleott.com/op-spam.html](https://myleott.com/op-spam.html)

## Project Structure

```
deceptive-reviews-project/
├── main.py                    # Pipeline orchestrator (run this)
├── requirements.txt           # Python dependencies
├── README.md
├── data/
│   ├── raw/                   # Raw corpus CSV
│   └── processed/             # Cleaned/split data
├── src/
│   ├── config.py              # Paths, constants, parameters
│   ├── data_loader.py         # Load and profile corpus
│   ├── preprocessing.py       # NLP text cleaning pipeline
│   ├── feature_engineering.py # TF-IDF + sentiment features
│   ├── model_training.py      # Train, validate, evaluate
│   ├── feature_analysis.py    # SVM feature weight extraction
│   └── powerbi_export.py      # Score & export for Power BI
├── outputs/                   # Model results JSON, feature weights
└── powerbi/                   # Scored reviews CSV for Power BI
```

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/deceptive-reviews-project.git
cd deceptive-reviews-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the corpus CSV in data/raw/
#    (download from https://myleott.com/op-spam.html)

# 4. Run the full pipeline
python main.py
```

## Pipeline Steps

1. **Load** — Read the 1,600-review corpus
2. **Profile** — Compute dataset statistics (mean word count, class balance, etc.)
3. **Preprocess** — Lowercase, remove non-alpha, tokenize, remove stop words, lemmatize
4. **Feature Engineering** — TF-IDF vectorization (5,000 bigram features) + TextBlob sentiment
5. **Train** — Logistic Regression, Linear SVM, Random Forest on 960-review training set
6. **Validate** — Compare on 320-review validation set; select best by F1
7. **Cross-Validate** — 5-fold stratified CV for stability check
8. **Test** — Evaluate best model on 320-review held-out test set
9. **Feature Analysis** — Extract top deceptive/genuine indicators from SVM weights
10. **Export** — Score all 1,600 reviews and export for Power BI dashboards

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 86.6% |
| Precision | 85.9% |
| Recall | 87.5% |
| F1-Score | 86.7% |

**Confusion Matrix (320 test reviews):**

|  | Pred Genuine | Pred Deceptive |
|---|---|---|
| Actual Genuine | 137 (TN) | 23 (FP) |
| Actual Deceptive | 20 (FN) | 140 (TP) |

**Key Finding:** Deceptive reviews use vague, atmospheric language ("luxury," "atmosphere," "would recommend") while genuine reviews use concrete operational detail ("floor," "elevator," "concierge").

## Power BI

The `powerbi/` folder contains:
- `scored_reviews.csv` — All 1,600 reviews with model predictions and probabilities
- `dashboard_metrics.json` — Computed DAX-equivalent metrics
- `hotel_summary.csv` — Per-hotel deception rates

Import `scored_reviews.csv` into Power BI via Power Query to build the dashboards.

## References

- Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). Finding deceptive opinion spam by any stretch of the imagination. *ACL-HLT 2011*.
- Ott, M., Cardie, C., & Hancock, J. T. (2013). Negative deceptive opinion spam. *NAACL-HLT 2013*.

## License

This project is submitted as academic coursework. The Deceptive Opinion Spam Corpus is released for academic/non-commercial use by its authors.
