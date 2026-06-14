"""
Project Configuration
=====================
Central configuration for paths, constants, and parameters.
"""
import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")
POWERBI = os.path.join(PROJECT_ROOT, "powerbi")

RAW_CORPUS_FILE = os.path.join(DATA_RAW, "deceptive_opinion.csv")

# --- Corpus schema (no header in raw CSV) ---
CORPUS_COLUMNS = ["label", "hotel", "polarity", "source", "text"]
# label: 0 = deceptive (MTurk), 1 = truthful (real travel sites)

# --- Preprocessing ---
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.25  # of remaining after test split -> 60/20/20 overall
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

# --- Models ---
MODELS_CONFIG = {
    "Logistic Regression": {
        "max_iter": 2000,
        "C": 1.0,
    },
    "Linear SVM": {
        "max_iter": 3000,
        "C": 1.0,
    },
    "Random Forest": {
        "n_estimators": 300,
        "max_depth": None,
    },
}

# --- Cross-validation ---
CV_FOLDS = 5

# --- Sentiment mismatch threshold ---
MISMATCH_THRESHOLD = 0.5
