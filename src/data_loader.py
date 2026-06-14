"""
Data Loader & Profiler
======================
Loads the Deceptive Opinion Spam Corpus v1.4 and computes
dataset statistics for reporting.

Dataset source: Ott et al. (2011, 2013)
    - 1,600 hotel reviews (800 deceptive + 800 truthful)
    - 20 most-reviewed Chicago hotels
    - Balanced positive/negative polarity
"""
import pandas as pd
import logging
from src.config import RAW_CORPUS_FILE, CORPUS_COLUMNS

logger = logging.getLogger(__name__)


def load_corpus(filepath=None):
    """Load the raw corpus CSV (no header) and assign column names."""
    filepath = filepath or RAW_CORPUS_FILE
    df = pd.read_csv(filepath, header=None, names=CORPUS_COLUMNS)
    logger.info(f"Loaded {len(df)} reviews from {filepath}")
    return df


def profile_dataset(df):
    """Compute and print dataset statistics for the report."""
    df = df.copy()
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))

    stats = {
        "total_records": len(df),
        "deceptive": int((df["label"] == 0).sum()),
        "genuine": int((df["label"] == 1).sum()),
        "hotels": int(df["hotel"].nunique()),
        "positive_polarity": int((df["polarity"] == "positive").sum()),
        "negative_polarity": int((df["polarity"] == "negative").sum()),
        "word_count_mean": round(df["word_count"].mean(), 1),
        "word_count_median": int(df["word_count"].median()),
        "word_count_min": int(df["word_count"].min()),
        "word_count_max": int(df["word_count"].max()),
        "word_count_std": round(df["word_count"].std(), 1),
        "deceptive_mean_words": round(
            df[df["label"] == 0]["word_count"].mean(), 1
        ),
        "genuine_mean_words": round(
            df[df["label"] == 1]["word_count"].mean(), 1
        ),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated(subset=["text"]).sum()),
    }

    logger.info("--- Dataset Profile ---")
    for key, val in stats.items():
        logger.info(f"  {key}: {val}")

    return stats
