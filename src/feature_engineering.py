"""
Feature Engineering
===================
Builds the feature matrix from preprocessed text:
    - TF-IDF vectorization (bigrams, 5000 features)
    - Sentiment polarity & subjectivity (TextBlob)
    - Rating-scaled mapping for sentiment-behavior gap
"""
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from src.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF

logger = logging.getLogger(__name__)


def build_tfidf(texts, fit=True, vectorizer=None):
    """Build or apply a TF-IDF vectorizer on the given texts."""
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
        )
        X = vectorizer.fit_transform(texts)
        logger.info(
            "TF-IDF fitted: %d documents x %d features", X.shape[0], X.shape[1]
        )
    else:
        X = vectorizer.transform(texts)
    return X, vectorizer


def compute_sentiment(df):
    """Compute TextBlob sentiment polarity and subjectivity for each review."""
    df = df.copy()
    logger.info("Computing sentiment scores for %d reviews...", len(df))

    df["sentiment_polarity"] = df["text"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )
    df["sentiment_subjectivity"] = df["text"].apply(
        lambda x: TextBlob(str(x)).sentiment.subjectivity
    )
    # Map corpus polarity label to numeric for comparison
    df["rating_scaled"] = df["polarity"].map({"positive": 0.5, "negative": -0.5})
    # Sentiment-rating gap (used for Sentiment Extremity Divergence)
    df["sent_rating_gap"] = np.abs(df["sentiment_polarity"] - df["rating_scaled"])

    logger.info("Sentiment computation complete.")
    return df
