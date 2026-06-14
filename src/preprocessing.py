"""
Text Preprocessing Pipeline
============================
Implements the NLP preprocessing pipeline described in Chapter 3:
    1. Lowercasing
    2. Non-alphabetic character removal
    3. Tokenization
    4. Stop-word removal (NLTK English)
    5. Lemmatization (WordNet Lemmatizer)
"""
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

# Download required NLTK data (silent)
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """Clean a single review text through the full NLP pipeline."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in tokens
        if t not in _stop_words and len(t) > 2
    ]
    return " ".join(tokens)


def preprocess_dataframe(df):
    """Apply preprocessing to the entire dataframe and add computed columns."""
    df = df.copy()
    logger.info("Preprocessing %d reviews...", len(df))

    df["clean_text"] = df["text"].apply(preprocess_text)
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))

    logger.info("Preprocessing complete. Added clean_text and word_count columns.")
    return df
