"""
Feature Analysis
=================
Extracts and displays the top discriminative features from the trained
Linear SVM, showing which words push toward deceptive vs genuine.
"""
import logging
import numpy as np
import pandas as pd
import os
from src.config import OUTPUTS

logger = logging.getLogger(__name__)


def extract_feature_weights(model, vectorizer, top_n=15):
    """Extract the top N features for deceptive and genuine from a linear model."""
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "coef_"):
        coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
    else:
        logger.warning("Model has no coef_ attribute. Skipping feature analysis.")
        return None, None

    # Top deceptive indicators (highest positive weights)
    top_deceptive_idx = np.argsort(coefs)[-top_n:][::-1]
    deceptive_features = [
        {"rank": i + 1, "feature": feature_names[idx], "weight": round(float(coefs[idx]), 4)}
        for i, idx in enumerate(top_deceptive_idx)
    ]

    # Top genuine indicators (most negative weights)
    top_genuine_idx = np.argsort(coefs)[:top_n]
    genuine_features = [
        {"rank": i + 1, "feature": feature_names[idx], "weight": round(float(coefs[idx]), 4)}
        for i, idx in enumerate(top_genuine_idx)
    ]

    logger.info("\nTop %d DECEPTIVE indicators:", top_n)
    for f in deceptive_features:
        logger.info("  %2d. %-25s  weight=%+.4f", f["rank"], f["feature"], f["weight"])

    logger.info("\nTop %d GENUINE indicators:", top_n)
    for f in genuine_features:
        logger.info("  %2d. %-25s  weight=%+.4f", f["rank"], f["feature"], f["weight"])

    return deceptive_features, genuine_features


def save_feature_analysis(deceptive_features, genuine_features):
    """Save feature analysis to CSV for reporting."""
    os.makedirs(OUTPUTS, exist_ok=True)

    df_dec = pd.DataFrame(deceptive_features)
    df_dec["direction"] = "deceptive"
    df_gen = pd.DataFrame(genuine_features)
    df_gen["direction"] = "genuine"

    df_all = pd.concat([df_dec, df_gen], ignore_index=True)
    filepath = os.path.join(OUTPUTS, "feature_weights.csv")
    df_all.to_csv(filepath, index=False)
    logger.info("Feature weights saved to %s", filepath)
    return df_all
