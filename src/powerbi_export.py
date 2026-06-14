"""
Power BI Export
================
Scores all 1,600 reviews using the trained model and exports
the orchestration dataset for Power BI ingestion.
Computes DAX-ready fields: predicted_label, auth_probability,
deceptive_probability, predicted_class, actual_class.
"""
import logging
import os
import json
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from src.config import POWERBI, OUTPUTS, MISMATCH_THRESHOLD

logger = logging.getLogger(__name__)


def score_all_reviews(model, X_all, y_all, df, X_train, y_train):
    """Score every review in the corpus using a calibrated version of the model."""
    # Calibrate the model to get probability estimates
    cal_model = CalibratedClassifierCV(model, cv=5)
    cal_model.fit(X_train, y_train)

    df = df.copy()
    df["predicted_label"] = cal_model.predict(X_all)
    df["auth_probability"] = cal_model.predict_proba(X_all)[:, 0]
    df["deceptive_probability"] = cal_model.predict_proba(X_all)[:, 1]
    df["predicted_class"] = df["predicted_label"].map({0: "Genuine", 1: "Deceptive"})
    df["actual_class"] = df["label"].map({0: "Deceptive", 1: "Genuine"})

    logger.info("Scored %d reviews. Flagged %d as deceptive.",
                len(df), df["predicted_label"].sum())
    return df, cal_model


def compute_dashboard_metrics(df):
    """Compute the DAX-equivalent metrics for the Power BI dashboards."""
    total = len(df)
    flagged = int(df["predicted_label"].sum())
    metrics = {
        "total_reviews": total,
        "flagged_deceptive": flagged,
        "deception_index": round(flagged / total, 3),
        "avg_authenticity": round(df["auth_probability"].mean(), 3),
        "sentiment_divergence": round(df["sent_rating_gap"].mean(), 3),
        "mismatch_pct": round(
            (df["sent_rating_gap"] > MISMATCH_THRESHOLD).sum() / total * 100, 1
        ),
        "mean_polarity": round(df["sentiment_polarity"].mean(), 3),
    }

    logger.info("--- Dashboard Metrics ---")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, v)

    return metrics


def export_for_powerbi(df, dashboard_metrics):
    """Export scored reviews and metrics for Power BI."""
    os.makedirs(POWERBI, exist_ok=True)

    # Full scored dataset
    export_cols = [
        "label", "hotel", "polarity", "source", "text", "clean_text",
        "word_count", "sentiment_polarity", "sentiment_subjectivity",
        "rating_scaled", "sent_rating_gap", "predicted_label",
        "auth_probability", "deceptive_probability", "predicted_class",
        "actual_class",
    ]
    filepath = os.path.join(POWERBI, "scored_reviews.csv")
    df[export_cols].to_csv(filepath, index=False)
    logger.info("Scored reviews exported to %s (%d rows)", filepath, len(df))

    # Dashboard metrics JSON
    metrics_path = os.path.join(POWERBI, "dashboard_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(dashboard_metrics, f, indent=2)
    logger.info("Dashboard metrics exported to %s", metrics_path)

    # Per-hotel summary
    hotel_summary = (
        df.groupby("hotel")
        .agg(
            total=("predicted_label", "count"),
            flagged=("predicted_label", "sum"),
            mean_auth=("auth_probability", "mean"),
        )
        .assign(deception_rate=lambda x: round(x["flagged"] / x["total"], 3))
        .sort_values("deception_rate", ascending=False)
    )
    hotel_path = os.path.join(POWERBI, "hotel_summary.csv")
    hotel_summary.to_csv(hotel_path)
    logger.info("Hotel summary exported to %s", hotel_path)

    return filepath
