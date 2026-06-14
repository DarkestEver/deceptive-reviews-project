"""
Deceptive Review Detection Pipeline
=====================================
MBA Final Project — Keshav Kant Singh (A9920124010871)
Amity University Online, MBA (Data Science)
Guide: Kunwar Saurabh Bisen

This script runs the complete end-to-end pipeline:
    1. Load the Deceptive Opinion Spam Corpus v1.4
    2. Profile the dataset
    3. Preprocess text (clean, tokenize, lemmatize)
    4. Engineer features (TF-IDF + sentiment)
    5. Train three classifiers (LR, SVM, RF)
    6. Cross-validate for stability
    7. Evaluate best model on held-out test set
    8. Analyze feature weights
    9. Score all reviews and export for Power BI

Usage:
    python main.py
"""
import logging
import numpy as np
from src.data_loader import load_corpus, profile_dataset
from src.preprocessing import preprocess_dataframe
from src.feature_engineering import build_tfidf, compute_sentiment
from src.model_training import (
    create_target, split_data, build_models,
    train_and_validate, cross_validate,
    evaluate_on_test, save_results,
)
from src.feature_analysis import extract_feature_weights, save_feature_analysis
from src.powerbi_export import (
    score_all_reviews, compute_dashboard_metrics, export_for_powerbi,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def main():
    logger.info("=" * 60)
    logger.info("DECEPTIVE REVIEW DETECTION PIPELINE")
    logger.info("=" * 60)

    # --- Step 1: Load corpus ---
    logger.info("\n>>> STEP 1: Loading corpus")
    df = load_corpus()

    # --- Step 2: Profile dataset ---
    logger.info("\n>>> STEP 2: Profiling dataset")
    dataset_stats = profile_dataset(df)

    # --- Step 3: Preprocess text ---
    logger.info("\n>>> STEP 3: Preprocessing text")
    df = preprocess_dataframe(df)

    # --- Step 4: Feature engineering ---
    logger.info("\n>>> STEP 4: Feature engineering")
    df = compute_sentiment(df)
    X_tfidf, vectorizer = build_tfidf(df["clean_text"])
    y = create_target(df)

    # --- Step 5: Train/val/test split ---
    logger.info("\n>>> STEP 5: Splitting data (60/20/20)")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_tfidf, y)

    # --- Step 6: Train and validate models ---
    logger.info("\n>>> STEP 6: Training classifiers")
    models = build_models()
    val_results = train_and_validate(models, X_train, y_train, X_val, y_val)

    # --- Step 7: Cross-validation ---
    logger.info("\n>>> STEP 7: Cross-validation (5-fold)")
    X_trainval = np.vstack([X_train.toarray(), X_val.toarray()])
    import pandas as pd
    y_trainval = pd.concat([y_train, y_val])
    cv_results = cross_validate(X_trainval, y_trainval)

    # --- Step 8: Select best model and evaluate on test set ---
    logger.info("\n>>> STEP 8: Test set evaluation")
    best_name = max(val_results, key=lambda k: val_results[k]["metrics"]["f1"])
    best_model = val_results[best_name]["model"]
    test_metrics, cm_dict = evaluate_on_test(best_model, X_test, y_test, best_name)

    # --- Step 9: Feature analysis ---
    logger.info("\n>>> STEP 9: Feature analysis")
    dec_feats, gen_feats = extract_feature_weights(best_model, vectorizer, top_n=15)
    if dec_feats:
        save_feature_analysis(dec_feats, gen_feats)

    # --- Step 10: Save results ---
    logger.info("\n>>> STEP 10: Saving results")
    save_results(dataset_stats, val_results, cv_results, test_metrics, cm_dict, best_name)

    # --- Step 11: Score all reviews and export for Power BI ---
    logger.info("\n>>> STEP 11: Scoring all reviews & exporting for Power BI")
    df_scored, cal_model = score_all_reviews(
        best_model, X_tfidf, y, df, X_train, y_train
    )
    dashboard_metrics = compute_dashboard_metrics(df_scored)
    export_for_powerbi(df_scored, dashboard_metrics)

    # --- Done ---
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Best model: %s", best_name)
    logger.info("Test accuracy: %.1f%%", test_metrics["accuracy"] * 100)
    logger.info("Outputs: outputs/ and powerbi/")


if __name__ == "__main__":
    main()
