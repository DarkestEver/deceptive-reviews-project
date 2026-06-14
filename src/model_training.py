"""
Model Training & Evaluation
============================
Trains three classifiers (Logistic Regression, Linear SVM, Random Forest),
evaluates on validation set, selects the best, and reports held-out test metrics.
Includes 5-fold stratified cross-validation for stability check.
"""
import logging
import json
import os
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from src.config import (
    RANDOM_STATE, TEST_SIZE, VAL_SIZE, MODELS_CONFIG, CV_FOLDS, OUTPUTS,
)

logger = logging.getLogger(__name__)


def create_target(df):
    """Create binary target: 1 = deceptive (label=0 in corpus), 0 = genuine."""
    y = (df["label"] == 0).astype(int)
    logger.info("Target created: %d deceptive, %d genuine", y.sum(), (y == 0).sum())
    return y


def split_data(X, y):
    """Stratified 60/20/20 train/val/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )
    logger.info(
        "Split: Train=%d, Val=%d, Test=%d", X_train.shape[0], X_val.shape[0], X_test.shape[0]
    )
    logger.info("Train class dist: %s", dict(Counter(y_train.values)))
    logger.info("Val class dist:   %s", dict(Counter(y_val.values)))
    logger.info("Test class dist:  %s", dict(Counter(y_test.values)))
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_models():
    """Instantiate the three candidate classifiers."""
    cfg = MODELS_CONFIG
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=cfg["Logistic Regression"]["max_iter"],
            C=cfg["Logistic Regression"]["C"],
            random_state=RANDOM_STATE,
        ),
        "Linear SVM": LinearSVC(
            max_iter=cfg["Linear SVM"]["max_iter"],
            C=cfg["Linear SVM"]["C"],
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=cfg["Random Forest"]["n_estimators"],
            max_depth=cfg["Random Forest"]["max_depth"],
            random_state=RANDOM_STATE,
        ),
    }
    return models


def train_and_validate(models, X_train, y_train, X_val, y_val):
    """Train each model and evaluate on validation set."""
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = {
            "accuracy": round(accuracy_score(y_val, preds), 4),
            "precision": round(precision_score(y_val, preds), 4),
            "recall": round(recall_score(y_val, preds), 4),
            "f1": round(f1_score(y_val, preds), 4),
        }
        results[name] = {"metrics": metrics, "model": model}
        logger.info(
            "%s — Val Acc: %.4f  P: %.4f  R: %.4f  F1: %.4f",
            name, metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"],
        )
    return results


def cross_validate(X_trainval, y_trainval):
    """5-fold stratified cross-validation on combined train+val data."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name in MODELS_CONFIG:
        if name == "Linear SVM":
            m = LinearSVC(max_iter=3000, C=1.0, random_state=RANDOM_STATE)
        elif name == "Logistic Regression":
            m = LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)
        else:
            m = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        scores = cross_val_score(m, X_trainval, y_trainval, cv=skf, scoring="f1")
        cv_results[name] = {
            "mean_f1": round(scores.mean(), 4),
            "std_f1": round(scores.std(), 4),
            "folds": [round(s, 3) for s in scores],
        }
        logger.info(
            "%s CV: mean F1=%.4f ± %.4f  %s",
            name, scores.mean(), scores.std(), cv_results[name]["folds"],
        )
    return cv_results


def evaluate_on_test(model, X_test, y_test, model_name):
    """Evaluate the selected best model on the held-out test set."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
    }
    cm_dict = {
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }

    logger.info("=" * 60)
    logger.info("BEST MODEL: %s — TEST SET EVALUATION", model_name)
    logger.info("=" * 60)
    logger.info("Accuracy:  %.4f (%.1f%%)", metrics["accuracy"], metrics["accuracy"] * 100)
    logger.info("Precision: %.4f", metrics["precision"])
    logger.info("Recall:    %.4f", metrics["recall"])
    logger.info("F1-score:  %.4f", metrics["f1"])
    logger.info("Confusion Matrix: TN=%d FP=%d FN=%d TP=%d",
                cm_dict["tn"], cm_dict["fp"], cm_dict["fn"], cm_dict["tp"])
    logger.info("\n%s", classification_report(y_test, y_pred,
                target_names=["Genuine", "Deceptive"]))

    return metrics, cm_dict


def save_results(dataset_stats, val_results, cv_results, test_metrics,
                 cm_dict, best_model_name):
    """Save all results to JSON for the report and Power BI."""
    os.makedirs(OUTPUTS, exist_ok=True)
    output = {
        "dataset": dataset_stats,
        "validation": {
            name: r["metrics"] for name, r in val_results.items()
        },
        "cross_validation": cv_results,
        "best_model": best_model_name,
        "test": test_metrics,
        "confusion_matrix": cm_dict,
    }
    filepath = os.path.join(OUTPUTS, "model_results.json")
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", filepath)
    return output
