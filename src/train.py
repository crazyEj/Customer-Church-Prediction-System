import joblib
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load pipeline configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_processed_data(config: dict):
    """Load the train/test arrays saved by data_pipeline.py."""
    processed_path = config["dataset"]["processed_path"]
    # Note: data_pipeline.py actually saves to a fixed path, not
    # config's processed_path — we load from where it was written.
    X_train, X_test, y_train, y_test = joblib.load(
        "data/processed/train_test_data.pkl"
    )
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Compute imbalance-aware metrics. Accuracy is deliberately
    NOT the headline metric here — with a 73/27 class split, a
    model that always predicts 'No churn' would score ~73%
    accuracy while catching zero actual churners. Precision,
    recall, F1, and PR-AUC give a truthful picture instead.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")

    return metrics