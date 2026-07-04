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


def get_models(y_train) -> dict:
    """
    Define the three model candidates. Each handles class imbalance
    in the way appropriate to its algorithm family:
      - Logistic Regression & Random Forest: class_weight='balanced'
        (sklearn automatically upweights the minority class based
        on its inverse frequency in y_train)
      - XGBoost: scale_pos_weight (its own equivalent mechanism —
        it doesn't accept class_weight)
    """
    # scale_pos_weight = (# negative samples) / (# positive samples)
    # This tells XGBoost to treat each churn example as this many
    # times more important than a non-churn example during training.
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }
    return models


def train_and_compare(models: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Train each model, evaluate it, and return a comparison table."""
    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        metrics["model_object"] = model
        results.append(metrics)

    return pd.DataFrame(results)


def select_best_model(results_df: pd.DataFrame):
    """
    Select the winner by F1 score — the balance of precision and
    recall — rather than accuracy or raw ROC-AUC, since those can
    look misleadingly good under class imbalance (see EDA: 73/27
    churn split).
    """
    best_row = results_df.loc[results_df["f1"].idxmax()]
    print(f"\n{'='*50}")
    print(f"Best model by F1 score: {best_row['model']} (F1={best_row['f1']:.4f})")
    print(f"{'='*50}")
    return best_row["model_object"], best_row["model"]


def main():
    config = load_config()

    print("Loading processed train/test data...")
    X_train, X_test, y_train, y_test = load_processed_data(config)

    models = get_models(y_train)
    results_df = train_and_compare(models, X_train, y_train, X_test, y_test)

    print("\n--- Model Comparison ---")
    print(results_df[["model", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
          .to_string(index=False))

    best_model, best_name = select_best_model(results_df)

    Path("saved_models").mkdir(exist_ok=True)
    joblib.dump(best_model, config["model"]["saved_path"])
    print(f"\nSaved best model ({best_name}) to {config['model']['saved_path']}")


if __name__ == "__main__":
    main()