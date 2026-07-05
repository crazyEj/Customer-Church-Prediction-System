import joblib
import yaml
import pandas as pd
import sys
from pathlib import Path

# Reuse the exact same cleaning logic as training, so predictions
# are guaranteed to see data prepared identically to what the
# model was trained on.
sys.path.append(str(Path(__file__).parent))
from data_pipeline import load_config


def clean_new_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply the same TotalCharges fix used in training. Unlike
    clean_data() in data_pipeline.py, this does NOT touch the
    target column (Churn) — new customer data won't have it,
    since predicting it is the whole point.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop ID column if present — same reasoning as training:
    # it's not a predictive feature.
    id_col = config["id_column"]
    if id_col in df.columns:
        df = df.drop(columns=[id_col])

    return df


def load_artifacts(config: dict):
    """Load the fitted preprocessor and trained model."""
    preprocessor = joblib.load("saved_models/preprocessor.pkl")
    model = joblib.load(config["model"]["saved_path"])
    return preprocessor, model


def predict_churn(df: pd.DataFrame, config: dict, preprocessor, model) -> pd.DataFrame:
    """
    Run the full inference pipeline on new customer data and
    return predictions alongside churn probability.
    """
    df_clean = clean_new_data(df, config)

    # IMPORTANT: .transform(), never .fit_transform() here — the
    # preprocessor must apply the exact scaling/encoding learned
    # during training, not refit new statistics on this data.
    X_processed = preprocessor.transform(df_clean)

    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1]

    results = df.copy()
    results["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in predictions]
    results["Churn_Probability"] = probabilities.round(4)

    return results


def main():
    """
    Example usage: predict churn for a small sample of customers
    from the raw dataset (simulating "new" data for demonstration).
    """
    config = load_config()
    preprocessor, model = load_artifacts(config)

    # Demo: grab a few rows from the raw dataset to simulate new customers
    raw_df = pd.read_csv(config["dataset"]["raw_path"])
    sample = raw_df.drop(columns=[config["target"]["column"]]).head(5)

    results = predict_churn(sample, config, preprocessor, model)
    print(results[["customerID", "Churn_Prediction", "Churn_Probability"]].to_string(index=False))


if __name__ == "__main__":
    main()