import joblib
import yaml
import pandas as pd
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
import sys
from pathlib import Path

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

    X_processed = preprocessor.transform(df_clean)

    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1]

    results = df.copy()
    results["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in predictions]
    results["Churn_Probability"] = probabilities.round(4)

    return results


def explain_prediction(df: pd.DataFrame, config: dict, preprocessor, model, X_train_background, top_n: int = 5) -> list:
    """
    Return the top N features driving a single customer's churn
    prediction, using SHAP. X_train_background provides the
    reference distribution SHAP compares this customer against —
    using the single customer's own row as background (as before)
    collapses all contributions to zero, since there's no variation
    to compare against.
    """
    if not SHAP_AVAILABLE:
        return [{"feature": "N/A", "shap_value": 0, "effect": "SHAP unavailable on this system"}]

    df_clean = clean_new_data(df, config)
    X_processed = preprocessor.transform(df_clean)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.LinearExplainer(model, X_train_background, feature_names=feature_names)
    shap_values = explainer(X_processed)

    contributions = list(zip(feature_names, shap_values.values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    explanations = []
    for feature, value in contributions[:top_n]:
        direction = "increases" if value > 0 else "decreases"
        explanations.append({
            "feature": feature,
            "shap_value": round(float(value), 4),
            "effect": f"{direction} churn risk",
        })

    return explanations


def main():
    config = load_config()
    preprocessor, model = load_artifacts(config)

    raw_df = pd.read_csv(config["dataset"]["raw_path"])
    sample = raw_df.drop(columns=[config["target"]["column"]]).head(5)

    results = predict_churn(sample, config, preprocessor, model)
    print(results[["customerID", "Churn_Prediction", "Churn_Probability"]].to_string(index=False))

    # Load training data to use as SHAP's background reference
    X_train, X_test, y_train, y_test = joblib.load("data/processed/train_test_data.pkl")

    print("\n--- Explanation for first customer ---")
    explanations = explain_prediction(sample.iloc[[0]], config, preprocessor, model, X_train)
    for exp in explanations:
        print(f"  {exp['feature']}: {exp['shap_value']:+.4f} ({exp['effect']})")


if __name__ == "__main__":
    main()