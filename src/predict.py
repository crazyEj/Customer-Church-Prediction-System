import joblib
import yaml
import pandas as pd
import shap
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression

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
    """
    Load the fitted preprocessor and trained model. Both are saved
    per-dataset by data_pipeline.py / train.py (e.g.
    preprocessor_telco_churn.pkl), so the preprocessor path must be
    derived from config["dataset"]["name"] rather than hardcoded —
    otherwise this silently loads Telco's preprocessor for an
    e-commerce config, or fails to find a file at all.
    """
    dataset_name = config["dataset"]["name"]
    preprocessor = joblib.load(f"saved_models/preprocessor_{dataset_name}.pkl")
    model = joblib.load(config["model"]["saved_path"])
    return preprocessor, model


def load_training_background(config: dict):
    """
    Load the X_train split saved by data_pipeline.py, used as SHAP's
    background reference. Dataset-aware for the same reason as
    load_artifacts() above.
    """
    dataset_name = config["dataset"]["name"]
    data_path = f"data/processed/train_test_data_{dataset_name}.pkl"
    X_train, X_test, y_train, y_test = joblib.load(data_path)
    return X_train


def get_shap_explainer(model, X_train_background, feature_names):
    """
    Select the right SHAP explainer for the model family. This
    matters because the "best model" differs by dataset (Logistic
    Regression for Telco, XGBoost for the e-commerce config) — a
    hardcoded LinearExplainer works for the former but throws or
    silently mis-explains the latter.

    - Logistic Regression (linear): LinearExplainer, which needs an
      explicit background sample to compare each customer against.
    - Random Forest / XGBoost (tree-based): TreeExplainer, which
      reads contributions directly from the tree structure and
      doesn't need a background sample.
    """
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, X_train_background, feature_names=feature_names)
    return shap.TreeExplainer(model, feature_names=feature_names)


def _positive_class_shap_values(shap_values):
    """
    Normalize SHAP output across explainer types to a 2D
    (n_samples, n_features) array of contributions toward the
    positive (churn) class. TreeExplainer on some model/SHAP version
    combinations returns a 3rd axis for per-class values; when that
    happens, take the churn=1 slice.
    """
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


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
    df_clean = clean_new_data(df, config)
    X_processed = preprocessor.transform(df_clean)
    feature_names = preprocessor.get_feature_names_out()

    explainer = get_shap_explainer(model, X_train_background, feature_names)
    shap_values = explainer(X_processed)
    values = _positive_class_shap_values(shap_values)

    contributions = list(zip(feature_names, values[0]))
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


def predict_batch(df: pd.DataFrame, config: dict, preprocessor, model, X_train_background, top_n: int = 5) -> pd.DataFrame:
    """
    Score a batch of customers in one vectorized pass and attach a
    human-readable summary of each row's top SHAP drivers.

    This computes SHAP values for the whole batch in a single
    explainer call rather than looping explain_prediction() per row —
    for LinearExplainer that's a single matrix multiply either way,
    but for TreeExplainer batching avoids re-walking the tree
    structure once per customer, which matters once a batch runs
    into the hundreds or thousands of rows.
    """
    df_clean = clean_new_data(df, config)
    X_processed = preprocessor.transform(df_clean)
    feature_names = preprocessor.get_feature_names_out()

    predictions = model.predict(X_processed)
    probabilities = model.predict_proba(X_processed)[:, 1]

    explainer = get_shap_explainer(model, X_train_background, feature_names)
    shap_values = explainer(X_processed)
    values = _positive_class_shap_values(shap_values)

    top_driver_summaries = []
    for row_values in values:
        contributions = list(zip(feature_names, row_values))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        parts = [f"{feature} ({value:+.3f})" for feature, value in contributions[:top_n]]
        top_driver_summaries.append("; ".join(parts))

    results = df.copy()
    results["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in predictions]
    results["Churn_Probability"] = probabilities.round(4)
    results["Top_Risk_Drivers"] = top_driver_summaries

    return results


def main():
    config = load_config()
    preprocessor, model = load_artifacts(config)

    raw_df = pd.read_csv(config["dataset"]["raw_path"])
    sample = raw_df.drop(columns=[config["target"]["column"]]).head(5)

    results = predict_churn(sample, config, preprocessor, model)
    print(results[["customerID", "Churn_Prediction", "Churn_Probability"]].to_string(index=False))

    # Load training data to use as SHAP's background reference
    X_train = load_training_background(config)

    print("\n--- Explanation for first customer ---")
    explanations = explain_prediction(sample.iloc[[0]], config, preprocessor, model, X_train)
    for exp in explanations:
        print(f"  {exp['feature']}: {exp['shap_value']:+.4f} ({exp['effect']})")

    print("\n--- Batch prediction on all 5 sample customers ---")
    batch_results = predict_batch(sample, config, preprocessor, model, X_train)
    print(batch_results[["customerID", "Churn_Prediction", "Churn_Probability", "Top_Risk_Drivers"]].to_string(index=False))

if __name__ == "__main__":
    main()