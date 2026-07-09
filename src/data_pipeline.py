import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load pipeline configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load the raw dataset specified in config. Supports CSV and Excel."""
    raw_path = config["dataset"]["raw_path"]
    if raw_path.endswith((".xlsx", ".xls")):
        # E-commerce dataset ships with multiple sheets; the real
        # data lives in "E Comm", not the "Data Dict" sheet.
        return pd.read_excel(raw_path, sheet_name="E Comm")
    return pd.read_csv(raw_path)

def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply cleaning that must happen BEFORE the sklearn Pipeline
    (scaling/encoding) ever sees the data.

    Handles two classes of missingness generically, so this works
    across datasets without dataset-specific hardcoded fixes:
      1. Numerical columns with real NaN values -> median imputation
      2. Columns stored as strings that should be numeric but contain
         blank/whitespace values (e.g. Telco's TotalCharges) -> coerced
         to numeric, then median imputation
    """
    df = df.copy()

    numerical_features = config["features"]["numerical"]

    for col in numerical_features:
        if col in df.columns:
            # Coerce to numeric in case the column is stored as text
            # with stray blank/whitespace entries (Telco's TotalCharges
            # case). This is a no-op for columns that are already numeric.
            df[col] = pd.to_numeric(df[col], errors="coerce")

            if df[col].isnull().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)

    # Encode target based on config's positive_label. Handles both
    # string labels (Telco: "Yes"/"No") and pre-encoded integer
    # labels (E-commerce: already 0/1) without needing a dataset-
    # specific branch.
    target_col = config["target"]["column"]
    positive_label = config["target"]["positive_label"]
    df[target_col] = (df[target_col] == positive_label).astype(int)

    # Drop the ID column — not a predictive feature
    df = df.drop(columns=[config["id_column"]])

    return df


def build_preprocessing_pipeline(config: dict) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numerical features and
    one-hot encodes categorical features. This is fit ONLY on
    training data to prevent test-set leakage.
    """
    numerical_features = config["features"]["numerical"]
    categorical_features = config["features"]["categorical"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
        ]
    )
    return preprocessor


def split_data(df: pd.DataFrame, config: dict):
    """
    Split into train/test sets. Stratified on the target to preserve
    the ~73/27 churn class ratio in both splits — critical given
    the class imbalance we found in EDA.
    """
    target_col = config["target"]["column"]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def main():
    config = load_config()

    print("Loading raw data...")
    df = load_raw_data(config)

    print("Cleaning data...")
    df = clean_data(df, config)

    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df, config)
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  Train churn rate: {y_train.mean():.3f}, Test churn rate: {y_test.mean():.3f}")

    print("Building and fitting preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline(config)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save the fitted preprocessor — predict.py will load this exact
    # object so new customer data gets the identical transformation
    # (same scaling parameters, same one-hot columns) as training data.
    Path("saved_models").mkdir(exist_ok=True)
    joblib.dump(preprocessor, "saved_models/preprocessor.pkl")
    print("Saved fitted preprocessor to saved_models/preprocessor.pkl")

    # Save processed splits so train.py doesn't need to redo this work
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    joblib.dump((X_train_processed, X_test_processed, y_train, y_test),
                "data/processed/train_test_data.pkl")
    print("Saved processed train/test data to data/processed/train_test_data.pkl")


if __name__ == "__main__":
    main()