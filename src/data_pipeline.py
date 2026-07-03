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
    """Load the raw dataset specified in config."""
    return pd.read_csv(config["dataset"]["raw_path"])

def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Apply dataset-specific cleaning that must happen BEFORE
    the sklearn Pipeline (scaling/encoding) ever sees the data.
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    target_col = config["target"]["column"]
    positive_label = config["target"]["positive_label"]
    df[target_col] = (df[target_col] == positive_label).astype(int)

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