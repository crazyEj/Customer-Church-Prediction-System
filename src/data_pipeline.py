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

    # Fix: TotalCharges has 11 blank-string rows (all tenure=0, new customers).
    # Coerce to numeric; blanks become NaN, then we fill with 0 since
    # zero tenure genuinely means zero total charges billed so far.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Encode target: Yes/No -> 1/0, based on config's positive_label
    target_col = config["target"]["column"]
    positive_label = config["target"]["positive_label"]
    df[target_col] = (df[target_col] == positive_label).astype(int)

    # Drop the ID column — it's an identifier, not a predictive feature
    df = df.drop(columns=[config["id_column"]])

    return df