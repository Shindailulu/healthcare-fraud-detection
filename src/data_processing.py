"""Data loading, cleaning, feature derivation, and splitting.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = [
    "PatientGender",
    "ProviderSpecialty",
    "PatientMaritalStatus",
    "PatientEmploymentStatus",
    "ClaimType",
    "ClaimSubmissionMethod",
]

NUMERICAL_COLS = [
    "ClaimAmount",
    "PatientAge",
    "PatientIncome",
    "ClaimYear",
    "ClaimMonth",
    "ClaimDayOfWeek",
]

TARGET_COL = "ClaimLegitimacy"


def load_dataset(file_obj_or_path: Any) -> pd.DataFrame:
    """Load dataset from Excel or CSV file object or path."""
    if hasattr(file_obj_or_path, "name"):
        filename = file_obj_or_path.name
    else:
        filename = str(file_obj_or_path)

    if filename.lower().endswith(".csv"):
        return pd.read_csv(file_obj_or_path)
    return pd.read_excel(file_obj_or_path)


def derive_date_features(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Derive ClaimYear, ClaimMonth, ClaimDayOfWeek from ClaimDate.
    
    Returns:
        (dataframe_with_features, invalid_date_count)
    """
    df_copy = df.copy()
    parsed_dates = pd.to_datetime(df_copy["ClaimDate"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())

    df_copy["ClaimYear"] = parsed_dates.dt.year
    df_copy["ClaimMonth"] = parsed_dates.dt.month
    df_copy["ClaimDayOfWeek"] = parsed_dates.dt.dayofweek
    return df_copy, invalid_dates


def clean_training_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean and prepare training dataset.
    
    Returns:
        (clean_df, audit_report)
    """
    audit: dict[str, Any] = {
        "initial_rows": len(df),
        "initial_duplicates": int(df.duplicated().sum()),
    }

    # Derive date features
    df_proc, invalid_dates = derive_date_features(df)
    audit["invalid_dates"] = invalid_dates

    # Normalize target column
    norm_target = (
        df_proc[TARGET_COL]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"fraud": 1, "legitimate": 0})
    )
    df_proc["target"] = norm_target

    # Clean categorical columns (trim strings, preserve representation)
    for col in CATEGORICAL_COLS:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna("Unknown").astype(str).str.strip()
        else:
            df_proc[col] = "Unknown"

    # Coerce numerical columns
    for col in NUMERICAL_COLS:
        df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")

    # Audit missing values
    missing_before_drop = df_proc[CATEGORICAL_COLS + NUMERICAL_COLS + ["target"]].isna().sum().to_dict()
    audit["missing_before_drop"] = missing_before_drop

    # Drop rows with missing essential fields or target
    clean_df = df_proc.dropna(subset=CATEGORICAL_COLS + NUMERICAL_COLS + ["target"]).copy()
    clean_df["target"] = clean_df["target"].astype(int)

    audit["final_clean_rows"] = len(clean_df)
    audit["dropped_rows"] = audit["initial_rows"] - audit["final_clean_rows"]
    audit["fraud_count"] = int((clean_df["target"] == 1).sum())
    audit["legitimate_count"] = int((clean_df["target"] == 0).sum())
    audit["fraud_percentage"] = round((audit["fraud_count"] / max(audit["final_clean_rows"], 1)) * 100, 2)

    return clean_df, audit


def prepare_claim_inference_record(record: dict[str, Any]) -> pd.DataFrame:
    """Transform a single dictionary input into a clean single-row DataFrame."""
    claim_date = pd.to_datetime(record["ClaimDate"])
    row = {
        "PatientGender": str(record.get("PatientGender", "Unknown")).strip(),
        "ProviderSpecialty": str(record.get("ProviderSpecialty", "Unknown")).strip(),
        "PatientMaritalStatus": str(record.get("PatientMaritalStatus", "Unknown")).strip(),
        "PatientEmploymentStatus": str(record.get("PatientEmploymentStatus", "Unknown")).strip(),
        "ClaimType": str(record.get("ClaimType", "Unknown")).strip(),
        "ClaimSubmissionMethod": str(record.get("ClaimSubmissionMethod", "Unknown")).strip(),
        "ClaimAmount": float(record["ClaimAmount"]),
        "PatientAge": float(record["PatientAge"]),
        "PatientIncome": float(record["PatientIncome"]),
        "ClaimYear": float(claim_date.year),
        "ClaimMonth": float(claim_date.month),
        "ClaimDayOfWeek": float(claim_date.dayofweek),
    }
    return pd.DataFrame([row])


def make_sentences(frame: pd.DataFrame, cat_cols: list[str] | None = None) -> list[list[str]]:
    """Convert categorical fields into token sentences for Word2Vec."""
    if cat_cols is None:
        cat_cols = CATEGORICAL_COLS
    return [
        [f"{col}={str(row[col]).strip()}" for col in cat_cols]
        for _, row in frame[cat_cols].iterrows()
    ]


def split_data(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified 80/20 train/test split."""
    X = df[CATEGORICAL_COLS + NUMERICAL_COLS].copy()
    y = df["target"].astype(int)

    # Use stratified split if both classes exist and minority has >= 2 samples
    stratify = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test
