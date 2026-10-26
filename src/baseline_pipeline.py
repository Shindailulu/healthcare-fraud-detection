"""Baseline model pipeline: One-Hot Encoding + Scaled Numerical Features + Logistic Regression.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_processing import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
)


def fit_baseline_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit OneHotEncoder, StandardScaler, and LogisticRegression baseline on training data only.
    
    Returns:
        (baseline_artifacts_dict, X_train_baseline_features)
    """
    # 1. One-hot encoder on categorical training data only (handling unknown categories safely)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_cat_encoded = encoder.fit_transform(X_train[CATEGORICAL_COLS])

    # 2. Scaler on numerical training data only
    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(X_train[NUMERICAL_COLS])

    # 3. Combine features
    X_train_combined = np.hstack([train_cat_encoded, train_num_scaled])

    # 4. Logistic regression with balanced class weights
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=random_state,
    )
    classifier.fit(X_train_combined, y_train)

    encoded_feature_names = list(encoder.get_feature_names_out(CATEGORICAL_COLS)) + list(NUMERICAL_COLS)

    artifacts = {
        "encoder": encoder,
        "scaler": scaler,
        "classifier": classifier,
        "categorical_columns": CATEGORICAL_COLS,
        "numerical_columns": NUMERICAL_COLS,
        "encoded_feature_names": encoded_feature_names,
        "num_encoded_features": len(encoded_feature_names),
        "training_rows": len(X_train),
    }

    return artifacts, X_train_combined


def transform_baseline_features(
    X_df: pd.DataFrame,
    artifacts: dict[str, Any],
) -> np.ndarray:
    """Transform arbitrary claims using fitted baseline artifacts."""
    encoder = artifacts["encoder"]
    scaler = artifacts["scaler"]
    cat_cols = artifacts["categorical_columns"]
    num_cols = artifacts["numerical_columns"]

    # Fill categorical missing with Unknown
    cat_data = X_df[cat_cols].fillna("Unknown").astype(str)
    cat_encoded = encoder.transform(cat_data)

    num_data = X_df[num_cols].fillna(0)
    num_scaled = scaler.transform(num_data)

    return np.hstack([cat_encoded, num_scaled])
