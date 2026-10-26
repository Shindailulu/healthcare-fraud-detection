"""Prediction logic for single claims and batch datasets, plus model serialization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd

from src.data_processing import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    prepare_claim_inference_record,
    derive_date_features,
)
from src.word2vec_pipeline import transform_word2vec_features


def screen_single_claim(
    record: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Screen a single claim using fitted Word2Vec model and scaler.
    
    Returns:
        Result dictionary containing prediction, probability, status label, and advisory.
    """
    input_df = prepare_claim_inference_record(record)
    features, zero_count = transform_word2vec_features(input_df, artifacts)

    classifier = artifacts["classifier"]
    pred_class = int(classifier.predict(features)[0])
    
    # Probabilities
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)[0]
        # class 1 probability
        fraud_proba = float(probabilities[1])
    else:
        fraud_proba = None

    status_wording = (
        "Flagged for review" if pred_class == 1 else "Not flagged by this model"
    )
    predicted_label = "Fraud" if pred_class == 1 else "Legitimate"

    return {
        "predicted_class": pred_class,
        "predicted_label": predicted_label,
        "fraud_probability": fraud_proba,
        "status_wording": status_wording,
        "zero_vector_tokens": zero_count > 0,
        "disclaimer": (
            "This result is a model-generated screening signal, not proof of fraud. "
            "A qualified human reviewer must assess the claim."
        ),
        "input_record": record,
    }


def screen_batch_claims(
    batch_df: pd.DataFrame,
    artifacts: dict[str, Any],
) -> pd.DataFrame:
    """Screen a batch dataframe of claims and return original dataframe with predictions."""
    df_copy = batch_df.copy()

    # Ensure date derived features are present
    df_proc, _ = derive_date_features(df_copy)

    # Clean categorical columns
    cat_cols = artifacts["categorical_columns"]
    for col in cat_cols:
        if col in df_proc.columns:
            df_proc[col] = df_proc[col].fillna("Unknown").astype(str).str.strip()
        else:
            df_proc[col] = "Unknown"

    num_cols = artifacts["numerical_columns"]
    for col in num_cols:
        if col in df_proc.columns:
            df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce").fillna(0)
        else:
            df_proc[col] = 0.0

    features, _ = transform_word2vec_features(df_proc, artifacts)
    classifier = artifacts["classifier"]

    predictions = classifier.predict(features)
    probabilities = (
        classifier.predict_proba(features)[:, 1]
        if hasattr(classifier, "predict_proba")
        else [None] * len(predictions)
    )

    df_copy["PredictedCategory"] = ["Fraud" if p == 1 else "Legitimate" for p in predictions]
    df_copy["ScreeningStatus"] = [
        "Flagged for review" if p == 1 else "Not flagged by this model"
        for p in predictions
    ]
    df_copy["EstimatedFraudProbability"] = [
        round(float(pr), 4) if pr is not None else "N/A" for pr in probabilities
    ]

    return df_copy


def save_trained_system(
    save_dir: Path | str,
    w2v_artifacts: dict[str, Any],
    baseline_artifacts: dict[str, Any],
    metrics_summary: dict[str, Any],
) -> Path:
    """Save models, scalers, encoders, and evaluation metadata."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    bundle = {
        "word2vec_artifacts": w2v_artifacts,
        "baseline_artifacts": baseline_artifacts,
        "metrics_summary": metrics_summary,
    }
    target_file = save_path / "healthcare_fraud_system.joblib"
    joblib.dump(bundle, target_file)
    return target_file


def load_trained_system(model_file: Path | str) -> dict[str, Any] | None:
    """Load system bundle from disk if exists."""
    path = Path(model_file)
    if not path.exists():
        return None
    try:
        data = joblib.load(path)
        return data
    except Exception:
        return None
