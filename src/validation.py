"""Validation rules and schema verification for healthcare fraud claims data.
"""
from __future__ import annotations

from typing import Any
import pandas as pd

EXPECTED_COLUMNS = [
    "ClaimID",
    "PatientID",
    "ProviderID",
    "ClaimAmount",
    "ClaimDate",
    "DiagnosisCode",
    "ProcedureCode",
    "PatientAge",
    "PatientGender",
    "ProviderSpecialty",
    "ClaimStatus",
    "PatientIncome",
    "PatientMaritalStatus",
    "PatientEmploymentStatus",
    "ProviderLocation",
    "ClaimType",
    "ClaimSubmissionMethod",
    "Cluster",
    "ClaimLegitimacy",
]

ESSENTIAL_COLUMNS = [
    "ClaimAmount",
    "ClaimDate",
    "PatientAge",
    "PatientGender",
    "ProviderSpecialty",
    "PatientIncome",
    "PatientMaritalStatus",
    "PatientEmploymentStatus",
    "ClaimType",
    "ClaimSubmissionMethod",
    "ClaimLegitimacy",
]

INFERENCE_REQUIRED_COLUMNS = [
    "ClaimAmount",
    "ClaimDate",
    "PatientAge",
    "PatientGender",
    "ProviderSpecialty",
    "PatientIncome",
    "PatientMaritalStatus",
    "PatientEmploymentStatus",
    "ClaimType",
    "ClaimSubmissionMethod",
]

EXCLUDED_COLUMNS_INFO = {
    "ClaimID": "Patient and claim identifiers do not provide generalizable fraud patterns and cause overfitting.",
    "PatientID": "Patient identifier excluded to prevent memorization and preserve patient privacy.",
    "ProviderID": "Provider identifier excluded from initial feature set to avoid provider-specific bias.",
    "Cluster": "Pre-computed cluster labels may contain target leakage from unsupervised/semi-supervised partitioning.",
    "ClaimStatus": "Claim status is assigned after adjudication/investigation, creating direct target leakage.",
    "DiagnosisCode": "Diagnosis codes are nearly unique (high cardinality) in this dataset, unsuitable for reliable Word2Vec token representation.",
    "ProcedureCode": "Procedure codes are nearly unique in this dataset and excluded from the initial Word2Vec embedding.",
    "ProviderLocation": "Provider location is excluded to focus on patient demographics and submission patterns.",
}


def validate_training_dataset(df: pd.DataFrame) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate that the uploaded training dataset has required columns and valid target.
    
    Returns:
        (is_valid, error_messages, summary_info)
    """
    errors: list[str] = []
    summary: dict[str, Any] = {}

    if df is None or df.empty:
        return False, ["The uploaded dataset is completely empty."], summary

    summary["total_rows"] = len(df)
    summary["total_cols"] = len(df.columns)
    summary["columns"] = list(df.columns)

    # Check for target column
    if "ClaimLegitimacy" not in df.columns:
        errors.append("Missing required target column 'ClaimLegitimacy'.")
    else:
        norm_target = (
            df["ClaimLegitimacy"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )
        unique_targets = set(norm_target.unique())
        expected_classes = {"fraud", "legitimate"}
        found_expected = unique_targets.intersection(expected_classes)
        
        if not found_expected:
            errors.append(
                f"Target 'ClaimLegitimacy' does not contain 'Fraud' or 'Legitimate' values. Found: {sorted(list(unique_targets))}"
            )
        elif len(found_expected) < 2:
            errors.append(
                f"Target 'ClaimLegitimacy' contains only one class ({list(found_expected)}). Both 'Fraud' and 'Legitimate' are required for model training."
            )
        summary["target_distribution"] = norm_target.value_counts().to_dict()

    # Check essential columns
    missing_essentials = [col for col in ESSENTIAL_COLUMNS if col not in df.columns]
    if missing_essentials:
        errors.append(f"Missing essential feature column(s): {', '.join(missing_essentials)}.")

    # Check missing values
    summary["missing_per_column"] = df.isnull().sum().to_dict()
    summary["duplicate_rows"] = int(df.duplicated().sum())

    is_valid = len(errors) == 0
    return is_valid, errors, summary


def validate_inference_record(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a single claim dictionary entered for screening."""
    errors: list[str] = []

    # ClaimAmount
    try:
        amt = float(record.get("ClaimAmount", -1))
        if amt < 0:
            errors.append("Claim Amount cannot be negative.")
    except (ValueError, TypeError):
        errors.append("Claim Amount must be a valid numerical value.")

    # PatientAge
    try:
        age = int(record.get("PatientAge", -1))
        if age < 0 or age > 120:
            errors.append("Patient Age must be between 0 and 120 years.")
    except (ValueError, TypeError):
        errors.append("Patient Age must be a valid integer.")

    # PatientIncome
    try:
        income = float(record.get("PatientIncome", -1))
        if income < 0:
            errors.append("Patient Income cannot be negative.")
    except (ValueError, TypeError):
        errors.append("Patient Income must be a valid numerical value.")

    # ClaimDate
    if not record.get("ClaimDate"):
        errors.append("Claim Date is required.")

    return len(errors) == 0, errors


def validate_batch_dataset(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate that an uploaded batch dataset has all required inference fields."""
    errors: list[str] = []
    if df is None or df.empty:
        return False, ["Batch claims file is empty."]

    missing_cols = [col for col in INFERENCE_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f"Batch file is missing required screening columns: {', '.join(missing_cols)}")

    return len(errors) == 0, errors
