"""Unit tests for dataset and record validation rules.
"""
import pandas as pd
import pytest

from src.validation import (
    validate_training_dataset,
    validate_inference_record,
    validate_batch_dataset,
    EXPECTED_COLUMNS,
    ESSENTIAL_COLUMNS,
)


def test_empty_dataset_validation():
    df_empty = pd.DataFrame()
    is_valid, errors, summary = validate_training_dataset(df_empty)
    assert not is_valid
    assert any("empty" in e.lower() for e in errors)


def test_missing_target_column():
    data = {col: [1, 2] for col in ESSENTIAL_COLUMNS if col != "ClaimLegitimacy"}
    df = pd.DataFrame(data)
    is_valid, errors, summary = validate_training_dataset(df)
    assert not is_valid
    assert any("ClaimLegitimacy" in e for e in errors)


def test_single_class_target():
    data = {col: ["test", "test"] for col in ESSENTIAL_COLUMNS}
    data["ClaimLegitimacy"] = ["Legitimate", "Legitimate"]
    df = pd.DataFrame(data)
    is_valid, errors, summary = validate_training_dataset(df)
    assert not is_valid
    assert any("only one class" in e.lower() for e in errors)


def test_target_normalization_and_whitespace():
    data = {col: ["1", "2"] for col in ESSENTIAL_COLUMNS}
    data["ClaimLegitimacy"] = [" Fraud ", " legitimate "]
    df = pd.DataFrame(data)
    is_valid, errors, summary = validate_training_dataset(df)
    assert is_valid
    assert summary["target_distribution"]["fraud"] == 1
    assert summary["target_distribution"]["legitimate"] == 1


def test_validate_inference_record_valid():
    record = {
        "ClaimAmount": 2500.50,
        "PatientAge": 45,
        "PatientIncome": 60000.0,
        "ClaimDate": "2024-03-15",
    }
    is_valid, errors = validate_inference_record(record)
    assert is_valid
    assert len(errors) == 0


def test_validate_inference_record_invalid_negative_amount():
    record = {
        "ClaimAmount": -50.0,
        "PatientAge": 30,
        "PatientIncome": 40000.0,
        "ClaimDate": "2024-01-01",
    }
    is_valid, errors = validate_inference_record(record)
    assert not is_valid
    assert any("Claim Amount cannot be negative" in e for e in errors)


def test_validate_inference_record_invalid_age():
    record = {
        "ClaimAmount": 1000.0,
        "PatientAge": 150,
        "PatientIncome": 50000.0,
        "ClaimDate": "2024-01-01",
    }
    is_valid, errors = validate_inference_record(record)
    assert not is_valid
    assert any("Patient Age must be between 0 and 120" in e for e in errors)


def test_validate_batch_dataset_missing_columns():
    df = pd.DataFrame({"PatientAge": [25, 40]})
    is_valid, errors = validate_batch_dataset(df)
    assert not is_valid
    assert any("missing required" in e.lower() for e in errors)
