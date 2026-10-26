"""Unit tests for preprocessing, pipelines, and evaluation logic.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    clean_training_dataset,
    derive_date_features,
    split_data,
    make_sentences,
    prepare_claim_inference_record,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
)
from src.word2vec_pipeline import (
    fit_word2vec_pipeline,
    transform_word2vec_features,
)
from src.baseline_pipeline import (
    fit_baseline_pipeline,
    transform_baseline_features,
)
from src.evaluation import (
    compute_model_metrics,
    build_comparison_dataframe,
)


@pytest.fixture
def sample_claims_dataframe():
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    data = {
        "ClaimID": [f"CLM{i:04d}" for i in range(n)],
        "PatientID": [f"PAT{i%10:04d}" for i in range(n)],
        "ProviderID": [f"PRV{i%5:04d}" for i in range(n)],
        "ClaimAmount": np.random.uniform(500, 15000, size=n),
        "ClaimDate": dates,
        "DiagnosisCode": [f"DX{i}" for i in range(n)],
        "ProcedureCode": [f"PX{i}" for i in range(n)],
        "PatientAge": np.random.randint(18, 85, size=n),
        "PatientGender": np.random.choice(["Male", "Female"], size=n),
        "ProviderSpecialty": np.random.choice(["Cardiology", "Internal Medicine", "Oncology"], size=n),
        "ClaimStatus": ["Approved"] * n,
        "PatientIncome": np.random.uniform(20000, 100000, size=n),
        "PatientMaritalStatus": np.random.choice(["Married", "Single", "Divorced"], size=n),
        "PatientEmploymentStatus": np.random.choice(["Employed", "Retired", "Unemployed"], size=n),
        "ProviderLocation": ["Lagos"] * n,
        "ClaimType": np.random.choice(["Inpatient", "Outpatient"], size=n),
        "ClaimSubmissionMethod": np.random.choice(["Electronic", "Paper"], size=n),
        "Cluster": [1] * n,
        "ClaimLegitimacy": ["Legitimate"] * 50 + ["Fraud"] * 10,
    }
    return pd.DataFrame(data)


def test_derive_date_features():
    df = pd.DataFrame({"ClaimDate": ["2024-05-18", "2023-11-05", "invalid-date"]})
    df_derived, invalid_cnt = derive_date_features(df)
    assert invalid_cnt == 1
    assert df_derived["ClaimYear"].iloc[0] == 2024
    assert df_derived["ClaimMonth"].iloc[0] == 5
    assert df_derived["ClaimDayOfWeek"].iloc[0] == 5  # Saturday is 5


def test_clean_training_dataset(sample_claims_dataframe):
    clean_df, audit = clean_training_dataset(sample_claims_dataframe)
    assert audit["final_clean_rows"] == 60
    assert audit["fraud_count"] == 10
    assert audit["legitimate_count"] == 50
    assert set(clean_df["target"].unique()) == {0, 1}


def test_stratified_splitting(sample_claims_dataframe):
    clean_df, _ = clean_training_dataset(sample_claims_dataframe)
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.20, random_state=42)
    assert len(X_train) == 48
    assert len(X_test) == 12
    # Verify both classes present in test split
    assert (y_test == 1).sum() == 2
    assert (y_test == 0).sum() == 10


def test_word2vec_and_baseline_pipelines(sample_claims_dataframe):
    clean_df, _ = clean_training_dataset(sample_claims_dataframe)
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.20, random_state=42)

    # Word2Vec pipeline
    w2v_artifacts, X_train_w2v, train_zeros = fit_word2vec_pipeline(X_train, y_train)
    assert "word2vec" in w2v_artifacts
    assert w2v_artifacts["vocabulary_size"] > 0
    assert X_train_w2v.shape[0] == len(X_train)

    X_test_w2v, test_zeros = transform_word2vec_features(X_test, w2v_artifacts)
    assert X_test_w2v.shape[0] == len(X_test)

    # Predictions
    w2v_preds = w2v_artifacts["classifier"].predict(X_test_w2v)
    w2v_probas = w2v_artifacts["classifier"].predict_proba(X_test_w2v)[:, 1]
    w2v_metrics = compute_model_metrics(y_test, w2v_preds, w2v_probas, model_name="Word2Vec")
    assert 0.0 <= w2v_metrics["accuracy"] <= 1.0
    assert 0.0 <= w2v_metrics["recall"] <= 1.0
    assert w2v_metrics["confusion_matrix"] is not None

    # Baseline pipeline
    base_artifacts, X_train_base = fit_baseline_pipeline(X_train, y_train)
    assert "encoder" in base_artifacts
    X_test_base = transform_baseline_features(X_test, base_artifacts)
    assert X_test_base.shape[0] == len(X_test)

    base_preds = base_artifacts["classifier"].predict(X_test_base)
    base_probas = base_artifacts["classifier"].predict_proba(X_test_base)[:, 1]
    base_metrics = compute_model_metrics(y_test, base_preds, base_probas, model_name="Baseline")
    assert 0.0 <= base_metrics["accuracy"] <= 1.0

    # Comparison dataframe
    comp_df = build_comparison_dataframe(w2v_metrics, base_metrics)
    assert not comp_df.empty
    assert "Word2Vec + LogReg" in comp_df.columns
    assert "One-Hot Baseline + LogReg" in comp_df.columns


def test_unknown_category_robustness(sample_claims_dataframe):
    clean_df, _ = clean_training_dataset(sample_claims_dataframe)
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.20, random_state=42)
    w2v_artifacts, _, _ = fit_word2vec_pipeline(X_train, y_train)
    base_artifacts, _ = fit_baseline_pipeline(X_train, y_train)

    # Completely alien category
    alien_record = {
        "PatientGender": "NonBinaryAlienCategory",
        "ProviderSpecialty": "CosmicSurgery",
        "PatientMaritalStatus": "Interdimensional",
        "PatientEmploymentStatus": "TimeTraveler",
        "ClaimType": "QuantumTeleportation",
        "ClaimSubmissionMethod": "Telepathy",
        "ClaimAmount": 7777.0,
        "PatientAge": 42,
        "PatientIncome": 55555.0,
        "ClaimDate": "2024-06-15",
    }
    input_df = prepare_claim_inference_record(alien_record)

    # Word2Vec transformation should not crash
    w2v_features, zero_cnt = transform_word2vec_features(input_df, w2v_artifacts)
    pred_w2v = w2v_artifacts["classifier"].predict(w2v_features)
    assert pred_w2v[0] in [0, 1]

    # Baseline transformation should not crash
    base_features = transform_baseline_features(input_df, base_artifacts)
    pred_base = base_artifacts["classifier"].predict(base_features)
    assert pred_base[0] in [0, 1]
