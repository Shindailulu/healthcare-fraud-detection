"""CLI script to train and evaluate Word2Vec and Baseline models on healthcare claims.

Usage:
    python train_model.py --data "data/Health Insurance Fraud Claims.xlsx"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.baseline_pipeline import (
    fit_baseline_pipeline,
    transform_baseline_features,
)
from src.data_processing import (
    clean_training_dataset,
    load_dataset,
    split_data,
)
from src.evaluation import (
    build_comparison_dataframe,
    compute_model_metrics,
    generate_interpretation,
)
from src.prediction import save_trained_system
from src.validation import validate_training_dataset
from src.word2vec_pipeline import (
    fit_word2vec_pipeline,
    transform_word2vec_features,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


def main():
    parser = argparse.ArgumentParser(description="Train Healthcare Fraud Detection Models")
    parser.add_argument(
        "--data",
        default="data/Health Insurance Fraud Claims.xlsx",
        help="Path to Excel (.xlsx) or CSV claims dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save trained artifacts and metrics",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    print(f"Loading dataset from: {data_path}...")
    df_raw = load_dataset(data_path)
    is_valid, errors, summary = validate_training_dataset(df_raw)
    if not is_valid:
        print("Dataset validation failed:")
        for err in errors:
            print(f"  - {err}")
        return

    print(f"Cleaning dataset ({len(df_raw)} raw rows)...")
    clean_df, audit = clean_training_dataset(df_raw)
    print(f"Cleaned dataset: {len(clean_df)} valid rows.")
    print(f"  - Fraud claims: {audit['fraud_count']} ({audit['fraud_percentage']}%)")
    print(f"  - Legitimate claims: {audit['legitimate_count']}")

    print("\nPerforming stratified 80/20 train/test split (random_state=42)...")
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.20, random_state=42)
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Held-out test samples: {len(X_test)} (Fraud: {(y_test == 1).sum()}, Legit: {(y_test == 0).sum()})")

    print("\nFitting Word2Vec embeddings (vector_size=16, window=5, min_count=2, epochs=100)...")
    w2v_artifacts, X_train_w2v, train_zeros = fit_word2vec_pipeline(X_train, y_train)
    print(f"  - Word2Vec vocabulary size: {w2v_artifacts['vocabulary_size']} tokens")
    print(f"  - Training zero-vector tokens: {train_zeros}")

    print("\nTransforming test set with Word2Vec pipeline...")
    X_test_w2v, test_zeros = transform_word2vec_features(X_test, w2v_artifacts)

    print("Fitting One-Hot encoded baseline with Logistic Regression...")
    base_artifacts, X_train_base = fit_baseline_pipeline(X_train, y_train)
    X_test_base = transform_baseline_features(X_test, base_artifacts)

    print("\nEvaluating models on held-out test split...")
    w2v_preds = w2v_artifacts["classifier"].predict(X_test_w2v)
    w2v_probas = w2v_artifacts["classifier"].predict_proba(X_test_w2v)[:, 1]
    w2v_metrics = compute_model_metrics(y_test, w2v_preds, w2v_probas, model_name="Word2Vec")

    base_preds = base_artifacts["classifier"].predict(X_test_base)
    base_probas = base_artifacts["classifier"].predict_proba(X_test_base)[:, 1]
    base_metrics = compute_model_metrics(y_test, base_preds, base_probas, model_name="Baseline")

    print("\n" + "=" * 65)
    print("MODEL COMPARISON RESULTS")
    print("=" * 65)
    comp_df = build_comparison_dataframe(w2v_metrics, base_metrics)
    print(comp_df.to_string(index=False))
    print("=" * 65)

    vocab_size = w2v_artifacts["vocabulary_size"]
    interp = generate_interpretation(w2v_metrics, base_metrics, vocab_size)
    print(f"\n{interp}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_summary = {"w2v": w2v_metrics, "baseline": base_metrics}
    saved_file = save_trained_system(out_dir, w2v_artifacts, base_artifacts, metrics_summary)
    metrics_file = out_dir / "evaluation_metrics.json"
    metrics_file.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    print(f"\nTrained system saved to: {saved_file}")
    print(f"Evaluation metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
