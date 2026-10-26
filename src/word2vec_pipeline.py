"""Word2Vec embedding pipeline and Logistic Regression classifier.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data_processing import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    make_sentences,
)


def train_word2vec_embeddings(
    sentences: list[list[str]],
    vector_size: int = 16,
    window: int = 5,
    min_count: int = 2,
    workers: int = 1,
    sg: int = 1,
    seed: int = 42,
    epochs: int = 100,
) -> Word2Vec:
    """Train Gensim Word2Vec on categorical token sentences from training records only."""
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        seed=seed,
        epochs=epochs,
    )


def embed_sentences_with_stats(
    sentences: list[list[str]],
    w2v_model: Word2Vec,
) -> tuple[np.ndarray, int]:
    """Average word vectors for tokens in each claim sentence.
    
    If no token is found in the vocabulary, uses a zero vector.
    Returns:
        (embeddings_matrix, zero_vector_count)
    """
    rows: list[np.ndarray] = []
    zero_count = 0
    vector_size = w2v_model.vector_size

    for sentence in sentences:
        vectors = [w2v_model.wv[token] for token in sentence if token in w2v_model.wv]
        if vectors:
            rows.append(np.mean(vectors, axis=0))
        else:
            rows.append(np.zeros(vector_size))
            zero_count += 1

    return np.vstack(rows), zero_count


def fit_word2vec_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    vector_size: int = 16,
    window: int = 5,
    min_count: int = 2,
    workers: int = 1,
    sg: int = 1,
    seed: int = 42,
    epochs: int = 100,
    random_state: int = 42,
) -> tuple[dict[str, Any], np.ndarray, int]:
    """Fit Word2Vec model, StandardScaler, and LogisticRegression on training data only.
    
    Returns:
        (pipeline_artifacts_dict, X_train_transformed, train_zero_vector_count)
    """
    # 1. Sentences from training data only
    train_sentences = make_sentences(X_train)
    w2v_model = train_word2vec_embeddings(
        train_sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        seed=seed,
        epochs=epochs,
    )

    # 2. Embed training sentences
    train_embeddings, train_zero_count = embed_sentences_with_stats(train_sentences, w2v_model)

    # 3. Fit scaler on numerical training features only
    scaler = StandardScaler()
    train_numerical_scaled = scaler.fit_transform(X_train[NUMERICAL_COLS])

    # 4. Combine embeddings + scaled numerical features
    X_train_combined = np.hstack([train_embeddings, train_numerical_scaled])

    # 5. Train classifier with balanced class weights
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=random_state,
    )
    classifier.fit(X_train_combined, y_train)

    # Extract distinct categories from training set for dropdowns
    categories = {
        col: sorted(X_train[col].dropna().astype(str).unique().tolist())
        for col in CATEGORICAL_COLS
    }

    artifacts = {
        "word2vec": w2v_model,
        "scaler": scaler,
        "classifier": classifier,
        "categorical_columns": CATEGORICAL_COLS,
        "numerical_columns": NUMERICAL_COLS,
        "categories": categories,
        "vector_size": vector_size,
        "vocabulary_size": len(w2v_model.wv),
        "vocabulary_tokens": sorted(list(w2v_model.wv.key_to_index.keys())),
        "training_rows": len(X_train),
    }

    return artifacts, X_train_combined, train_zero_count


def transform_word2vec_features(
    X_df: pd.DataFrame,
    artifacts: dict[str, Any],
) -> tuple[np.ndarray, int]:
    """Transform an arbitrary dataframe of claims using fitted Word2Vec artifacts.
    
    Returns:
        (features_matrix, zero_vector_count)
    """
    w2v_model = artifacts["word2vec"]
    scaler = artifacts["scaler"]
    cat_cols = artifacts["categorical_columns"]
    num_cols = artifacts["numerical_columns"]

    sentences = make_sentences(X_df, cat_cols=cat_cols)
    embeddings, zero_count = embed_sentences_with_stats(sentences, w2v_model)

    # Handle missing numerical features with 0 imputation if any
    num_data = X_df[num_cols].fillna(0)
    numerical_scaled = scaler.transform(num_data)

    combined = np.hstack([embeddings, numerical_scaled])
    return combined, zero_count
