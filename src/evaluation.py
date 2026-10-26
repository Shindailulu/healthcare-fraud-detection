"""Model evaluation metrics, comparison tables, and result interpretations.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_model_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    model_name: str = "Model",
) -> dict[str, Any]:
    """Calculate evaluation metrics from actual test set ground truth and predictions."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    # ROC-AUC computation
    roc_auc_val: float | None = None
    roc_auc_display: str = "Not available for this split"
    roc_auc_note: str = ""
    fpr: list[float] = []
    tpr: list[float] = []

    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) < 2:
        roc_auc_note = "Only one class present in test ground truth."
    elif y_proba is not None:
        try:
            roc_auc_val = float(roc_auc_score(y_true_arr, y_proba))
            roc_auc_display = f"{roc_auc_val:.4f} ({roc_auc_val * 100:.2f}%)"
            curve_fpr, curve_tpr, _ = roc_curve(y_true_arr, y_proba)
            fpr = curve_fpr.tolist()
            tpr = curve_tpr.tolist()
        except Exception as e:
            roc_auc_note = f"ROC-AUC calculation failed: {str(e)}"
    else:
        roc_auc_note = "Probability scores not provided."

    # Confusion matrix with explicit labels [0, 1] (0 = Legitimate, 1 = Fraud)
    # cm layout:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    total_test = len(y_true_arr)
    fraud_count = int((y_true_arr == 1).sum())
    legit_count = int((y_true_arr == 0).sum())

    return {
        "model_name": model_name,
        "total_test_samples": total_test,
        "fraud_cases_count": fraud_count,
        "legitimate_cases_count": legit_count,
        "accuracy": acc,
        "accuracy_pct": f"{acc * 100:.2f}%",
        "precision": prec,
        "precision_pct": f"{prec * 100:.2f}%",
        "recall": rec,
        "recall_pct": f"{rec * 100:.2f}%",
        "f1_score": f1,
        "f1_score_pct": f"{f1 * 100:.2f}%",
        "roc_auc": roc_auc_val,
        "roc_auc_display": roc_auc_display,
        "roc_auc_note": roc_auc_note,
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
    }


def build_comparison_dataframe(
    w2v_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> pd.DataFrame:
    """Build a side-by-side comparison table between Word2Vec and Baseline."""
    data = [
        {
            "Metric": "Accuracy",
            "Word2Vec + LogReg": w2v_metrics["accuracy_pct"],
            "One-Hot Baseline + LogReg": baseline_metrics["accuracy_pct"],
            "Explanation": "Overall proportion of claims correctly classified.",
        },
        {
            "Metric": "Precision (Fraud)",
            "Word2Vec + LogReg": w2v_metrics["precision_pct"],
            "One-Hot Baseline + LogReg": baseline_metrics["precision_pct"],
            "Explanation": "Proportion of claims flagged as fraud that are actually fraudulent.",
        },
        {
            "Metric": "Recall (Fraud)",
            "Word2Vec + LogReg": w2v_metrics["recall_pct"],
            "One-Hot Baseline + LogReg": baseline_metrics["recall_pct"],
            "Explanation": "Proportion of actual fraud claims captured by the screening tool.",
        },
        {
            "Metric": "F1-Score (Fraud)",
            "Word2Vec + LogReg": w2v_metrics["f1_score_pct"],
            "One-Hot Baseline + LogReg": baseline_metrics["f1_score_pct"],
            "Explanation": "Harmonic mean of precision and recall for the fraud class.",
        },
        {
            "Metric": "ROC-AUC",
            "Word2Vec + LogReg": w2v_metrics["roc_auc_display"],
            "One-Hot Baseline + LogReg": baseline_metrics["roc_auc_display"],
            "Explanation": "Discriminative ability across all possible probability thresholds.",
        },
        {
            "Metric": "True Positives (Fraud caught)",
            "Word2Vec + LogReg": str(w2v_metrics["tp"]),
            "One-Hot Baseline + LogReg": str(baseline_metrics["tp"]),
            "Explanation": "Correctly flagged fraudulent claims.",
        },
        {
            "Metric": "False Positives (Legit flagged)",
            "Word2Vec + LogReg": str(w2v_metrics["fp"]),
            "One-Hot Baseline + LogReg": str(baseline_metrics["fp"]),
            "Explanation": "Legitimate claims erroneously flagged for review.",
        },
        {
            "Metric": "False Negatives (Fraud missed)",
            "Word2Vec + LogReg": str(w2v_metrics["fn"]),
            "One-Hot Baseline + LogReg": str(baseline_metrics["fn"]),
            "Explanation": "Fraudulent claims that bypassed screening undetected.",
        },
    ]
    return pd.DataFrame(data)


def generate_interpretation(
    w2v_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    vocab_size: int,
) -> str:
    """Generate honest academic interpretation based on calculated results."""
    w2v_rec = w2v_metrics["recall"]
    w2v_prec = w2v_metrics["precision"]
    base_rec = baseline_metrics["recall"]
    base_prec = baseline_metrics["precision"]

    lines = [
        "### Academic Result Interpretation",
        f"- **Vocabulary Size Warning**: Word2Vec vocabulary contains **{vocab_size} tokens**. Because categorical features in this dataset have relatively low cardinality, Word2Vec embeddings are learned over a very compact token set. In small vocabularies, semantic relationships are constrained compared to large text corpora.",
        f"- **Precision-Recall Trade-off**: The Word2Vec model achieved a recall of **{w2v_metrics['recall_pct']}** with precision of **{w2v_metrics['precision_pct']}**, catching **{w2v_metrics['tp']}** of **{w2v_metrics['fraud_cases_count']}** fraudulent claims while producing **{w2v_metrics['fp']}** false positive reviews. The baseline one-hot model achieved **{baseline_metrics['recall_pct']}** recall and **{baseline_metrics['precision_pct']}** precision.",
    ]

    diff_recall = w2v_rec - base_rec
    diff_prec = w2v_prec - base_prec

    if abs(diff_recall) < 0.03 and abs(diff_prec) < 0.03:
        lines.append(
            "- **Model Comparison**: Word2Vec embeddings and standard One-Hot Encoding exhibit very similar screening performance on this dataset. Word2Vec does not inherently outperform one-hot representation here, confirming that dense embeddings do not automatically surpass discrete indicators when categorical cardinality is small."
        )
    elif w2v_rec > base_rec:
        lines.append(
            f"- **Model Comparison**: Word2Vec achieved higher recall (+{diff_recall * 100:.2f} percentage points), catching more potential fraud, though one-hot encoding may differ in false-alarm rates."
        )
    else:
        lines.append(
            f"- **Model Comparison**: One-Hot Encoding achieved higher recall (+{(-diff_recall) * 100:.2f} percentage points) or better precision (+{(-diff_prec) * 100:.2f} percentage points). This demonstrates that one-hot encoding remains a robust, transparent baseline."
        )

    lines.append(
        "- **Generalizability Caution**: High laboratory test scores on this benchmark dataset do not guarantee identical performance on real-world insurance streams. Real-world claims exhibit non-stationary fraud patterns, seasonal drift, and operational noise that require continuous human auditing and external validation."
    )

    return "\n\n".join(lines)
