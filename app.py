"""Healthcare Fraud Detection System - Academic Prototype
National Open University of Nigeria (NOUN), Abuja
Intelligent Framework for Fraud Detection in Healthcare Ecosystem Using Word2Vec Embedding Technique
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.baseline_pipeline import (
    fit_baseline_pipeline,
    transform_baseline_features,
)
from src.data_processing import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    TARGET_COL,
    clean_training_dataset,
    load_dataset,
    prepare_claim_inference_record,
    split_data,
)
from src.word2vec_pipeline import (
    fit_word2vec_pipeline,
    transform_word2vec_features,
)
from src.evaluation import (
    build_comparison_dataframe,
    compute_model_metrics,
    generate_interpretation,
)
from src.prediction import (
    load_trained_system,
    save_trained_system,
    screen_batch_claims,
    screen_single_claim,
)
from src.validation import (
    ESSENTIAL_COLUMNS,
    EXCLUDED_COLUMNS_INFO,
    EXPECTED_COLUMNS,
    validate_batch_dataset,
    validate_inference_record,
    validate_training_dataset,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "Health Insurance Fraud Claims.xlsx"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "healthcare_fraud_system.joblib"

st.set_page_config(
    page_title="Healthcare Fraud Detection | NOUN",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom academic styling with restrained orange accent
st.markdown(
    """
    <style>
    .main {
        background-color: #fcfcfc;
    }
    .header-box {
        background: linear-gradient(135deg, #fefbf6 0%, #ffffff 100%);
        border: 1px solid #fed7aa;
        border-left: 5px solid #ea580c;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .header-box h1 {
        color: #1e293b;
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .header-box p {
        color: #475569;
        font-size: 0.95rem;
        margin-bottom: 0;
    }
    .badge-noun {
        display: inline-block;
        background-color: #ea580c;
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .academic-disclaimer {
        background-color: #fff7ed;
        border: 1px solid #ffedd5;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        font-size: 0.88rem;
        color: #9a3412;
        margin: 1rem 0;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ea580c;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 500;
    }
    .flagged-box {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #dc2626;
        border-radius: 6px;
        padding: 1rem;
        color: #991b1b;
    }
    .legit-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 6px;
        padding: 1rem;
        color: #166534;
    }
    /* Responsive styling for Mobile and Tablet */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.75rem 2.5rem 0.75rem !important;
        }
        .header-box {
            padding: 1rem 1rem !important;
            margin-bottom: 1rem !important;
        }
        .header-box h1 {
            font-size: 1.35rem !important;
            line-height: 1.25 !important;
        }
        .header-box p {
            font-size: 0.85rem !important;
            line-height: 1.4 !important;
        }
        .metric-card {
            padding: 0.75rem 0.5rem !important;
            margin-bottom: 0.6rem !important;
        }
        .metric-val {
            font-size: 1.25rem !important;
        }
        .metric-label {
            font-size: 0.75rem !important;
        }
        .stButton button {
            width: 100% !important;
        }
    }
    @media (min-width: 769px) and (max-width: 1024px) {
        .header-box h1 {
            font-size: 1.55rem !important;
        }
        .metric-val {
            font-size: 1.35rem !important;
        }
        .block-container {
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "clean_dataset" not in st.session_state:
        st.session_state.clean_dataset = None
    if "data_audit" not in st.session_state:
        st.session_state.data_audit = None
    if "w2v_artifacts" not in st.session_state:
        st.session_state.w2v_artifacts = None
    if "baseline_artifacts" not in st.session_state:
        st.session_state.baseline_artifacts = None
    if "w2v_metrics" not in st.session_state:
        st.session_state.w2v_metrics = None
    if "baseline_metrics" not in st.session_state:
        st.session_state.baseline_metrics = None
    if "dataset_filename" not in st.session_state:
        st.session_state.dataset_filename = None

    # Auto-load saved model artifacts if present and not yet loaded
    if st.session_state.w2v_artifacts is None and MODEL_FILE.exists():
        bundle = load_trained_system(MODEL_FILE)
        if bundle:
            st.session_state.w2v_artifacts = bundle.get("word2vec_artifacts")
            st.session_state.baseline_artifacts = bundle.get("baseline_artifacts")
            metrics = bundle.get("metrics_summary", {})
            st.session_state.w2v_metrics = metrics.get("w2v")
            st.session_state.baseline_metrics = metrics.get("baseline")

    # Auto-load default dataset if present
    if st.session_state.dataset is None and DEFAULT_DATA_PATH.exists():
        try:
            df = load_dataset(DEFAULT_DATA_PATH)
            is_valid, _, _ = validate_training_dataset(df)
            if is_valid:
                clean_df, audit = clean_training_dataset(df)
                st.session_state.dataset = df
                st.session_state.clean_dataset = clean_df
                st.session_state.data_audit = audit
                st.session_state.dataset_filename = DEFAULT_DATA_PATH.name
        except Exception:
            pass


init_session_state()

# Navigation Sidebar
with st.sidebar:
    st.markdown(
        """
        <div style="padding-bottom: 0.5rem;">
            <span class="badge-noun">Academic Research</span>
            <h3 style="margin-top: 0.2rem; color: #1e293b; font-size: 1.15rem;">Healthcare Claim Screening</h3>
            <p style="color: #64748b; font-size: 0.82rem; margin-bottom: 0.5rem;">National Open University of Nigeria</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    menu = st.radio(
        "Navigation",
        [
            "Home",
            "Dataset Upload & Inspection",
            "Model Training",
            "Evaluation Results",
            "Screen Single Claim",
            "Batch Screening",
            "About & Limitations",
        ],
        index=0,
    )

    st.divider()
    st.markdown("**System Status**")
    if st.session_state.clean_dataset is not None:
        st.caption(f"📊 Dataset: `{st.session_state.dataset_filename}` ({len(st.session_state.clean_dataset):,} rows)")
    else:
        st.caption("📊 Dataset: None loaded")

    if st.session_state.w2v_artifacts is not None:
        st.caption(f"🧠 Models: Fitted (Vocab: {st.session_state.w2v_artifacts.get('vocabulary_size', 0)} tokens)")
    else:
        st.caption("🧠 Models: Not trained yet")

    st.markdown(
        """
        <div class="academic-disclaimer" style="margin-top: 1rem;">
            <strong>Advisory Note</strong><br>
            Screening outputs are experimental indicators, not evidence of fraud. Qualified human adjudication is required.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str):
    """Render consistent header banner."""
    st.markdown(
        f"""
        <div class="header-box">
            <span class="badge-noun">NOUN Research Prototype</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- PAGE A: HOME ----------------
if menu == "Home":
    render_header(
        "Healthcare Fraud Detection System",
        "Intelligent Framework for Fraud Detection in Healthcare Ecosystem Using Word2Vec Embedding Technique",
    )

    st.markdown(
        """
        ### Welcome to the Academic Research Framework
        This application implements an experimental machine-learning methodology developed at the 
        **National Open University of Nigeria (NOUN), Abuja**. The system demonstrates how semantic word embeddings 
        (Gensim Word2Vec) can be learned over high-dimensional categorical claim attributes and fused with scaled numerical 
        features to screen health insurance claims for suspicious billing indicators.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            "**Step 1: Dataset Inspection**\n\n"
            "Upload and audit claim records from the Zenodo Healthcare Benchmark (`Health Insurance Fraud Claims.xlsx`)."
        )
    with col2:
        st.warning(
            "**Step 2: Model Training**\n\n"
            "Train Gensim Word2Vec on training-only tokens, fuse with scaled financials, and fit balanced Logistic Regression."
        )
    with col3:
        st.success(
            "**Step 3: Screening & Evaluation**\n\n"
            "Review comparative metrics against a One-Hot baseline and screen interactive individual or batch claims."
        )

    st.divider()

    st.subheader("Key Workflow Capabilities")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.markdown(
            """
            - **Strict Anti-Leakage Separation**: Features derived from dates, scalers, and Word2Vec models are fitted exclusively on training splits.
            - **Honest Comparative Baseline**: Evaluates Word2Vec against standard One-Hot Encoding without assuming superiority.
            - **Realistic Vocabulary Auditing**: Discloses vocabulary constraints and token coverage rather than treating representations as a black box.
            """
        )
    with fcol2:
        st.markdown(
            """
            - **Interactive Claim Screening**: Form-based input with instant screening probabilities and safe unknown category handling.
            - **Batch Ingestion**: High-throughput assessment for sets of incoming claims with downloadable review exports.
            - **Academic Integrity**: All metrics, confusion matrices, and ROC values are dynamically computed from held-out data.
            """
        )

    st.markdown(
        """
        <div class="academic-disclaimer">
            <strong>Important Research Prototype Disclaimer</strong><br>
            This software is designed solely for academic demonstration and research purposes at the National Open University of Nigeria.
            Predictions generated by these algorithms are preliminary screening signals and do not constitute legal or clinical proof of fraud.
            Human review by an authorised medical auditor is mandatory prior to any financial adjudication or administrative action.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- PAGE B: DATASET UPLOAD & INSPECTION ----------------
elif menu == "Dataset Upload & Inspection":
    render_header(
        "Dataset Upload & Schema Inspection",
        "Load Zenodo Healthcare Fraud benchmark dataset (.xlsx or .csv) and verify data consistency.",
    )

    st.markdown(
        """
        You can upload the official **Zenodo Healthcare Fraud Claims** dataset (DOI `10.5281/zenodo.13289814`) 
        or an equivalent dataset in Excel (`.xlsx`) or CSV format.
        """
    )

    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file",
        type=["xlsx", "csv"],
        help="Upload 'Health Insurance Fraud Claims.xlsx' or equivalent CSV.",
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Parsing and validating uploaded dataset..."):
                raw_df = load_dataset(uploaded_file)
                is_valid, validation_errors, summary = validate_training_dataset(raw_df)

                if not is_valid:
                    st.error("Dataset validation failed. Please rectify the following issues:")
                    for err in validation_errors:
                        st.markdown(f"- ❌ {err}")
                else:
                    clean_df, audit = clean_training_dataset(raw_df)
                    st.session_state.dataset = raw_df
                    st.session_state.clean_dataset = clean_df
                    st.session_state.data_audit = audit
                    st.session_state.dataset_filename = uploaded_file.name
                    st.success(f"Successfully loaded and validated `{uploaded_file.name}`.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        clean_df = st.session_state.clean_dataset
        audit = st.session_state.data_audit

        st.subheader("Dataset Summary & Health Check")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Records", f"{len(df):,}")
        with c2:
            st.metric("Features / Columns", f"{len(df.columns)}")
        with c3:
            st.metric("Duplicate Rows", f"{audit['initial_duplicates']:,}")
        with c4:
            st.metric("Invalid Dates", f"{audit['invalid_dates']}")

        st.divider()

        tab1, tab2, tab3 = st.tabs(["Data Preview (First 10 Rows)", "Target Distribution", "Schema & Missingness"])

        with tab1:
            st.dataframe(df.head(10), use_container_width=True)

        with tab2:
            tc1, tc2 = st.columns([1, 1])
            with tc1:
                st.markdown("**Claim Legitimacy Class Balance**")
                fraud_cnt = audit["fraud_count"]
                legit_cnt = audit["legitimate_count"]
                total = max(audit["final_clean_rows"], 1)

                dist_df = pd.DataFrame(
                    [
                        {"Class": "Legitimate (0)", "Count": legit_cnt, "Percentage": f"{(legit_cnt/total)*100:.2f}%"},
                        {"Class": "Fraud (1)", "Count": fraud_cnt, "Percentage": f"{(fraud_cnt/total)*100:.2f}%"},
                    ]
                )
                st.dataframe(dist_df, hide_index=True, use_container_width=True)
                st.caption(f"Imbalance Ratio: Approximately 1 fraud case for every {legit_cnt//max(fraud_cnt, 1)} legitimate claims.")

            with tc2:
                fig, ax = plt.subplots(figsize=(4, 2.5))
                bars = ax.bar(["Legitimate (0)", "Fraud (1)"], [legit_cnt, fraud_cnt], color=["#10b981", "#ea580c"])
                ax.set_ylabel("Count")
                ax.set_title("Target Distribution (Cleaned Data)")
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, yval + 10, f"{yval:,}", ha="center", va="bottom", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with tab3:
            st.markdown("**Column Missing Value Audit**")
            missing_series = df.isna().sum()
            missing_df = pd.DataFrame(
                {
                    "Column": missing_series.index,
                    "Missing Values": missing_series.values,
                    "Missing %": [f"{(v/len(df))*100:.2f}%" for v in missing_series.values],
                    "Status": ["Essential Feature" if c in ESSENTIAL_COLUMNS else "Optional / Excluded" for c in missing_series.index],
                }
            )
            st.dataframe(missing_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No dataset is currently loaded. Please upload a dataset above or place `Health Insurance Fraud Claims.xlsx` in the `data/` directory.")


# ---------------- PAGE C: MODEL TRAINING ----------------
elif menu == "Model Training":
    render_header(
        "Model Training & Architecture",
        "Train Word2Vec embeddings and Logistic Regression, alongside a One-Hot encoded baseline.",
    )

    if st.session_state.clean_dataset is None:
        st.warning("Please upload and validate a dataset in the 'Dataset Upload & Inspection' page before training.")
        st.stop()

    clean_df = st.session_state.clean_dataset

    st.subheader("Training Configuration & Pipeline Specification")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            - **Target Variable**: `{TARGET_COL}` (`Fraud` = 1, `Legitimate` = 0)
            - **Stratified Train/Test Split**: 80% Training, 20% Held-Out Testing (`seed=42`)
            - **Categorical Fields for Word2Vec**: `{', '.join(CATEGORICAL_COLS)}`
            - **Numerical Fields Scaled**: `{', '.join(NUMERICAL_COLS)}`
            - **Word2Vec Hyperparameters**:
              - Vector Size: `16`
              - Context Window: `5`
              - Min Count: `2`
              - Architecture: `Skip-Gram (sg=1)`
              - Epochs: `100` | Random Seed: `42`
            """
        )
    with c2:
        st.markdown("**Feature Exclusions & Integrity Rationale:**")
        for col_name, reason in EXCLUDED_COLUMNS_INFO.items():
            st.markdown(f"- **`{col_name}`**: {reason}")

    st.divider()

    train_btn = st.button("🚀 Train and Evaluate Models", type="primary", use_container_width=True)

    if train_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Step 1/5: Performing stratified 80/20 train/test split...")
            progress_bar.progress(20)
            X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.20, random_state=42)

            status_text.text("Step 2/5: Fitting Gensim Word2Vec on training records only...")
            progress_bar.progress(40)
            w2v_artifacts, X_train_w2v, train_zeros = fit_word2vec_pipeline(X_train, y_train)

            status_text.text("Step 3/5: Transforming test split and training One-Hot baseline...")
            progress_bar.progress(60)
            X_test_w2v, test_zeros = transform_word2vec_features(X_test, w2v_artifacts)
            base_artifacts, X_train_base = fit_baseline_pipeline(X_train, y_train)
            X_test_base = transform_baseline_features(X_test, base_artifacts)

            status_text.text("Step 4/5: Computing dynamic evaluation metrics...")
            progress_bar.progress(80)
            # Evaluate Word2Vec model
            w2v_preds = w2v_artifacts["classifier"].predict(X_test_w2v)
            w2v_probas = w2v_artifacts["classifier"].predict_proba(X_test_w2v)[:, 1]
            w2v_metrics = compute_model_metrics(y_test, w2v_preds, w2v_probas, model_name="Word2Vec")

            # Evaluate Baseline model
            base_preds = base_artifacts["classifier"].predict(X_test_base)
            base_probas = base_artifacts["classifier"].predict_proba(X_test_base)[:, 1]
            base_metrics = compute_model_metrics(y_test, base_preds, base_probas, model_name="Baseline")

            status_text.text("Step 5/5: Persisting model artifacts to disk...")
            progress_bar.progress(100)
            metrics_summary = {"w2v": w2v_metrics, "baseline": base_metrics}
            save_trained_system(MODEL_DIR, w2v_artifacts, base_artifacts, metrics_summary)

            # Store in session state
            st.session_state.w2v_artifacts = w2v_artifacts
            st.session_state.baseline_artifacts = base_artifacts
            st.session_state.w2v_metrics = w2v_metrics
            st.session_state.baseline_metrics = base_metrics

            status_text.empty()
            progress_bar.empty()
            st.success("Training and evaluation completed successfully! Artifacts persisted to `models/`.")

        except Exception as e:
            st.error(f"Training failed: {str(e)}")

    if st.session_state.w2v_artifacts is not None:
        st.subheader("Training Run Summary")
        w_art = st.session_state.w2v_artifacts
        m = st.session_state.w2v_metrics

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Training Set Size", f"{w_art['training_rows']:,}")
        with c2:
            st.metric("Held-Out Test Size", f"{m['total_test_samples']:,}")
        with c3:
            st.metric("Word2Vec Vocabulary", f"{w_art['vocabulary_size']} tokens")
        with c4:
            st.metric("Test Fraud Cases", f"{m['fraud_cases_count']:,}")

        with st.expander("Inspect Learned Word2Vec Vocabulary Tokens"):
            st.write(w_art.get("vocabulary_tokens", []))

        st.caption(
            "⚠️ Research Note: High scores on this test split reflect laboratory benchmark performance. "
            "This model is a research prototype and is not certified for autonomous production deployment without external validation."
        )


# ---------------- PAGE D: EVALUATION RESULTS ----------------
elif menu == "Evaluation Results":
    render_header(
        "Model Evaluation & Comparison",
        "Empirical performance comparison of Word2Vec embeddings vs One-Hot encoded baseline.",
    )

    if st.session_state.w2v_metrics is None or st.session_state.baseline_metrics is None:
        st.warning("No evaluation metrics available yet. Please train the models in the 'Model Training' page.")
        st.stop()

    w_m = st.session_state.w2v_metrics
    b_m = st.session_state.baseline_metrics
    vocab_size = st.session_state.w2v_artifacts.get("vocabulary_size", 0)

    st.subheader("Primary Performance Metrics (Held-Out Test Split)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{w_m['accuracy_pct']}</div>
                <div class="metric-label">Word2Vec Accuracy</div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">Baseline: {b_m['accuracy_pct']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{w_m['recall_pct']}</div>
                <div class="metric-label">Word2Vec Recall (Fraud)</div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">Baseline: {b_m['recall_pct']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{w_m['precision_pct']}</div>
                <div class="metric-label">Word2Vec Precision (Fraud)</div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">Baseline: {b_m['precision_pct']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-val">{w_m['f1_score_pct']}</div>
                <div class="metric-label">Word2Vec F1-Score</div>
                <div style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">Baseline: {b_m['f1_score_pct']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Side-by-Side Model Comparison Table")
    comp_df = build_comparison_dataframe(w_m, b_m)
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # Downloads
    csv_bytes = comp_df.to_csv(index=False).encode("utf-8")
    json_bytes = json.dumps({"word2vec": w_m, "baseline": b_m}, indent=2).encode("utf-8")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button(
            "📥 Download Metrics Comparison (CSV)",
            data=csv_bytes,
            file_name="model_comparison_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dcol2:
        st.download_button(
            "📥 Download Complete Evaluation Metadata (JSON)",
            data=json_bytes,
            file_name="evaluation_metrics.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()

    st.subheader("Confusion Matrices & Discriminative Performance")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.markdown("**Word2Vec + Logistic Regression**")
        fig_w, ax_w = plt.subplots(figsize=(4, 3.2))
        cm_w = np.array(w_m["confusion_matrix"])
        im_w = ax_w.imshow(cm_w, cmap="Blues", interpolation="nearest")
        ax_w.set_xticks([0, 1])
        ax_w.set_yticks([0, 1])
        ax_w.set_xticklabels(["Pred Legitimate (0)", "Pred Fraud (1)"], fontsize=8)
        ax_w.set_yticklabels(["True Legitimate (0)", "True Fraud (1)"], fontsize=8)
        ax_w.set_ylabel("True Ground Truth")
        ax_w.set_xlabel("Model Screening Prediction")

        for i in range(2):
            for j in range(2):
                val = cm_w[i, j]
                color = "white" if val > cm_w.max() / 2 else "black"
                ax_w.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_w)
        plt.close(fig_w)

    with cm_col2:
        st.markdown("**One-Hot Baseline + Logistic Regression**")
        fig_b, ax_b = plt.subplots(figsize=(4, 3.2))
        cm_b = np.array(b_m["confusion_matrix"])
        im_b = ax_b.imshow(cm_b, cmap="Oranges", interpolation="nearest")
        ax_b.set_xticks([0, 1])
        ax_b.set_yticks([0, 1])
        ax_b.set_xticklabels(["Pred Legitimate (0)", "Pred Fraud (1)"], fontsize=8)
        ax_b.set_yticklabels(["True Legitimate (0)", "True Fraud (1)"], fontsize=8)
        ax_b.set_ylabel("True Ground Truth")
        ax_b.set_xlabel("Baseline Screening Prediction")

        for i in range(2):
            for j in range(2):
                val = cm_b[i, j]
                color = "white" if val > cm_b.max() / 2 else "black"
                ax_b.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_b)
        plt.close(fig_b)

    # ROC Curves if available
    if w_m.get("roc_fpr") and b_m.get("roc_fpr"):
        st.divider()
        st.markdown("**Receiver Operating Characteristic (ROC) Curves**")
        fig_roc, ax_roc = plt.subplots(figsize=(7, 3.5))
        ax_roc.plot(
            w_m["roc_fpr"],
            w_m["roc_tpr"],
            label=f"Word2Vec (AUC = {w_m.get('roc_auc', 0.0):.4f})",
            color="#ea580c",
            lw=2,
        )
        ax_roc.plot(
            b_m["roc_fpr"],
            b_m["roc_tpr"],
            label=f"One-Hot Baseline (AUC = {b_m.get('roc_auc', 0.0):.4f})",
            color="#2563eb",
            lw=2,
            linestyle="--",
        )
        ax_roc.plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", label="Random Classifier (AUC = 0.50)")
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
        ax_roc.set_ylabel("True Positive Rate (Recall / Sensitivity)")
        ax_roc.set_title("Held-Out Test ROC Discrimination")
        ax_roc.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    st.divider()
    interpretation = generate_interpretation(w_m, b_m, vocab_size)
    st.markdown(interpretation)


# ---------------- PAGE E: SCREEN A SINGLE CLAIM ----------------
elif menu == "Screen Single Claim":
    render_header(
        "Screen Individual Healthcare Claim",
        "Enter claim attributes to generate a preliminary fraud screening score and recommendation.",
    )

    if st.session_state.w2v_artifacts is None:
        st.warning("Please train or load model artifacts before performing claim screening.")
        st.stop()

    artifacts = st.session_state.w2v_artifacts
    categories = artifacts.get("categories", {})

    st.markdown(
        """
        Complete the claim information below. Dropdown menus provide categories observed during model training, 
        and you can select or enter new values safely.
        """
    )

    with st.form("single_claim_form"):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("##### Financial & Demographics")
            claim_amount = st.number_input(
                "Claim Amount (₦ / $)",
                min_value=0.0,
                max_value=1_000_000.0,
                value=4500.0,
                step=100.0,
                help="Requested claim reimbursement amount.",
            )
            patient_age = st.number_input(
                "Patient Age",
                min_value=0,
                max_value=120,
                value=42,
                step=1,
                help="Age of the patient (0 to 120 years).",
            )
            patient_income = st.number_input(
                "Patient Annual Income",
                min_value=0.0,
                max_value=10_000_000.0,
                value=45000.0,
                step=1000.0,
                help="Annual recorded income of the patient.",
            )
            claim_date = st.date_input(
                "Claim Submission Date",
                value=date.today(),
                help="Date on which the healthcare claim was submitted.",
            )

        with col2:
            st.markdown("##### Clinical & Administrative Categories")
            patient_gender = st.selectbox(
                "Patient Gender",
                options=categories.get("PatientGender", ["Male", "Female", "Unknown"]),
                index=0,
            )
            provider_specialty = st.selectbox(
                "Provider Specialty",
                options=categories.get(
                    "ProviderSpecialty",
                    ["Cardiology", "Internal Medicine", "Oncology", "Unknown"],
                ),
                index=0,
            )
            marital_status = st.selectbox(
                "Patient Marital Status",
                options=categories.get(
                    "PatientMaritalStatus",
                    ["Married", "Single", "Divorced", "Widowed", "Unknown"],
                ),
                index=0,
            )
            employment_status = st.selectbox(
                "Patient Employment Status",
                options=categories.get(
                    "PatientEmploymentStatus",
                    ["Employed", "Retired", "Unemployed", "Self-Employed", "Unknown"],
                ),
                index=0,
            )
            claim_type = st.selectbox(
                "Claim Type",
                options=categories.get("ClaimType", ["Inpatient", "Outpatient", "Unknown"]),
                index=0,
            )
            submission_method = st.selectbox(
                "Claim Submission Method",
                options=categories.get("ClaimSubmissionMethod", ["Electronic", "Paper", "Unknown"]),
                index=0,
            )

        submitted = st.form_submit_button("🔍 Screen Claim", type="primary", use_container_width=True)

    if submitted:
        claim_dict = {
            "ClaimAmount": claim_amount,
            "PatientAge": patient_age,
            "PatientIncome": patient_income,
            "ClaimDate": str(claim_date),
            "PatientGender": patient_gender,
            "ProviderSpecialty": provider_specialty,
            "PatientMaritalStatus": marital_status,
            "PatientEmploymentStatus": employment_status,
            "ClaimType": claim_type,
            "ClaimSubmissionMethod": submission_method,
        }

        is_valid, validation_errors = validate_inference_record(claim_dict)
        if not is_valid:
            st.error("Input validation failed:")
            for err in validation_errors:
                st.markdown(f"- ❌ {err}")
        else:
            result = screen_single_claim(claim_dict, artifacts)

            st.divider()
            st.subheader("Screening Assessment Output")

            r_col1, r_col2 = st.columns([1, 1], gap="large")

            with r_col1:
                if result["predicted_class"] == 1:
                    st.markdown(
                        f"""
                        <div class="flagged-box">
                            <h3 style="margin-top:0; color:#b91c1c;">⚠️ {result['status_wording']}</h3>
                            <p style="margin-bottom:0;">
                                The model identified feature characteristics correlated with potential fraud patterns.
                                <strong>An authorised human auditor must examine the original billing records.</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="legit-box">
                            <h3 style="margin-top:0; color:#15803d;">✅ {result['status_wording']}</h3>
                            <p style="margin-bottom:0;">
                                The claim does not trigger elevated statistical fraud indicators under this model.
                                <em>This screening result does not guarantee that the claim is legally valid.</em>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with r_col2:
                proba = result["fraud_probability"]
                if proba is not None:
                    st.metric("Estimated Fraud Risk Score", f"{proba * 100:.2f}%")
                    st.progress(min(max(proba, 0.0), 1.0))
                    st.caption("Score reflects the calibrated Logistic Regression posterior probability.")
                else:
                    st.info("Probability estimation is not available for this model configuration.")

            st.markdown(
                f"""
                <div class="academic-disclaimer" style="margin-top: 1rem;">
                    <strong>Academic Advisory Notice</strong><br>
                    {result['disclaimer']}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Download screening record
            download_df = pd.DataFrame(
                [
                    {
                        "ClaimAmount": claim_amount,
                        "PatientAge": patient_age,
                        "PatientIncome": patient_income,
                        "ClaimDate": str(claim_date),
                        "PatientGender": patient_gender,
                        "ProviderSpecialty": provider_specialty,
                        "PatientMaritalStatus": marital_status,
                        "PatientEmploymentStatus": employment_status,
                        "ClaimType": claim_type,
                        "ClaimSubmissionMethod": submission_method,
                        "PredictedCategory": result["predicted_label"],
                        "ScreeningStatus": result["status_wording"],
                        "FraudProbability": f"{result['fraud_probability']:.4f}" if result["fraud_probability"] is not None else "N/A",
                        "Timestamp": datetime.now().isoformat(),
                    }
                ]
            )

            st.download_button(
                "📥 Download This Screening Result (CSV)",
                data=download_df.to_csv(index=False).encode("utf-8"),
                file_name=f"claim_screening_{date.today().isoformat()}.csv",
                mime="text/csv",
            )


# ---------------- PAGE F: BATCH SCREENING ----------------
elif menu == "Batch Screening":
    render_header(
        "Batch Healthcare Claims Screening",
        "Ingest a batch file of unlabelled claims to generate screening scores and audit flags.",
    )

    if st.session_state.w2v_artifacts is None:
        st.warning("Please train or load model artifacts before performing batch screening.")
        st.stop()

    artifacts = st.session_state.w2v_artifacts

    st.markdown(
        """
        Upload a batch claims file in Excel (`.xlsx`) or CSV (`.csv`) format containing pending claims.
        The file does **not** need the `ClaimLegitimacy` target column.
        """
    )

    batch_file = st.file_uploader("Upload Batch Claims File", type=["xlsx", "csv"], key="batch_uploader")

    if batch_file is not None:
        try:
            batch_df = load_dataset(batch_file)
            is_valid, batch_errors = validate_batch_dataset(batch_df)

            if not is_valid:
                st.error("Batch validation failed:")
                for err in batch_errors:
                    st.markdown(f"- ❌ {err}")
            else:
                st.info(f"Loaded batch file with {len(batch_df):,} claims. Click below to begin screening.")

                if st.button("⚡ Execute Batch Screening", type="primary", use_container_width=True):
                    with st.spinner("Processing batch claims through Word2Vec pipeline..."):
                        screened_df = screen_batch_claims(batch_df, artifacts)

                        st.success("Batch screening complete!")

                        flagged_count = int((screened_df["PredictedCategory"] == "Fraud").sum())
                        total_claims = len(screened_df)
                        flagged_pct = (flagged_count / max(total_claims, 1)) * 100

                        bc1, bc2, bc3 = st.columns(3)
                        with bc1:
                            st.metric("Total Batch Claims", f"{total_claims:,}")
                        with bc2:
                            st.metric("Flagged for Review", f"{flagged_count:,}")
                        with bc3:
                            st.metric("Flagged Ratio", f"{flagged_pct:.2f}%")

                        st.subheader("Screened Batch Claims Preview")
                        st.dataframe(screened_df.head(20), use_container_width=True)

                        csv_batch = screened_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "📥 Download Full Screened Batch Report (CSV)",
                            data=csv_batch,
                            file_name=f"screened_claims_batch_{date.today().isoformat()}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

        except Exception as e:
            st.error(f"Error processing batch claims file: {str(e)}")


# ---------------- PAGE G: ABOUT & LIMITATIONS ----------------
elif menu == "About & Limitations":
    render_header(
        "About Framework & Methodological Limitations",
        "National Open University of Nigeria (NOUN), Abuja - Department of Computer Science",
    )

    st.markdown(
        """
        ### Research Framework Context
        - **Project Title**: Intelligent Framework for Fraud Detection in Healthcare Ecosystem Using Word2Vec Embedding Technique
        - **Institution**: National Open University of Nigeria (NOUN), Abuja
        - **Objective**: Develop an empirical prototype exploring distributed token representations for categorical health claim records.

        ---

        ### Methodological Overview

        #### 1. Why Word2Vec for Tabular Categorical Fields?
        Traditional tabular fraud detection relies heavily on one-hot encoding, which treats each categorical category 
        (such as `ProviderSpecialty=Cardiology` or `ClaimType=Inpatient`) as completely independent and mutually orthogonal. 
        Word2Vec models each claim record as a "sentence" composed of categorical attribute-value tokens. 
        Through Skip-Gram optimization (`sg=1`), tokens that frequently co-occur in similar claim patterns learn dense geometric representations.

        #### 2. Feature Fusion Architecture
        The framework combines:
        - **Averaged Token Embeddings**: 16-dimensional dense continuous vectors summarizing patient demographics, specialties, and submission channels.
        - **Standardized Financial & Temporal Features**: Scaled claim amounts, patient ages, income, and cyclical date components.
        - **Balanced Logistic Regression**: Linear classification boundary with inverse frequency class weighting (`class_weight="balanced"`) to counteract severe minority fraud imbalance.

        #### 3. Baseline Comparison
        The prototype implements an equivalent baseline pipeline using standard One-Hot Encoding and identical class balancing. 
        As demonstrated in empirical results, Word2Vec and One-Hot encoding perform comparably on low-cardinality vocabularies, 
        emphasizing the importance of empirical validation rather than assuming neural embeddings automatically dominate discrete methods.

        ---

        ### Academic Limitations & Ethical Safeguards

        1. **Small Vocabulary Constraints**:
           In this benchmark dataset, categorical fields produce approximately 22 unique tokens. Word2Vec embeddings trained over 
           small vocabularies capture limited contextual co-occurrence compared to natural language corpora with millions of terms.

        2. **Laboratory vs Real-World Generalization**:
           High precision and recall achieved on static held-out test splits may not transfer directly to operational healthcare streams. 
           Real-world claims exhibit concept drift, regulatory shifts, evolving adversarial fraud strategies, and regional demographic divergence.

        3. **Data Privacy & Identifier Redaction**:
           Patient identifiers (`PatientID`), claim tracking IDs (`ClaimID`), and provider IDs (`ProviderID`) are explicitly excluded to 
           prevent memorization and adhere to data protection ethics.

        4. **Target Leakage Safeguards**:
           Variables such as `Cluster` and post-adjudication `ClaimStatus` are strictly excluded from feature sets to prevent artificial performance inflation.

        5. **Mandatory Human Review**:
           Model outputs represent statistical screening flags, not legal or clinical proof of fraudulent intent. 
           Human clinical auditors must always evaluate the complete clinical documentation before any claims adjudication or sanctions.
        """
    )
