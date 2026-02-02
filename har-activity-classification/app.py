import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# Resolve base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# App title and description
st.set_page_config(
    page_title="Human Activity Recognition",
    layout="wide"
)

st.title("Human Activity Recognition - Multi-Class Classification")
st.write(
    """
    This application demonstrates multiple machine learning classification models
    trained on the Human Activity Recognition (HAR) dataset.
    
    Upload a test dataset, select a model, and view performance metrics
    along with the confusion matrix.
    """
)

# =========================
# Load models and preprocessors
# =========================

@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
        "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl")),
        "K-Nearest Neighbors": joblib.load(os.path.join(MODEL_DIR, "knn.pkl")),
        "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl")),
        "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
        "XGBoost": joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl")),
    }

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    return models, scaler, label_encoder


models, scaler, label_encoder = load_artifacts()

st.success("Models and preprocessing objects loaded successfully.")

# =========================
# Download Sample Dataset
# =========================

st.header("Download Sample Test Dataset")

sample_file_path = os.path.join(BASE_DIR, "sample_data", "test.csv")

if os.path.exists(sample_file_path):
    with open(sample_file_path, "rb") as f:
        st.download_button(
            label="Download Sample Test CSV",
            data=f,
            file_name="test.csv",
            mime="text/csv"
        )
else:
    st.warning("Sample dataset not available.")


# =========================
# CSV Upload & Model Selection
# =========================

st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file (test data only)",
    type=["csv"]
)

st.header("Select Model")

model_name = st.selectbox(
    "Choose a classification model",
    options=list(models.keys())
)

# =========================
# Data Validation & Preprocessing
# =========================

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("CSV file uploaded successfully.")

        st.write("Dataset Preview:")
        st.dataframe(data.head())

        # Identify target column
        if "Activity" not in data.columns:
            st.error("Target column 'Activity' not found in uploaded CSV.")
            st.stop()

        X = data.drop(columns=["Activity"])
        y_true = data["Activity"]

        # Encode target labels
        y_true_encoded = label_encoder.transform(y_true)

        # Scale features
        X_scaled = scaler.transform(X)

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.stop()

# =========================
# Model Prediction & Metrics
# =========================

if uploaded_file is not None:
    st.header("Model Evaluation Results")

    # Get selected model
    selected_model = models[model_name]

    # Predictions
    y_pred = selected_model.predict(X_scaled)

    # Some models provide probabilities (needed for AUC)
    if hasattr(selected_model, "predict_proba"):
        y_pred_proba = selected_model.predict_proba(X_scaled)
        auc_score = roc_auc_score(
            y_true_encoded,
            y_pred_proba,
            multi_class="ovr",
            average="weighted"
        )
    else:
        auc_score = "Not Available"

    # Metrics
    accuracy = accuracy_score(y_true_encoded, y_pred)
    precision = precision_score(y_true_encoded, y_pred, average="weighted")
    recall = recall_score(y_true_encoded, y_pred, average="weighted")
    f1 = f1_score(y_true_encoded, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_true_encoded, y_pred)

    # Display metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy, 4))
    col1.metric("Precision", round(precision, 4))

    col2.metric("Recall", round(recall, 4))
    col2.metric("F1 Score", round(f1, 4))

    col3.metric("MCC", round(mcc, 4))
    col3.metric("AUC", round(auc_score, 4) if auc_score != "Not Available" else auc_score)

# =========================
# Confusion Matrix
# =========================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true_encoded, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"{model_name} - Confusion Matrix")

    st.pyplot(fig)

