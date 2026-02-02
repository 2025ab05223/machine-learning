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
