# Human Activity Recognition – Multi-Class Classification

## 1. Problem Statement
The objective of this project is to build and evaluate multiple machine learning
classification models for recognizing human activities using smartphone sensor data.
The project also demonstrates model deployment through an interactive Streamlit web
application.

The goal is to compare the performance of different classification algorithms using
standard evaluation metrics and provide an end-to-end machine learning workflow from
model training to deployment.

---

## 2. Dataset Description
- **Dataset Name:** Human Activity Recognition Using Smartphones
- **Source:** Kaggle (UCI HAR Dataset)
- **Problem Type:** Multi-class classification
- **Number of Classes:** 6
  - WALKING
  - WALKING_UPSTAIRS
  - WALKING_DOWNSTAIRS
  - SITTING
  - STANDING
  - LAYING
- **Number of Features:** 561
- **Data Type:** Numeric, feature-engineered sensor data

The dataset contains preprocessed features extracted from accelerometer and gyroscope
signals collected from smartphones worn by participants.

---

## 3. Models Used and Evaluation Metrics

The following machine learning models were implemented on the same dataset:

- Logistic Regression  
- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- Naive Bayes (Gaussian)  
- Random Forest (Ensemble Model)  
- XGBoost (Ensemble Model)

### Evaluation Metrics
Each model was evaluated using the following metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | High | High | High | High | High | High |
| Decision Tree | Moderate | Moderate | Moderate | Moderate | Moderate | Moderate |
| KNN | High | High | High | High | High | High |
| Naive Bayes | Lower | Moderate | Lower | Lower | Lower | Lower |
| Random Forest | High | High | High | High | High | High |
| XGBoost | High | High | High | High | High | High |

*(Exact metric values are computed dynamically in the notebook and Streamlit app.)*

---

## 4. Observations on Model Performance

| Model | Observation |
|------|-------------|
| Logistic Regression | Performs consistently well with high accuracy and strong generalization due to linear decision boundaries. |
| Decision Tree | Shows lower accuracy due to overfitting and difficulty in generalizing similar activities. |
| KNN | Achieves high accuracy but is computationally expensive and sensitive to feature scaling. |
| Naive Bayes | Performs comparatively lower due to the strong independence assumption between features. |
| Random Forest | Improves performance over a single decision tree by reducing overfitting through ensembling. |
| XGBoost | Provides strong and stable performance by leveraging gradient boosting and ensemble learning. |

---

## 5. Streamlit Web Application
The trained models are deployed using **Streamlit Community Cloud**.

### Application Features
- Upload test dataset (CSV)
- Select classification model
- Display evaluation metrics
- Visualize confusion matrix

---

## 6. Repository Structure
machine-learning/har-activity-classification/
│── app.py
│── requirements.txt
│── README.md
│
└── models/
├── logistic_regression.pkl
├── decision_tree.pkl
├── knn.pkl
├── naive_bayes.pkl
├── random_forest.pkl
├── xgboost.pkl
├── scaler.pkl
└── label_encoder.pkl


---

## 7. Deployment
The application is deployed on **Streamlit Community Cloud** using the `app.py` file
from this repository. The live application link is shared as part of the assignment
submission.
