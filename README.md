# Credit Card Fraud Detection

End-to-end machine learning project to detect fraudulent credit card transactions using multiple classifiers with class imbalance handling.

## Project Overview

- **Goal:** Identify fraudulent transactions from highly imbalanced data (0.17% fraud rate)
- **Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 30 features
- **Approach:** SMOTE-based oversampling + 6 classifiers compared, best model hyperparameter-tuned

## Models Compared

| Model | Status |
|-------|--------|
| Logistic Regression | Baseline |
| Random Forest | Ensemble |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| CatBoost | Gradient Boosting |
| SVM | Max-margin classifier |

## Key Results

The notebook automatically selects and tunes the best model. Expected performance:

- **F1-Score:** ~0.85-0.90 (fraud class)
- **ROC-AUC:** ~0.97-0.99
- Gradient boosting models consistently outperform linear models and random forests on this dataset.

## Notebook Structure

1. **Imports & Setup**
2. **Data Loading & Cleaning** — missing values, duplicates, class distribution
3. **Exploratory Data Analysis** — distributions, correlations, PCA feature analysis
4. **Preprocessing** — RobustScaler, train/test split
5. **Class Imbalance** — SMOTE oversampling
6. **Model Training** — 6 classifiers trained and compared
7. **Hyperparameter Tuning** — GridSearchCV on best model
8. **Evaluation** — Confusion matrix, ROC-AUC, precision-recall, classification report
9. **Model Saving** — Joblib persistence
10. **Summary**

## Requirements
