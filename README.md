# Customer Churn Classification: Predicting Customer Attrition with Machine Learning

## Overview

This project builds and evaluates machine learning classification models to predict customer churn.

The goal is not only to classify customers, but also to evaluate model performance using business-relevant metrics, analyze classification errors, tune decision thresholds, and generate retention recommendations.

## Business Context

Customer churn occurs when a customer stops using a company’s service.

For subscription-based businesses, detecting churn risk early can help prioritize retention campaigns, reduce revenue loss, and improve customer lifetime value.

In churn prediction, accuracy alone is not enough. The business cost of missing a customer who is likely to churn can be higher than the cost of contacting a customer who would have stayed.

## Objectives

- Audit and understand the customer churn dataset.
- Prepare numerical and categorical features.
- Handle missing values and data quality issues.
- Train baseline and machine learning classification models.
- Compare Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Evaluate models using ROC-AUC, PR-AUC, F1-score, precision, recall, and confusion matrix.
- Analyze threshold selection for business decision-making.
- Interpret the main churn drivers.
- Generate business recommendations.

## Project Structure

```text
customer-churn-classification/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── 01_churn_classification.ipynb
├── reports/
│   ├── executive_summary_en.md
│   ├── resumen_ejecutivo_es.md
│   └── figures/
├── src/
│   ├── load_data.py
│   ├── audit_data.py
│   ├── preprocess_data.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── threshold_analysis.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Status

Project in progress.