# Executive Summary — Customer Churn Classification

## 1. Objective

This project builds and evaluates machine learning classification models to predict customer churn.

The goal is not only to classify customers, but also to evaluate churn risk using business-relevant metrics, analyze decision thresholds, and generate retention recommendations.

## 2. Business Context

Customer churn occurs when customers stop using a company’s service.

For subscription-based businesses, churn prediction can help prioritize retention campaigns, reduce revenue loss, and improve customer lifetime value.

In this context, accuracy alone is not enough. A model can achieve high accuracy by predicting the majority class, while failing to identify customers who are likely to churn.

The business problem requires balancing:

- capturing as many churners as possible,
- avoiding excessive false positives,
- matching the threshold to the retention team’s capacity.

## 3. Dataset Scope

The project uses the Telco Customer Churn dataset.

The dataset contains:

| Dataset | Rows | Columns |
|---|---:|---:|
| Raw data | 7,043 | 21 |
| Clean data | 7,043 | 22 |
| Train split | 5,634 | 22 |
| Validation split | 1,409 | 22 |

The target variable is `Churn`.

The binary modeling target is `ChurnLabel`, where:

| Label | Meaning |
|---:|---|
| 0 | No churn |
| 1 | Churn |

## 4. Target Distribution

The churn distribution shows moderate class imbalance:

| Class | Share |
|---|---:|
| No churn | 73.46% |
| Churn | 26.54% |

A naive model that always predicts “No churn” would achieve 73.46% accuracy while detecting zero churners.

For this reason, model evaluation focuses on precision, recall, F1-score, ROC-AUC, Average Precision, confusion matrices, and threshold analysis.

## 5. Data Quality Findings

The dataset has no duplicate rows and no duplicate customer IDs.

The main data quality issue is `TotalCharges`, which is stored as text and contains 11 blank values.

All blank `TotalCharges` records have `tenure = 0`, which indicates newly registered customers with no accumulated charges.

The chosen cleaning rule is:

```text
Blank TotalCharges values with tenure = 0 → TotalCharges = 0
```

This preserves the business meaning of the data instead of imputing values with mean or median.

## 6. Initial Business Signals

The audit revealed strong churn patterns:

| Segment | Churn Rate |
|---|---:|
| Month-to-month contract | 42.71% |
| Two-year contract | 2.83% |
| Fiber optic internet service | 41.89% |
| Electronic check payment | 45.29% |
| No OnlineSecurity | 41.77% |
| No TechSupport | 41.64% |
| Paperless billing enabled | 33.57% |

These patterns suggest that churn is associated with contract type, service type, support/security add-ons, and payment behavior.

## 7. Models Compared

The following models were evaluated:

| Model | Purpose |
|---|---|
| Most frequent baseline | Minimum benchmark |
| Logistic Regression | Interpretable linear classifier |
| Decision Tree | Simple nonlinear classifier |
| Random Forest | Ensemble classifier |
| Gradient Boosting | Boosted ensemble classifier |

## 8. Validation Results at Default Threshold 0.50

At the default threshold of 0.50, Gradient Boosting achieved the strongest ranking performance:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8048 | 0.6689 | 0.5241 | 0.5877 | 0.8442 | 0.6631 |
| Random Forest | 0.7615 | 0.5354 | 0.7674 | 0.6308 | 0.8433 | 0.6507 |
| Logistic Regression | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8416 | 0.6327 |
| Decision Tree | 0.7452 | 0.5131 | 0.7861 | 0.6209 | 0.8332 | 0.6230 |
| Baseline | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2654 |

The baseline accuracy is misleading because it detects no churners.

## 9. Cross-Validation Results

Cross-validation confirmed that Gradient Boosting is the strongest model for ranking churn risk.

| Model | Mean Accuracy | Mean ROC-AUC | Mean Average Precision |
|---|---:|---:|---:|
| Gradient Boosting | 0.8046 | 0.8471 | 0.6616 |
| Random Forest | 0.7657 | 0.8465 | 0.6576 |
| Logistic Regression | 0.7456 | 0.8450 | 0.6555 |
| Decision Tree | 0.7292 | 0.8306 | 0.6218 |
| Baseline | 0.7346 | 0.5000 | 0.2654 |

Gradient Boosting was selected as the final predictive model because it achieved the best ROC-AUC and Average Precision under cross-validation.

## 10. Threshold Analysis

The default threshold of 0.50 is not ideal for retention.

| Threshold Strategy | Threshold | Precision | Recall | Customers Flagged | Churners Captured | Churners Missed |
|---|---:|---:|---:|---:|---:|---:|
| Default | 0.50 | 0.6689 | 0.5241 | 293 | 196 | 178 |
| Balanced F1 | 0.24 | 0.5143 | 0.8182 | 595 | 306 | 68 |
| Retention Recall | 0.29 | 0.5306 | 0.7647 | 539 | 286 | 88 |
| Efficient Precision | 0.55 | 0.7046 | 0.4465 | 237 | 167 | 207 |

The recommended threshold for this project is **0.24**, because it provides the best balance between precision and recall while substantially reducing missed churners.

Compared with threshold 0.50, threshold 0.24:

- captures 110 additional churners,
- reduces missed churners from 178 to 68,
- increases customers flagged from 293 to 595,
- increases false positives from 97 to 289.

This is a business trade-off, not a purely technical decision.

## 11. Final Model Decision

The final model is:

```text
Gradient Boosting Classifier
```

The recommended decision threshold is:

```text
0.24
```

This configuration is appropriate when the business wants to prioritize churn detection and can support a larger retention outreach list.

If the retention team has limited capacity, a higher threshold such as 0.55 can be used to prioritize fewer customers with higher churn probability.

## 12. Model Interpretation

Gradient Boosting feature importance shows that the most predictive features include:

- month-to-month contract,
- tenure,
- TotalCharges,
- fiber optic internet service,
- MonthlyCharges,
- no OnlineSecurity,
- no TechSupport,
- electronic check payment.

Logistic Regression coefficients suggest positive churn associations with:

- fiber optic internet service,
- month-to-month contract,
- higher TotalCharges,
- streaming services,
- electronic check payment,
- no OnlineSecurity,
- no TechSupport.

Negative churn associations include:

- longer tenure,
- two-year contract,
- DSL service,
- no internet service,
- no paperless billing,
- having dependents.

These findings should be interpreted as associations, not causal effects.

## 13. Business Recommendations

1. Use the model to prioritize retention outreach, not as an automatic decision system.
2. Use threshold 0.24 when the business wants to maximize churn detection.
3. Use a higher threshold when retention capacity is limited.
4. Focus retention campaigns on month-to-month customers with short tenure and high-risk service patterns.
5. Investigate why fiber optic customers show higher churn.
6. Improve support and security offerings because lack of OnlineSecurity and TechSupport is strongly associated with churn risk.
7. Track campaign cost, retention success rate, and customer lifetime value to refine threshold selection.

## 14. Limitations

- The dataset is static and does not include customer behavior over time.
- The model does not include support tickets, complaints, customer satisfaction, competitor pricing, or retention campaign history.
- Threshold selection should be adjusted using real business costs and retention capacity.
- Feature interpretation is associative, not causal.
- The model estimates churn probability, but it does not explain why an individual customer will churn.

## 15. Next Steps

- Add cost-sensitive threshold optimization using customer lifetime value and retention campaign cost.
- Compare performance with calibrated probabilities.
- Add explainability with SHAP in a future version.
- Build a dashboard for retention teams.
- Deploy a churn scoring API in a later production-focused project.
