
# Credit Card Fraud Detection

## Overview
This project uses XGBoost to classify fraudulent transactions in a highly imbalanced, anonymized credit-card dataset. The emphasis is on precision, recall, and F1-score rather than accuracy alone: a model predicting nearly every transaction as legitimate could appear accurate while missing fraud.

## Operational Tradeoffs

| Error | Why it matters |
|---|---|
| False negative | Fraud goes undetected and may lead to financial losses. |
| False positive | A legitimate transaction may be flagged, adding review work or customer friction. |

Choosing a classification threshold depends on the relative costs of these errors. **Threshold tuning and cost-sensitive evaluation are future work**, not results established by this project.

## Problem
The task is to classify transactions as fraudulent or legitimate based on anonymized transaction features. This is a binary classification problem with severe class imbalance.

## Data
Public credit card transaction dataset released by Université Libre de Bruxelles (ULB) and distributed via Kaggle. The data is fully anonymized and highly imbalanced.

## Approach
- Exploratory data analysis to understand class imbalance
- Supervised learning using an XGBoost classifier
- Evaluation using precision, recall, and F1-score

## Results
The final model achieved strong performance on the test set, effectively identifying fraudulent transactions while maintaining generalization across cross-validation folds.

## Limitations & Next Steps
- Limited time window of transactions
- Anonymized features restrict interpretability
- Future work includes threshold tuning and cost-sensitive learning

## How to Run
Install dependencies using `pip install -r requirements.txt` and run the notebook located in the `notebooks/` directory.


