# Credit Card Fraud Detection

## Overview
This project applies supervised machine learning to detect fraudulent credit card transactions in a highly imbalanced dataset. The analysis focuses on appropriate evaluation metrics and tradeoffs relevant to real-world fraud detection.

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

