# Credit card fraud detection

Classify anonymized transactions with a class-weighted XGBoost model, compare a logistic baseline, and measure fraud precision/recall at explicit operating thresholds.

## Data and reproducibility
Source: [ULB credit-card dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
This run used the supplied `creditcard.csv`: 284,807 raw transactions. Removing 1,081 exact duplicates leaves 283,726 rows and 473 frauds. Duplicate removal happens before splitting so identical records do not cross partitions.
The input SHA-256 and library versions are recorded in [results/summary.json](results/summary.json). The 150 MB source CSV is kept locally rather than committed.

Python 3.12 setup:
```bash
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -r requirements-dev.txt
python fraud_analysis.py --data data/creditcard.csv
python -m pytest tests
```
Create `data/` and put the downloaded/supplied CSV there, or pass any local path to `--data`. Open `notebooks/credit_card_fraud_detection.ipynb` to view saved results and precision–recall curves. Set RECOMPUTE=True to retrain.

## Evaluation
A stratified 60/20/20 train/validation/test split uses random_state=42. Three-fold stratified training CV scores average precision, recall and F1. The XGBoost threshold maximizes validation F1. The fitted model is retained after threshold selection so the operating score distribution does not change. The test set never selects the threshold.

| model | operating_point | threshold | average_precision | precision | recall | f1 | false_positive | false_negative |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | default | 0.5000 | 0.8180 | 0.8132 | 0.7789 | 0.7957 | 17 | 21 |
| XGBoost | illustrative | 0.1000 | 0.8180 | 0.3922 | 0.8421 | 0.5351 | 124 | 15 |
| XGBoost | illustrative | 0.2500 | 0.8180 | 0.6290 | 0.8211 | 0.7123 | 46 | 17 |
| XGBoost | illustrative | 0.7500 | 0.8180 | 0.9241 | 0.7684 | 0.8391 | 6 | 22 |
| XGBoost | validation F1 | 0.9464 | 0.8180 | 0.9589 | 0.7368 | 0.8333 | 3 | 25 |
| Logistic regression | default | 0.5000 | 0.6865 | 0.0508 | 0.8737 | 0.0961 | 1550 | 12 |
| Prior baseline | default | 0.5000 | 0.0017 | 0.0000 | 0.0000 | 0.0000 | 0 | 95 |

Average precision summarizes the precision–recall curve. The prior baseline AP equals the holdout fraud prevalence. Accuracy alone would obscure the minority class. At the validation-selected threshold, the model trades recall for fewer false alarms. The other test thresholds illustrate that tradeoff; they do not replace validation-based selection. Monetary loss and review costs were not supplied, so no business-optimal threshold is claimed.

## Limits
The corpus spans two days. Random splitting does not validate future-time deployment, repeated-customer independence or new fraud patterns. Only 95 fraud cases occur in the holdout, so small changes can materially affect recall. The old mixed notebook outputs are superseded by this reproducible run.
