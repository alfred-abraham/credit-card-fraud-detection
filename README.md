# Credit Card Fraud Detection

## Overview

A supervised machine learning project that detects fraudulent credit card transactions in a highly imbalanced dataset.

## Objective

Classify transactions as fraudulent or legitimate while balancing missed fraud against false alarms. Precision, recall, and F1-score provide more useful context than accuracy alone for the minority fraud class.

## Data

The anonymized credit card transaction dataset was released by Université Libre de Bruxelles (ULB) and distributed through Kaggle. The source CSV is not included in this repository.

## Approach

1. Explore transaction features and the imbalance between legitimate and fraudulent transactions.
2. Train an XGBoost classifier.
3. Evaluate fraud-class precision, recall, and F1-score on the test set.

The complete analysis is in [the project notebook](notebooks/credit_card_fraud_detection.ipynb).

## Results

| Fraud-class metric | Reported test result |
| --- | ---: |
| Precision | 88% |
| Recall | 85% |
| F1-score | 86% |

These results show a balance between detecting fraud and limiting false positives within the evaluated dataset.

## Tools

Python, XGBoost, and Jupyter Notebook. Python dependencies are listed in [requirements.txt](requirements.txt).

## How to Run

1. Clone this repository and enter its directory.
2. Install the project dependencies and Jupyter:

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install notebook
   python -m notebook
   ```

3. Obtain the ULB credit card dataset from Kaggle.
4. Open [`notebooks/credit_card_fraud_detection.ipynb`](notebooks/credit_card_fraud_detection.ipynb). The notebook reads `/content/creditcard.csv`; update that path to your local CSV location.
5. Run the cells from top to bottom. Alternatively, open the notebook in Google Colab and upload the CSV to `/content/`.

## Limitations and Next Steps

- The data covers a limited transaction time window.
- Anonymized features restrict interpretation of individual predictions.
- Future work includes decision-threshold tuning and cost-sensitive learning.

## License

See [LICENSE](LICENSE) for repository licensing. Refer to the dataset source for its usage terms.
