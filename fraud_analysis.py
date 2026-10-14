"""Reproducible evaluation with a validation-only decision threshold."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, precision_recall_curve)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def load_data(path):
    data = pd.read_csv(path)
    expected = ['Time', *[f'V{i}' for i in range(1, 29)], 'Amount', 'Class']
    if set(data) != set(expected) or data.isna().any().any():
        raise ValueError('Expected complete ULB transaction columns.')
    if set(data.Class.unique()) != {0, 1}:
        raise ValueError('Class must contain both 0 and 1.')
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError('Features must be finite numeric values.')
    # Exact duplicate transactions must not appear in both training and holdout.
    return data.drop_duplicates().reset_index(drop=True)


def select_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(scores)])


def metrics(y, probabilities, threshold):
    prediction = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(threshold=float(threshold), average_precision=float(average_precision_score(y, probabilities)),
        roc_auc=float(roc_auc_score(y, probabilities)), precision=float(precision_score(y, prediction, zero_division=0)),
        recall=float(recall_score(y, prediction, zero_division=0)), f1=float(f1_score(y, prediction, zero_division=0)),
        true_negative=int(tn), false_positive=int(fp), false_negative=int(fn), true_positive=int(tp))


def run(path, output='results'):
    path, output = Path(path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(path)
    data = load_data(path)
    X, y = data.drop(columns='Class'), data.Class
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=42)
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.1,
        scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
        tree_method='hist', n_jobs=4, random_state=42, eval_metric='logloss')
    cv = cross_validate(model, X_train, y_train, cv=StratifiedKFold(3, shuffle=True, random_state=42),
        scoring={'ap': 'average_precision', 'recall': 'recall', 'f1': 'f1'}, n_jobs=1)
    model.fit(X_train, y_train)
    threshold = select_threshold(y_val, model.predict_proba(X_val)[:, 1])
    # Keep this fitted model: refitting would change the score distribution used to choose the threshold.
    models = {'XGBoost': model, 'Logistic regression': make_pipeline(StandardScaler(),
        LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)),
        'Prior baseline': DummyClassifier(strategy='prior')}
    rows = []
    for name, estimator in models.items():
        if name != 'XGBoost':
            estimator.fit(X_train, y_train)
        probability = estimator.predict_proba(X_test)[:, 1]
        rows.append({'model': name, 'operating_point': 'default', **metrics(y_test, probability, .5)})
        if name == 'XGBoost':
            for value in [.1, .25, .75, threshold]:
                rows.append({'model': name, 'operating_point': 'validation F1' if value == threshold else 'illustrative',
                    **metrics(y_test, probability, value)})
            pd.DataFrame({'actual': y_test, 'probability': probability}).to_csv(output/'holdout_predictions.csv', index=False)
    pd.DataFrame(rows).to_csv(output/'metrics.csv', index=False)
    summary = {'input_filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'raw_rows': len(raw), 'duplicate_rows_removed': len(raw)-len(data), 'rows': len(data),
        'fraud_rows': int(y.sum()), 'split_rows': {'train':len(y_train), 'validation':len(y_val), 'test':len(y_test)},
        'split_fraud': {'train':int(y_train.sum()), 'validation':int(y_val.sum()), 'test':int(y_test.sum())},
        'validation_threshold': threshold, 'cv': {k:float(np.mean(v)) for k,v in cv.items() if k.startswith('test_')},
        'versions': versions()}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False))
    return summary


def versions():
    from importlib.metadata import version
    return {name:version(name) for name in ['numpy','pandas','scikit-learn','xgboost']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/creditcard.csv'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    run(args.data, args.output)
