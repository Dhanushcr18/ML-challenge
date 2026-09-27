"""Precision-heavy metric and threshold selection."""
import numpy as np
from sklearn.metrics import precision_score, recall_score
from .config import THRESHOLDS


def f05(y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    return (1.25*p*r)/(0.25*p+r) if p+r else 0.0


def tune_threshold(y_true, probabilities, thresholds=THRESHOLDS):
    rows=[]
    for threshold in thresholds:
        pred = np.asarray(probabilities) >= threshold
        p = precision_score(y_true, pred, zero_division=0); r = recall_score(y_true, pred, zero_division=0)
        rows.append({"threshold": threshold, "precision": p, "recall": r, "f0.5": f05(y_true, pred)})
    best = max(rows, key=lambda x: (x["f0.5"], x["precision"], x["threshold"]))
    return best, rows
