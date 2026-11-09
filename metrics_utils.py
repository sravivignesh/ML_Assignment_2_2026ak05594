"""Consistent evaluation for all five multi-class classifiers.

For this 10-class task, use macro-averaged precision/recall/F1 and
one-vs-rest macro ROC AUC. MCC supports multi-class labels directly.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, matthews_corrcoef, precision_score,
    recall_score, f1_score, roc_auc_score,
)


def score_predictions(y_true, y_pred, y_proba, classes):
    """Return six metrics, leaving AUC as NaN when it cannot be computed."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.asarray(classes)

    if y_proba.shape != (len(y_true), len(classes)):
        raise ValueError("Probability matrix size doesn't match rows/classes")

    try:
        # AUC across all 10 digit classes is defined when all classes occur in labels.
        auc = roc_auc_score(y_true, y_proba, labels=classes,
                            multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC (OvR macro)": float(auc),
        "Precision (macro)": float(precision_score(y_true, y_pred,
                                                     average="macro", zero_division=0)),
        "Recall (macro)": float(recall_score(y_true, y_pred,
                                              average="macro", zero_division=0)),
        "F1 (macro)": float(f1_score(y_true, y_pred,
                                      average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
