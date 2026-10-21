"""Threshold-based and ranking metrics for multi-label classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

def threshold_predictions(y_prob, threshold=0.5):
    """Convert probability scores into binary predictions."""
    probs = np.asarray(y_prob)
    if probs.ndim != 2:
        raise ValueError("Expected a 2D probability matrix.")
    return (probs >= threshold).astype(np.uint8)

def calculate_metrics(y_true, y_prob, threshold=0.5, label_names=None):
    """Calculate micro/macro metrics and supported per-label ranking metrics."""
    y_true = np.asarray(y_true).astype(np.uint8)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_prob must have the same 2D shape.")
    if not np.isfinite(y_prob).all():
        raise ValueError("Probability matrix contains NaN or infinite values.")

    y_pred = threshold_predictions(y_prob, threshold)
    result = {
        "threshold": float(threshold),
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_label": {},
    }

    names = label_names or [f"label_{i}" for i in range(y_true.shape[1])]
    for i, name in enumerate(names):
        row = {
            "support": int(y_true[:, i].sum()),
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
        }
        # ROC-AUC/AP require positive and negative examples.
        if np.unique(y_true[:, i]).size == 2:
            row["roc_auc"] = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
            row["average_precision"] = float(average_precision_score(y_true[:, i], y_prob[:, i]))
        else:
            row["roc_auc"] = None
            row["average_precision"] = None
        result["per_label"][name] = row
    return result
