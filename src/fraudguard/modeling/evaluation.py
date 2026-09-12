"""Metrics and threshold selection tailored to rare fraud events."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_risk_scores(actual_labels, fraud_probabilities, threshold: float) -> dict:
    """Calculate threshold-aware classification metrics and ranking metrics."""
    risk_verdicts = (np.asarray(fraud_probabilities) >= threshold).astype(int)
    labels = np.asarray(actual_labels)
    return {"precision": float(precision_score(labels, risk_verdicts, zero_division=0)), "recall": float(recall_score(labels, risk_verdicts, zero_division=0)), "f1": float(f1_score(labels, risk_verdicts, zero_division=0)), "pr_auc": float(average_precision_score(labels, fraud_probabilities)), "roc_auc": float(roc_auc_score(labels, fraud_probabilities)), "confusion_matrix": confusion_matrix(labels, risk_verdicts, labels=[0, 1]).tolist()}


def select_operating_threshold(actual_labels, fraud_probabilities, candidates=(0.30, 0.40, 0.50, 0.60, 0.70, 0.80)) -> tuple[float, list[dict]]:
    """Choose highest F1 threshold, retaining the full evidence table."""
    threshold_evidence = [{"threshold": threshold, **evaluate_risk_scores(actual_labels, fraud_probabilities, threshold)} for threshold in candidates]
    chosen = max(threshold_evidence, key=lambda evidence: evidence["f1"])
    return float(chosen["threshold"]), threshold_evidence

