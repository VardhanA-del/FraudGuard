"""Observed prediction performance once labels are available in the replay dataset."""
from __future__ import annotations

from fraudguard.modeling.evaluation import evaluate_risk_scores


def calculate_observed_performance(prediction_frame):
    """Evaluate replay predictions against historical labels; not live production labels."""
    if prediction_frame.empty or prediction_frame["actual_class"].nunique() < 2:
        return None
    return evaluate_risk_scores(prediction_frame["actual_class"], prediction_frame["fraud_probability"], 0.5)

