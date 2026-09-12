import numpy as np
from fraudguard.modeling.evaluation import evaluate_risk_scores, select_operating_threshold

def test_threshold_metrics_are_valid():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, evidence = select_operating_threshold(labels, probabilities)
    assert 0 < threshold < 1
    assert 0 <= evaluate_risk_scores(labels, probabilities, threshold)["pr_auc"] <= 1
    assert evidence

