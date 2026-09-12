"""Population Stability Index based drift checks."""
from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(reference_values, current_values, bins: int = 10) -> float:
    """Measure distribution movement using reference quantile bins."""
    reference = np.asarray(reference_values, dtype=float)
    current = np.asarray(current_values, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    reference_share, _ = np.histogram(reference, bins=edges)
    current_share, _ = np.histogram(current, bins=edges)
    reference_share = np.clip(reference_share / max(reference_share.sum(), 1), 1e-6, None)
    current_share = np.clip(current_share / max(current_share.sum(), 1), 1e-6, None)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def profile_feature_drift(reference_ledger: pd.DataFrame, current_ledger: pd.DataFrame, feature_names: list[str], watch_threshold: float, alert_threshold: float) -> list[dict]:
    drift_profile = []
    for feature_name in feature_names:
        score = population_stability_index(reference_ledger[feature_name], current_ledger[feature_name])
        status = "DRIFT" if score >= alert_threshold else "WATCH" if score >= watch_threshold else "STABLE"
        drift_profile.append({"feature_name": feature_name, "drift_score": score, "drift_status": status, "reference_window": len(reference_ledger), "current_window": len(current_ledger)})
    return drift_profile

