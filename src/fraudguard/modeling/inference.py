"""Artifact-backed replay inference."""
from __future__ import annotations

import joblib
import pandas as pd

from fraudguard.transformation.feature_engineering import build_risk_features


def assign_risk_band(fraud_probability: float, medium_threshold: float, high_threshold: float) -> str:
    if fraud_probability >= high_threshold:
        return "HIGH"
    if fraud_probability >= medium_threshold:
        return "MEDIUM"
    return "LOW"


def score_transaction_batch(model_snapshot: dict, transaction_ledger: pd.DataFrame, threshold: float, medium_threshold: float, high_threshold: float) -> pd.DataFrame:
    """Return probabilities and risk labels for source-supported replay records."""
    risk_signals = build_risk_features(transaction_ledger)
    probabilities = model_snapshot["estimator"].predict_proba(risk_signals[model_snapshot["feature_columns"]])[:, 1]
    verdict_frame = pd.DataFrame({"transaction_id": transaction_ledger["transaction_id"].values, "fraud_probability": probabilities})
    verdict_frame["risk_band"] = verdict_frame["fraud_probability"].map(lambda score: assign_risk_band(float(score), medium_threshold, high_threshold))
    verdict_frame["predicted_class"] = (verdict_frame["fraud_probability"] >= threshold).astype(int)
    return verdict_frame


def load_model_snapshot(artifact_path):
    """Load a saved model artifact with a useful missing-file failure."""
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}. Run scripts/train_model.py first.")
    return joblib.load(artifact_path)

