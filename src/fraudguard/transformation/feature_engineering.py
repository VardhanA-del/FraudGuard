"""Deterministic features shared by training and replay inference."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_risk_features(transaction_ledger: pd.DataFrame) -> pd.DataFrame:
    """Preserve anonymized signals and derive only source-supported time/amount features."""
    feature_matrix = transaction_ledger.copy()
    feature_matrix["amount_log"] = np.log1p(feature_matrix["Amount"].clip(lower=0))
    feature_matrix["transaction_hour"] = (feature_matrix["Time"] / 3600.0) % 24
    feature_matrix["transaction_time_bucket"] = (feature_matrix["Time"] // 3600).astype(int)
    amount_spread = feature_matrix["Amount"].std(ddof=0)
    feature_matrix["amount_zscore"] = 0.0 if amount_spread == 0 else (feature_matrix["Amount"] - feature_matrix["Amount"].mean()) / amount_spread
    return feature_matrix


def model_feature_columns(transaction_ledger: pd.DataFrame) -> list[str]:
    """Choose available PCA-like variables plus reproducible derived numeric features."""
    pca_signals = sorted(name for name in transaction_ledger.columns if name.startswith("V"))
    return ["Time", "Amount", *pca_signals, "amount_log", "transaction_hour", "transaction_time_bucket", "amount_zscore"]

