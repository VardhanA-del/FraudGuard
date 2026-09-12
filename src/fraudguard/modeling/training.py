"""Chronological candidate-model training."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraudguard.modeling.evaluation import evaluate_risk_scores, select_operating_threshold
from fraudguard.transformation.feature_engineering import build_risk_features, model_feature_columns


def train_candidate_models(transaction_ledger: pd.DataFrame, artifact_path: Path, metadata_path: Path, random_seed: int) -> dict:
    """Fit two scikit-learn candidates using an earlier-to-later chronological holdout."""
    chronological_ledger = transaction_ledger.sort_values("Time").reset_index(drop=True)
    split_cursor = int(len(chronological_ledger) * 0.80)
    learning_ledger, holdout_ledger = chronological_ledger.iloc[:split_cursor], chronological_ledger.iloc[split_cursor:]
    learning_signals = build_risk_features(learning_ledger)
    holdout_signals = build_risk_features(holdout_ledger)
    signal_columns = model_feature_columns(learning_signals)
    candidates = {
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("estimator", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_seed))]),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=14, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=random_seed),
    }
    comparison = []
    fitted_candidates = {}
    for candidate_name, fraud_estimator in candidates.items():
        fraud_estimator.fit(learning_signals[signal_columns], learning_ledger["Class"])
        holdout_probabilities = fraud_estimator.predict_proba(holdout_signals[signal_columns])[:, 1]
        operating_threshold, _ = select_operating_threshold(holdout_ledger["Class"], holdout_probabilities)
        comparison.append({"model_name": candidate_name, "threshold": operating_threshold, **evaluate_risk_scores(holdout_ledger["Class"], holdout_probabilities, operating_threshold)})
        fitted_candidates[candidate_name] = fraud_estimator
    champion_metrics = max(comparison, key=lambda metrics: metrics["pr_auc"])
    champion_name = champion_metrics["model_name"]
    model_version = "fraudguard-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": fitted_candidates[champion_name], "feature_columns": signal_columns, "threshold": champion_metrics["threshold"], "model_version": model_version}, artifact_path)
    metadata = {"model_version": model_version, "model_name": champion_name, "training_timestamp": datetime.now(timezone.utc).isoformat(), "training_rows": len(learning_ledger), "holdout_rows": len(holdout_ledger), "feature_list": signal_columns, "threshold": champion_metrics["threshold"], "metrics": champion_metrics, "comparison": comparison, "artifact_path": str(artifact_path)}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

