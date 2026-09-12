"""Intent-revealing database operations; UI never assembles ad-hoc SQL."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def persist_transaction_batch(risk_store: sqlite3.Connection, transaction_ledger: pd.DataFrame) -> int:
    """Persist a clean batch idempotently, retaining raw model inputs as JSON."""
    records = []
    for position, transaction_signal in transaction_ledger.reset_index(drop=True).iterrows():
        fingerprint = f"{transaction_signal['Time']:.6f}:{transaction_signal['Amount']:.6f}:{int(transaction_signal['Class'])}:{position}"
        payload = transaction_signal.drop(labels=["Time", "Amount", "Class"]).to_dict()
        records.append((fingerprint, float(transaction_signal["Time"]), float(transaction_signal["Amount"]), json.dumps(payload), int(transaction_signal["Class"]), utc_now()))
    before_count = risk_store.total_changes
    risk_store.executemany("INSERT OR IGNORE INTO transaction_ledger VALUES (?, ?, ?, ?, ?, ?)", records)
    risk_store.commit()
    return risk_store.total_changes - before_count


def record_quality_snapshot(risk_store: sqlite3.Connection, quality_run_id: str, snapshot: dict) -> None:
    risk_store.execute("INSERT OR REPLACE INTO data_quality_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (quality_run_id, utc_now(), snapshot["rows_received"], snapshot["rows_valid"], snapshot["rows_rejected"], snapshot["missing_value_count"], snapshot["duplicate_count"], snapshot["schema_issue_count"], snapshot["quality_score"], snapshot["status"]))
    risk_store.commit()


def register_model_snapshot(risk_store: sqlite3.Connection, model_snapshot: dict) -> None:
    risk_store.execute("INSERT OR REPLACE INTO model_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (model_snapshot["model_version"], model_snapshot["model_name"], model_snapshot["training_timestamp"], model_snapshot["training_rows"], model_snapshot["precision"], model_snapshot["recall"], model_snapshot["f1"], model_snapshot["pr_auc"], model_snapshot["roc_auc"], model_snapshot["artifact_path"], "ACTIVE"))
    risk_store.commit()


def fetch_transaction_window(risk_store: sqlite3.Connection, limit: int, offset: int = 0) -> pd.DataFrame:
    """Read chronological source records for the replay cursor."""
    return pd.read_sql_query("SELECT * FROM transaction_ledger ORDER BY transaction_time, transaction_id LIMIT ? OFFSET ?", risk_store, params=(limit, offset))


def persist_risk_verdicts(risk_store: sqlite3.Connection, prediction_records: Iterable[tuple]) -> None:
    risk_store.executemany("INSERT INTO prediction_registry (transaction_id, model_version, fraud_probability, risk_band, predicted_class, prediction_timestamp) VALUES (?, ?, ?, ?, ?, ?)", prediction_records)
    risk_store.commit()


def fetch_recent_predictions(risk_store: sqlite3.Connection, limit: int = 100) -> pd.DataFrame:
    return pd.read_sql_query("SELECT p.transaction_id, t.amount, p.fraud_probability, p.risk_band, p.predicted_class, p.prediction_timestamp, t.actual_class FROM prediction_registry p JOIN transaction_ledger t USING(transaction_id) ORDER BY p.prediction_id DESC LIMIT ?", risk_store, params=(limit,))


def fetch_prediction_count(risk_store: sqlite3.Connection) -> int:
    """Return the persisted replay position used to resume historical simulation."""
    return int(risk_store.execute("SELECT COUNT(*) FROM prediction_registry").fetchone()[0])


def fetch_prediction_overview(risk_store: sqlite3.Connection) -> dict:
    """Return cumulative operational KPIs rather than a display-window count."""
    overview = risk_store.execute("""
        SELECT COUNT(*) AS transactions_processed,
               SUM(CASE WHEN risk_band = 'HIGH' THEN 1 ELSE 0 END) AS high_risk_events,
               AVG(predicted_class) AS fraud_prediction_rate,
               AVG(fraud_probability) AS average_risk_score
        FROM prediction_registry
    """).fetchone()
    return {
        "transactions_processed": int(overview["transactions_processed"] or 0),
        "high_risk_events": int(overview["high_risk_events"] or 0),
        "fraud_prediction_rate": float(overview["fraud_prediction_rate"] or 0.0),
        "average_risk_score": float(overview["average_risk_score"] or 0.0),
    }


def fetch_model_metrics(risk_store: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM model_registry ORDER BY training_timestamp DESC", risk_store)


def fetch_quality_history(risk_store: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM data_quality_log ORDER BY run_timestamp DESC", risk_store)


def fetch_drift_metrics(risk_store: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM drift_registry ORDER BY run_timestamp DESC", risk_store)
