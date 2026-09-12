"""SQLite schema for the simulated risk platform."""
from __future__ import annotations

import sqlite3


def create_fraudguard_schema(risk_store: sqlite3.Connection) -> None:
    """Create durable tables and indexes used by pipeline and dashboard."""
    risk_store.executescript("""
    CREATE TABLE IF NOT EXISTS transaction_ledger (
      transaction_id TEXT PRIMARY KEY, transaction_time REAL NOT NULL, amount REAL NOT NULL,
      feature_payload TEXT NOT NULL, actual_class INTEGER NOT NULL CHECK(actual_class IN (0,1)),
      ingested_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS prediction_registry (
      prediction_id INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id TEXT NOT NULL,
      model_version TEXT NOT NULL, fraud_probability REAL NOT NULL, risk_band TEXT NOT NULL,
      predicted_class INTEGER NOT NULL CHECK(predicted_class IN (0,1)), prediction_timestamp TEXT NOT NULL,
      FOREIGN KEY(transaction_id) REFERENCES transaction_ledger(transaction_id)
    );
    CREATE TABLE IF NOT EXISTS model_registry (
      model_version TEXT PRIMARY KEY, model_name TEXT NOT NULL, training_timestamp TEXT NOT NULL,
      training_rows INTEGER NOT NULL, precision_score REAL, recall_score REAL, f1_score REAL,
      pr_auc REAL, roc_auc REAL, artifact_path TEXT NOT NULL, status TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS data_quality_log (
      quality_run_id TEXT PRIMARY KEY, run_timestamp TEXT NOT NULL, rows_received INTEGER NOT NULL,
      rows_valid INTEGER NOT NULL, rows_rejected INTEGER NOT NULL, missing_value_count INTEGER NOT NULL,
      duplicate_count INTEGER NOT NULL, schema_issue_count INTEGER NOT NULL, quality_score REAL NOT NULL, status TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS drift_registry (
      drift_run_id TEXT NOT NULL, run_timestamp TEXT NOT NULL, feature_name TEXT NOT NULL,
      drift_score REAL NOT NULL, drift_status TEXT NOT NULL, reference_window INTEGER NOT NULL, current_window INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS pipeline_runs (
      run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, completed_at TEXT, records_processed INTEGER NOT NULL DEFAULT 0,
      records_failed INTEGER NOT NULL DEFAULT 0, pipeline_status TEXT NOT NULL, error_message TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_ledger_time ON transaction_ledger(transaction_time);
    CREATE INDEX IF NOT EXISTS idx_prediction_time ON prediction_registry(prediction_timestamp);
    CREATE INDEX IF NOT EXISTS idx_prediction_transaction ON prediction_registry(transaction_id);
    """)
    risk_store.commit()

