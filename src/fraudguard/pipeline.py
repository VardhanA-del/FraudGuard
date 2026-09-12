"""Orchestration for ingestion, training, replay, and monitoring."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fraudguard.config import FraudGuardSettings
from fraudguard.database.connection import open_fraud_store
from fraudguard.database.repository import (fetch_transaction_window, persist_risk_verdicts,
    persist_transaction_batch, record_quality_snapshot, register_model_snapshot, utc_now)
from fraudguard.database.schema import create_fraudguard_schema
from fraudguard.ingestion.loader import iterate_transaction_chunks
from fraudguard.modeling.inference import load_model_snapshot, score_transaction_batch
from fraudguard.modeling.training import train_candidate_models
from fraudguard.monitoring.drift import profile_feature_drift
from fraudguard.validation.quality_checks import assess_transaction_quality, clean_transaction_batch


def initialize_platform(settings: FraudGuardSettings) -> None:
    with open_fraud_store(settings.database_path) as risk_store:
        create_fraudguard_schema(risk_store)


def run_ingestion(settings: FraudGuardSettings, logger: logging.Logger) -> dict:
    """Validate and persist the source in transactional chunks."""
    initialize_platform(settings)
    run_id = str(uuid.uuid4())
    accepted_records = rejected_records = 0
    with open_fraud_store(settings.database_path) as risk_store:
        risk_store.execute("INSERT INTO pipeline_runs (run_id, started_at, pipeline_status) VALUES (?, ?, ?)", (run_id, utc_now(), "RUNNING"))
        risk_store.commit()
        try:
            for chunk_number, source_chunk in enumerate(iterate_transaction_chunks(settings.dataset_path, settings.chunk_size), start=1):
                quality_snapshot = assess_transaction_quality(source_chunk)
                clean_ledger = clean_transaction_batch(source_chunk)
                inserted = persist_transaction_batch(risk_store, clean_ledger)
                record_quality_snapshot(risk_store, f"{run_id}-{chunk_number}", quality_snapshot.as_dict())
                accepted_records += inserted
                rejected_records += quality_snapshot.rows_rejected
                logger.info("chunk=%s rows=%s inserted=%s quality=%s", chunk_number, len(source_chunk), inserted, quality_snapshot.quality_score)
            risk_store.execute("UPDATE pipeline_runs SET completed_at=?, records_processed=?, records_failed=?, pipeline_status=? WHERE run_id=?", (utc_now(), accepted_records, rejected_records, "COMPLETED", run_id))
            risk_store.commit()
        except Exception as failure:
            risk_store.execute("UPDATE pipeline_runs SET completed_at=?, pipeline_status=?, error_message=? WHERE run_id=?", (utc_now(), "FAILED", str(failure), run_id))
            risk_store.commit()
            raise
    return {"run_id": run_id, "records_inserted": accepted_records, "records_rejected": rejected_records}


def _load_training_ledger(database_path: Path) -> pd.DataFrame:
    with open_fraud_store(database_path) as risk_store:
        stored_ledger = pd.read_sql_query("SELECT transaction_id, transaction_time, amount, feature_payload, actual_class FROM transaction_ledger ORDER BY transaction_time", risk_store)
    signal_payload = pd.json_normalize(stored_ledger.pop("feature_payload").map(json.loads))
    stored_ledger = pd.concat([stored_ledger.drop(columns=["transaction_id"]), signal_payload], axis=1)
    return stored_ledger.rename(columns={"transaction_time": "Time", "amount": "Amount", "actual_class": "Class"})


def train_champion_model(settings: FraudGuardSettings) -> dict:
    transaction_ledger = _load_training_ledger(settings.database_path)
    if len(transaction_ledger) < 100 or transaction_ledger["Class"].nunique() < 2:
        raise ValueError("At least 100 records with both classes are required before training.")
    metadata = train_candidate_models(transaction_ledger, settings.model_path, settings.metadata_path, settings.random_seed)
    metrics = metadata["metrics"]
    registry_record = {"model_version": metadata["model_version"], "model_name": metadata["model_name"], "training_timestamp": metadata["training_timestamp"], "training_rows": metadata["training_rows"], "artifact_path": metadata["artifact_path"], **metrics}
    with open_fraud_store(settings.database_path) as risk_store:
        register_model_snapshot(risk_store, registry_record)
    return metadata


def replay_predictions(settings: FraudGuardSettings, batch_size: int, transaction_limit: int, threshold: float, start_offset: int = 0) -> int:
    if batch_size <= 0 or transaction_limit <= 0:
        raise ValueError("Batch size and transaction limit must be positive.")
    if not 0 < threshold < 1:
        raise ValueError("Risk threshold must be between 0 and 1.")
    model_snapshot = load_model_snapshot(settings.model_path)
    replayed_events = 0
    with open_fraud_store(settings.database_path) as risk_store:
        for transaction_cursor in range(0, transaction_limit, batch_size):
            stored_window = fetch_transaction_window(risk_store, min(batch_size, transaction_limit - transaction_cursor), start_offset + transaction_cursor)
            if stored_window.empty:
                break
            payload_frame = pd.json_normalize(stored_window.pop("feature_payload").map(json.loads))
            replay_ledger = pd.concat([stored_window.rename(columns={"transaction_time": "Time", "amount": "Amount", "actual_class": "Class"}), payload_frame], axis=1)
            verdict_frame = score_transaction_batch(model_snapshot, replay_ledger, threshold, settings.medium_risk_threshold, settings.high_risk_threshold)
            verdict_records = [(signal.transaction_id, model_snapshot["model_version"], float(signal.fraud_probability), signal.risk_band, int(signal.predicted_class), utc_now()) for signal in verdict_frame.itertuples(index=False)]
            persist_risk_verdicts(risk_store, verdict_records)
            replayed_events += len(verdict_records)
    return replayed_events


def calculate_and_store_drift(settings: FraudGuardSettings, window_size: int = 5000) -> list[dict]:
    transaction_ledger = _load_training_ledger(settings.database_path)
    if len(transaction_ledger) < window_size * 2:
        raise ValueError("Not enough stored transactions for two drift windows.")
    important_signals = [name for name in ["Amount", "Time", "V1", "V2", "V3"] if name in transaction_ledger]
    drift_profile = profile_feature_drift(transaction_ledger.iloc[-window_size * 2:-window_size], transaction_ledger.iloc[-window_size:], important_signals, settings.drift_watch_threshold, settings.drift_alert_threshold)
    drift_run_id = str(uuid.uuid4())
    with open_fraud_store(settings.database_path) as risk_store:
        risk_store.executemany("INSERT INTO drift_registry VALUES (?, ?, ?, ?, ?, ?, ?)", [(drift_run_id, utc_now(), profile["feature_name"], profile["drift_score"], profile["drift_status"], profile["reference_window"], profile["current_window"]) for profile in drift_profile])
        risk_store.commit()
    return drift_profile
