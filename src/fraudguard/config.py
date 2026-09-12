"""Central settings for FraudGuard."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FraudGuardSettings:
    dataset_path: Path = Path(os.getenv("FRAUDGUARD_DATASET_PATH", PROJECT_ROOT / "data/raw/creditcard.csv"))
    database_path: Path = PROJECT_ROOT / "data/database/fraudguard.db"
    model_path: Path = PROJECT_ROOT / "models/fraudguard_model.joblib"
    metadata_path: Path = PROJECT_ROOT / "models/model_metadata.json"
    log_path: Path = PROJECT_ROOT / "logs/fraudguard.log"
    chunk_size: int = int(os.getenv("FRAUDGUARD_CHUNK_SIZE", "10000"))
    default_threshold: float = float(os.getenv("FRAUDGUARD_DEFAULT_THRESHOLD", "0.50"))
    medium_risk_threshold: float = 0.30
    high_risk_threshold: float = 0.70
    drift_watch_threshold: float = 0.10
    drift_alert_threshold: float = 0.25
    random_seed: int = 42


settings = FraudGuardSettings()

