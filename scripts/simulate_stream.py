"""Replay historical transactions through the saved FraudGuard artifact."""
import argparse
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudguard.config import settings
from fraudguard.pipeline import calculate_and_store_drift, replay_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay historical fraud records into prediction_registry.")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--transactions", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=settings.default_threshold)
    arguments = parser.parse_args()
    emitted = replay_predictions(settings, arguments.batch_size, arguments.transactions, arguments.threshold)
    print(f"Replay complete. Generated {emitted} simulated predictions.")
    try:
        print(calculate_and_store_drift(settings))
    except ValueError as drift_issue:
        print(f"Drift skipped: {drift_issue}")
