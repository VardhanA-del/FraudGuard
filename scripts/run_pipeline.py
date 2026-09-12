"""Ingest and validate the configured source CSV."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudguard.config import settings
from fraudguard.logging_config import configure_logging
from fraudguard.pipeline import run_ingestion

if __name__ == "__main__":
    summary = run_ingestion(settings, configure_logging(settings.log_path))
    print(summary)
