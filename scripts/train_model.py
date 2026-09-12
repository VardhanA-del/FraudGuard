"""Train candidate estimators and register the PR-AUC champion."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudguard.config import settings
from fraudguard.pipeline import train_champion_model

if __name__ == "__main__":
    metadata = train_champion_model(settings)
    print(f"Saved {metadata['model_name']} as {metadata['model_version']} (PR-AUC={metadata['metrics']['pr_auc']:.4f})")
