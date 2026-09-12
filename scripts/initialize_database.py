"""Create FraudGuard's local SQLite schema."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudguard.config import settings
from fraudguard.pipeline import initialize_platform

if __name__ == "__main__":
    initialize_platform(settings)
    print(f"Database initialized: {settings.database_path}")
