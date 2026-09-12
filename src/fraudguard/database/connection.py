"""SQLite connection factory."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def open_fraud_store(database_path: Path) -> sqlite3.Connection:
    """Open a foreign-key-aware SQLite connection."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection

