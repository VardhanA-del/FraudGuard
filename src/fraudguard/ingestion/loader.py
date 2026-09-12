"""Safe chunked CSV ingestion."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd


def inspect_source_dataset(source_path: Path) -> dict:
    """Inspect source facts without materializing the full CSV."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {source_path}. Put creditcard.csv in data/raw/ or set FRAUDGUARD_DATASET_PATH.")
    sample_ledger = pd.read_csv(source_path, nrows=5000)
    required = {"Time", "Amount", "Class"}
    return {"columns": sample_ledger.columns.tolist(), "dtypes": sample_ledger.dtypes.astype(str).to_dict(), "sample_missing_values": int(sample_ledger.isna().sum().sum()), "has_time": "Time" in sample_ledger, "has_amount": "Amount" in sample_ledger, "has_target": "Class" in sample_ledger, "required_present": required.issubset(sample_ledger.columns)}


def iterate_transaction_chunks(source_path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Yield source records in bounded batches after checking essential schema."""
    profile = inspect_source_dataset(source_path)
    if not profile["required_present"]:
        raise ValueError("CSV must contain Time, Amount, and Class columns.")
    yield from pd.read_csv(source_path, chunksize=chunk_size)

