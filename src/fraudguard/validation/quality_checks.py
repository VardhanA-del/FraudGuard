"""Dataset validation with transparent, source-derived scoring."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualitySnapshot:
    rows_received: int
    rows_valid: int
    rows_rejected: int
    missing_value_count: int
    duplicate_count: int
    schema_issue_count: int
    negative_amount_count: int
    invalid_target_count: int
    quality_score: float
    status: str

    def as_dict(self) -> dict:
        return asdict(self)


def expected_columns(column_names: Iterable[str]) -> list[str]:
    """Return required credit-card columns inferred from the available schema."""
    names = list(column_names)
    required = [name for name in ("Time", "Amount", "Class") if name in names]
    required.extend(sorted(name for name in names if name.startswith("V")),)
    return required


def assess_transaction_quality(transaction_ledger: pd.DataFrame, required_columns: Iterable[str] | None = None) -> QualitySnapshot:
    """Assess a transaction batch; score is valid-row share after schema checks."""
    required = list(required_columns or ("Time", "Amount", "Class"))
    schema_issues = len(set(required) - set(transaction_ledger.columns))
    if schema_issues:
        return QualitySnapshot(len(transaction_ledger), 0, len(transaction_ledger), 0, 0, schema_issues, 0, 0, 0.0, "REJECTED")
    missing_count = int(transaction_ledger[required].isna().sum().sum())
    duplicate_count = int(transaction_ledger.duplicated().sum())
    numeric_frame = transaction_ledger.select_dtypes(include=[np.number])
    non_finite = int((~np.isfinite(numeric_frame.to_numpy())).sum())
    negative_amounts = int((pd.to_numeric(transaction_ledger["Amount"], errors="coerce") < 0).sum())
    invalid_targets = int((~transaction_ledger["Class"].isin([0, 1])).sum())
    invalid_mask = transaction_ledger[required].isna().any(axis=1) | transaction_ledger.duplicated() | (pd.to_numeric(transaction_ledger["Amount"], errors="coerce") < 0) | (~transaction_ledger["Class"].isin([0, 1]))
    valid_rows = int((~invalid_mask).sum())
    score = round(100 * valid_rows / len(transaction_ledger), 2) if len(transaction_ledger) else 0.0
    status = "ACCEPTED" if schema_issues == 0 and valid_rows else "REJECTED"
    return QualitySnapshot(len(transaction_ledger), valid_rows, len(transaction_ledger) - valid_rows, missing_count + non_finite, duplicate_count, schema_issues, negative_amounts, invalid_targets, score, status)


def clean_transaction_batch(transaction_ledger: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that cannot be safely persisted or modeled."""
    cleaned_ledger = transaction_ledger.drop_duplicates().copy()
    cleaned_ledger = cleaned_ledger.dropna(subset=["Time", "Amount", "Class"])
    cleaned_ledger = cleaned_ledger[cleaned_ledger["Amount"] >= 0]
    cleaned_ledger = cleaned_ledger[cleaned_ledger["Class"].isin([0, 1])]
    return cleaned_ledger.replace([np.inf, -np.inf], np.nan).dropna()

