"""Simple health status from recorded pipeline evidence."""
from __future__ import annotations

from typing import Optional


def determine_system_health(quality_score: Optional[float], drift_statuses: list[str]) -> str:
    """Prioritize evidence of drift, then data-quality degradation."""
    if "DRIFT" in drift_statuses or (quality_score is not None and quality_score < 90):
        return "DRIFT DETECTED"
    if "WATCH" in drift_statuses or (quality_score is not None and quality_score < 98):
        return "WATCH"
    return "STABLE"
