from pathlib import Path
import pytest
from fraudguard.ingestion.loader import inspect_source_dataset

def test_missing_source_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_source_dataset(Path(tmp_path / "absent.csv"))

