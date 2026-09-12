import pandas as pd
from fraudguard.validation.quality_checks import assess_transaction_quality

def test_quality_rejects_invalid_target_and_duplicate():
    transaction_ledger = pd.DataFrame({"Time": [1, 1, 3], "Amount": [2.0, 2.0, -1.0], "Class": [0, 0, 4]})
    quality_snapshot = assess_transaction_quality(transaction_ledger)
    assert quality_snapshot.duplicate_count == 1
    assert quality_snapshot.invalid_target_count == 1
    assert quality_snapshot.rows_valid == 1

