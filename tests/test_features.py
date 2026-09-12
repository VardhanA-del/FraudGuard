import pandas as pd
from fraudguard.transformation.feature_engineering import build_risk_features, model_feature_columns

def test_risk_features_are_deterministic():
    transaction_ledger = pd.DataFrame({"Time": [0.0, 3600.0], "Amount": [10.0, 30.0], "Class": [0, 1], "V1": [1.0, 2.0]})
    first_matrix = build_risk_features(transaction_ledger)
    second_matrix = build_risk_features(transaction_ledger)
    assert first_matrix.equals(second_matrix)
    assert "amount_log" in model_feature_columns(first_matrix)

