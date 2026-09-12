import pandas as pd
from fraudguard.database.connection import open_fraud_store
from fraudguard.database.repository import persist_transaction_batch
from fraudguard.database.schema import create_fraudguard_schema

def test_transaction_persistence_is_idempotent(tmp_path):
    with open_fraud_store(tmp_path / "risk.db") as risk_store:
        create_fraudguard_schema(risk_store)
        batch = pd.DataFrame({"Time": [1.0], "Amount": [3.0], "Class": [0], "V1": [0.2]})
        assert persist_transaction_batch(risk_store, batch) == 1
        assert persist_transaction_batch(risk_store, batch) == 0

