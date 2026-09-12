from pathlib import Path
import sys
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fraudguard.config import settings
from fraudguard.database.connection import open_fraud_store
from fraudguard.database.repository import fetch_drift_metrics
st.title("Model Drift")
with open_fraud_store(settings.database_path) as risk_store: drift_frame = fetch_drift_metrics(risk_store)
if drift_frame.empty: st.info("Complete replay after ingestion to calculate PSI drift.")
else:
    status = "🔴 Drift Detected" if "DRIFT" in set(drift_frame.drift_status) else "🟡 Watch" if "WATCH" in set(drift_frame.drift_status) else "🟢 Stable"
    st.subheader(f"SYSTEM HEALTH: {status}"); st.dataframe(drift_frame, width="stretch")
