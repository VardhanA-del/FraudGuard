from pathlib import Path
import sys
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fraudguard.config import settings
from fraudguard.database.connection import open_fraud_store
from fraudguard.database.repository import fetch_quality_history
st.title("Data Quality")
with open_fraud_store(settings.database_path) as risk_store: quality_frame = fetch_quality_history(risk_store)
if quality_frame.empty: st.info("Run ingestion to record source-quality evidence.")
else: st.dataframe(quality_frame, width="stretch")
