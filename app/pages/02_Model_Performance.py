from pathlib import Path
import sys
import streamlit as st
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fraudguard.config import settings
from fraudguard.database.connection import open_fraud_store
from fraudguard.database.repository import fetch_model_metrics
st.title("Model Performance")
with open_fraud_store(settings.database_path) as risk_store: metrics_frame = fetch_model_metrics(risk_store)
if metrics_frame.empty: st.info("Train a model to register measured metrics.")
else: st.dataframe(metrics_frame[["model_name", "model_version", "precision_score", "recall_score", "f1_score", "roc_auc", "pr_auc", "training_timestamp"]], width="stretch")
