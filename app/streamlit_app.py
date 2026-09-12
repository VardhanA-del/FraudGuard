"""FraudGuard Risk Intelligence Console entrypoint."""
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from fraudguard.config import settings
from fraudguard.database.connection import open_fraud_store
from fraudguard.database.repository import fetch_prediction_overview, fetch_recent_predictions

st.set_page_config(page_title="FraudGuard | Risk Intelligence Console", page_icon="🛡️", layout="wide")
st.title("FraudGuard")
st.caption("Risk Intelligence Console · historical transaction replay simulation")
try:
    with open_fraud_store(settings.database_path) as risk_store:
        prediction_frame = fetch_recent_predictions(risk_store, 500)
        prediction_overview = fetch_prediction_overview(risk_store)
    if prediction_frame.empty:
        st.info("No replay predictions yet. Run the pipeline, train a model, then simulate the stream.")
    else:
        processed, high_risk, fraud_rate, average_score = st.columns(4)
        processed.metric("Transactions Processed", prediction_overview["transactions_processed"])
        high_risk.metric("High-Risk Events", prediction_overview["high_risk_events"])
        fraud_rate.metric("Fraud Prediction Rate", f"{prediction_overview['fraud_prediction_rate']:.1%}")
        average_score.metric("Average Risk Score", f"{prediction_overview['average_risk_score']:.3f}")
        st.caption("KPI cards summarize all saved replay predictions. The table below shows the most recent 500 events.")
        newest_batch_size = int(st.session_state.get("last_replay_size", 0))
        if newest_batch_size:
            st.subheader(f"Current Replay Batch ({newest_batch_size} New Events)")
            st.dataframe(prediction_frame.head(newest_batch_size)[["transaction_id", "amount", "fraud_probability", "risk_band", "prediction_timestamp"]], width="stretch")
        st.subheader("Recent Simulated Events (Latest 500)")
        marked_events = prediction_frame[["transaction_id", "amount", "fraud_probability", "risk_band", "prediction_timestamp"]].copy()
        marked_events.insert(0, "replay_status", "PREVIOUS")
        if newest_batch_size:
            marked_events.loc[marked_events.index[:newest_batch_size], "replay_status"] = "NEW"
        st.dataframe(marked_events, width="stretch")
except Exception as dashboard_issue:
    st.error(f"FraudGuard cannot read the local store: {dashboard_issue}")
