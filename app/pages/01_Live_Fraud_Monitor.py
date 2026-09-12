"""Interactive simulated-replay monitor and analyst risk-check workspace."""
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from components.charts import probability_chart, risk_band_chart
from fraudguard.config import settings
from fraudguard.database.connection import open_fraud_store
from fraudguard.database.repository import fetch_prediction_count, fetch_recent_predictions
from fraudguard.modeling.inference import load_model_snapshot, score_transaction_batch
from fraudguard.pipeline import replay_predictions


def risk_lab_record(model_snapshot: dict) -> pd.DataFrame:
    """Collect a source-compatible analyst scenario without inventing customer attributes."""
    observation = {
        "transaction_id": "analyst-what-if",
        "Time": st.number_input("Transaction time (seconds)", min_value=0.0, value=0.0, help="Use the dataset's elapsed-time convention."),
        "Amount": st.number_input("Transaction amount", min_value=0.0, value=100.0, help="Enter the amount you want to evaluate."),
        "Class": 0,
    }
    source_signals = [name for name in model_snapshot["feature_columns"] if name.startswith("V")]
    with st.expander("Advanced anonymized V-signals", expanded=False):
        st.caption("These PCA-style inputs exist in the source dataset. Leave them at zero for a neutral what-if scenario or enter observed source-compatible values.")
        signal_columns = st.columns(4)
        for position, signal_name in enumerate(source_signals):
            with signal_columns[position % 4]:
                observation[signal_name] = st.number_input(signal_name, value=0.0, format="%.5f")
    return pd.DataFrame([observation])


st.title("Live Fraud Monitor")
st.caption("Replay controls write historical simulation results to SQLite. The Risk Lab evaluates an analyst-entered what-if event without claiming it is a live bank transaction.")

try:
    model_snapshot = load_model_snapshot(settings.model_path)
except FileNotFoundError as model_issue:
    st.warning(str(model_issue))
    st.stop()

control_one, control_two, control_three = st.columns(3)
with control_one:
    active_threshold = st.slider("Fraud decision threshold", min_value=0.05, max_value=0.95, value=float(model_snapshot["threshold"]), step=0.05, help="Lower thresholds flag more events; higher thresholds reduce false alerts but can miss fraud.")
with control_two:
    release_size = st.selectbox("Replay batch size", [10, 25, 50, 100], index=1)
with control_three:
    if "replay_cursor" not in st.session_state:
        with open_fraud_store(settings.database_path) as risk_store:
            st.session_state.replay_cursor = fetch_prediction_count(risk_store)
    st.metric("Replay cursor", st.session_state.replay_cursor)
    if st.button("Replay next batch", type="primary", width="stretch"):
        generated_events = replay_predictions(settings, release_size, release_size, active_threshold, st.session_state.replay_cursor)
        st.session_state.replay_cursor += generated_events
        st.session_state.last_replay_size = generated_events
        st.success(f"Released and scored {generated_events} historical transactions.")
        st.rerun()

st.divider()
st.subheader("Transaction Risk Lab")
st.caption("Interactive what-if prediction. This is useful for demonstrating the model workflow; it is not evidence of real-time payment processing.")
with st.form("risk_lab_form"):
    analyst_event = risk_lab_record(model_snapshot)
    evaluate_event = st.form_submit_button("Evaluate transaction risk", type="primary", width="stretch")
if evaluate_event:
    risk_verdict = score_transaction_batch(model_snapshot, analyst_event, active_threshold, settings.medium_risk_threshold, settings.high_risk_threshold).iloc[0]
    if "risk_lab_history" not in st.session_state:
        st.session_state.risk_lab_history = []
    st.session_state.risk_lab_history.append({
        "scenario": len(st.session_state.risk_lab_history) + 1,
        "amount": float(analyst_event.iloc[0]["Amount"]),
        "fraud_probability": float(risk_verdict.fraud_probability),
        "risk_band": risk_verdict.risk_band,
        "threshold": active_threshold,
    })
    outcome_one, outcome_two, outcome_three = st.columns(3)
    outcome_one.metric("Fraud probability", f"{risk_verdict.fraud_probability:.2%}")
    outcome_two.metric("Risk band", risk_verdict.risk_band)
    outcome_three.metric("Decision", "FLAG FOR REVIEW" if risk_verdict.predicted_class else "NO FRAUD FLAG")
    st.info("The result uses the saved model and your current threshold. It is intentionally not added to the historical evaluation ledger because no real ground truth exists for a what-if input.")

if st.session_state.get("risk_lab_history"):
    scenario_history = pd.DataFrame(st.session_state.risk_lab_history)
    st.subheader("What-if analysis")
    gauge_chart = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(scenario_history.iloc[-1]["fraud_probability"]) * 100,
        number={"suffix": "%"},
        title={"text": "Latest scenario fraud probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#d9534f"}, "steps": [
            {"range": [0, settings.medium_risk_threshold * 100], "color": "#dff0d8"},
            {"range": [settings.medium_risk_threshold * 100, settings.high_risk_threshold * 100], "color": "#fcf8e3"},
            {"range": [settings.high_risk_threshold * 100, 100], "color": "#f2dede"},
        ], "threshold": {"line": {"color": "#1f4e79", "width": 4}, "value": active_threshold * 100}},
    ))
    history_chart = px.bar(scenario_history, x="scenario", y="fraud_probability", color="risk_band", hover_data=["amount", "threshold"], title="Risk Lab scenario history")
    gauge_column, history_column = st.columns(2)
    gauge_column.plotly_chart(gauge_chart, use_container_width=True)
    history_column.plotly_chart(history_chart, use_container_width=True)
    if st.button("Clear Risk Lab scenarios", width="stretch"):
        st.session_state.risk_lab_history = []
        st.rerun()

st.divider()
with open_fraud_store(settings.database_path) as risk_store:
    prediction_frame = fetch_recent_predictions(risk_store, 500)
if prediction_frame.empty:
    st.info("No replay predictions yet. Use **Replay next batch** to populate the operational feed.")
else:
    left_chart, right_chart = st.columns(2)
    left_chart.plotly_chart(risk_band_chart(prediction_frame), use_container_width=True)
    right_chart.plotly_chart(probability_chart(prediction_frame), use_container_width=True)
    newest_batch_size = int(st.session_state.get("last_replay_size", 0))
    if newest_batch_size:
        st.subheader(f"Current replay batch ({newest_batch_size} new events)")
        st.dataframe(prediction_frame.head(newest_batch_size), width="stretch")
    st.subheader("Recent simulated events (latest 500)")
    marked_events = prediction_frame.copy()
    marked_events.insert(0, "replay_status", "PREVIOUS")
    if newest_batch_size:
        marked_events.loc[marked_events.index[:newest_batch_size], "replay_status"] = "NEW"
    st.dataframe(marked_events, width="stretch")
