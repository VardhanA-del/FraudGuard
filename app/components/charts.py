"""Reusable dashboard visuals."""
import plotly.express as px


def risk_band_chart(prediction_frame):
    distribution = prediction_frame["risk_band"].value_counts().rename_axis("risk_band").reset_index(name="events")
    return px.bar(distribution, x="risk_band", y="events", color="risk_band", color_discrete_map={"LOW": "#2ca02c", "MEDIUM": "#f0ad4e", "HIGH": "#d9534f"}, title="Risk-band distribution")


def probability_chart(prediction_frame):
    return px.histogram(prediction_frame, x="fraud_probability", nbins=30, title="Fraud probability distribution")

