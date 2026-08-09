"""Interactive Streamlit front-end for the MVP-Risk Monte Carlo engine.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.simulation import simulate_cost_risk, simulate_schedule_risk
from src.metrics import summary_metrics, risk_driver_table, add_emv
from src.charts import plot_histogram, plot_s_curve, plot_risk_drivers
from src.validation import validate_risks

st.set_page_config(page_title="MVP-Risk — QCRA/QSRA", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
st.sidebar.title("MVP-Risk")
st.sidebar.caption("Monte Carlo Quantitative Risk Analysis")

mode = st.sidebar.radio("Analysis type", ["Cost (QCRA)", "Schedule (QSRA-lite)"])
is_cost = mode.startswith("Cost")
baseline_label = "Baseline cost (R$)" if is_cost else "Baseline duration (days)"
unit_label = "R$" if is_cost else "days"

uploaded = st.sidebar.file_uploader("Risk register (CSV)", type="csv")
if uploaded is not None:
    risks = pd.read_csv(uploaded)
    st.sidebar.success(f"Loaded {len(risks)} rows from upload.")
else:
    risks = pd.read_csv("data/example_risk_register.csv")
    st.sidebar.info("Using the bundled example risk register.")

st.sidebar.divider()
st.sidebar.subheader("Assumptions")

baseline_default = 40_000_000.0 if is_cost else 180.0
use_baseline_uncertainty = st.sidebar.checkbox("Model baseline uncertainty", value=False)

if use_baseline_uncertainty:
    col_a, col_b, col_c = st.sidebar.columns(3)
    b_min = col_a.number_input("Min", value=baseline_default * 0.95, step=1.0)
    b_ml = col_b.number_input("Most likely", value=baseline_default, step=1.0)
    b_max = col_c.number_input("Max", value=baseline_default * 1.08, step=1.0)
    baseline = {"distribution": "pert", "min": b_min, "most_likely": b_ml, "max": b_max}
else:
    baseline = st.sidebar.number_input(baseline_label, value=baseline_default, step=1.0)

budget_default = 44_000_000.0 if is_cost else 200.0
budget = st.sidebar.number_input(
    "Budget / target limit" if is_cost else "Target duration limit (days)",
    value=budget_default, step=1.0,
)

n_simulations = st.sidebar.select_slider(
    "Number of simulations", options=[1_000, 5_000, 10_000, 50_000, 100_000], value=100_000,
)
seed = st.sidebar.number_input("Seed", value=42, step=1)

st.sidebar.divider()
st.sidebar.subheader("Correlation (optional)")
st.sidebar.caption(
    "Correlate the *occurrence* of two risks — e.g. if one delay driver "
    "fires, a related one becomes more likely too."
)
enable_correlation = st.sidebar.checkbox("Correlate two risks", value=False)
correlation_matrix = None
if enable_correlation and len(risks) >= 2:
    ids = risks["id"].astype(str).tolist()
    risk_a = st.sidebar.selectbox("Risk A", ids, index=0)
    risk_b = st.sidebar.selectbox("Risk B", ids, index=1)
    strength = st.sidebar.slider("Correlation strength", 0.0, 0.95, 0.6, 0.05)
    if risk_a != risk_b:
        n = len(risks)
        correlation_matrix = np.eye(n)
        idx_a, idx_b = ids.index(risk_a), ids.index(risk_b)
        correlation_matrix[idx_a, idx_b] = strength
        correlation_matrix[idx_b, idx_a] = strength
    else:
        st.sidebar.warning("Pick two different risks to correlate.")

run = st.sidebar.button("Run simulation", type="primary")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("MVP-Risk")
st.caption(
    "Monte Carlo engine for Quantitative Cost & Schedule Risk Analysis — "
    "upload a risk register, adjust assumptions, and get a full probabilistic view."
)

with st.expander("Risk register", expanded=False):
    st.dataframe(risks, use_container_width=True)

errors = validate_risks(risks)
if errors:
    st.error("The risk register has issues:\n\n" + "\n".join(f"- {e}" for e in errors))
    st.stop()

if not run:
    st.info("Set your assumptions in the sidebar and click **Run simulation**.")
    st.stop()


@st.cache_data(show_spinner=False)
def run_simulation(risks_df, baseline, n_simulations, seed, is_cost, correlation_matrix):
    fn = simulate_cost_risk if is_cost else simulate_schedule_risk
    kwargs = dict(n_simulations=n_simulations, seed=seed, correlation_matrix=correlation_matrix)
    if is_cost:
        return fn(risks_df, baseline_cost=baseline, **kwargs)
    return fn(risks_df, baseline_duration=baseline, **kwargs)


with st.spinner(f"Running {n_simulations:,} simulations..."):
    final_value, contributions = run_simulation(
        risks, baseline, int(n_simulations), int(seed), is_cost, correlation_matrix,
    )
    baseline_point = baseline["most_likely"] if isinstance(baseline, dict) else baseline
    metrics = summary_metrics(final_value, baseline_cost=baseline_point, budget=budget)
    drivers = risk_driver_table(contributions, final_value)

# --- KPIs ---
st.subheader("Key results")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Mean", f"{metrics['mean']:,.0f} {unit_label}")
k2.metric("P50", f"{metrics['p50']:,.0f} {unit_label}")
k3.metric("P80", f"{metrics['p80']:,.0f} {unit_label}")
k4.metric("P90", f"{metrics['p90']:,.0f} {unit_label}")
k5.metric(
    "Prob. within limit" if "prob_within_budget" in metrics else "—",
    f"{metrics.get('prob_within_budget', 0):.1%}" if "prob_within_budget" in metrics else "—",
)

st.caption(
    f"P80 contingency vs. baseline: {metrics['p80_contingency']:,.0f} {unit_label} "
    f"({metrics['p80_contingency'] / baseline_point:.1%} of baseline) · "
    f"Std dev: {metrics['std_dev']:,.0f} {unit_label}"
)

# --- Charts ---
c1, c2 = st.columns(2)
with c1:
    st.pyplot(plot_histogram(final_value, metrics, baseline_point), use_container_width=True)
with c2:
    st.pyplot(plot_s_curve(final_value, budget=budget), use_container_width=True)

st.pyplot(plot_risk_drivers(drivers), use_container_width=True)

# --- Risk drivers table + EMV ---
with st.expander("Risk driver ranking (Spearman)"):
    st.dataframe(drivers, use_container_width=True)

if is_cost:
    with st.expander("Expected Monetary Value (deterministic reference)"):
        st.dataframe(add_emv(risks), use_container_width=True)

st.divider()
st.caption(
    "MVP-Risk — open-source Monte Carlo engine for cost & schedule risk analysis. "
    "Schedule mode is a QSRA-lite overlay (independent delay drivers), not a full CPM network simulation."
)
