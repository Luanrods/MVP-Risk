"""Matplotlib chart builders for the QCRA/QSRA PDF report and the Streamlit app."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_histogram(final_cost, metrics, baseline_cost):
    """Histogram of simulated final values with P50/P80/P90 and baseline marked."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(final_cost, bins=45, color="#1f6f78", alpha=0.85)
    for p, c in zip((50, 80, 90), ("#d9822b", "#c0392b", "#7d3c98")):
        ax.axvline(metrics[f"p{p}"], linestyle="--", color=c, label=f"P{p}")
    ax.axvline(baseline_cost, color="black", linewidth=1, label="Baseline")
    ax.legend(fontsize=8)
    ax.set_xlabel("Custo Final Simulado")
    ax.set_ylabel("Frequência")
    fig.tight_layout()
    return fig

def plot_s_curve(final_cost, budget=None):
    """Cumulative probability (S-curve) of the simulated final value, budget marked."""
    x = np.sort(final_cost)
    y = np.arange(1, len(x) + 1) / len(x) * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(x, y, color="#1f6f78")
    if budget is not None:
        ax.axvline(budget, linestyle="--", color="#c0392b")
    ax.set_xlabel("Custo Final Simulado")
    ax.set_ylabel("Probabilidade acumulada (%)")
    fig.tight_layout()
    
    return fig

def plot_risk_drivers(drivers, top_n=10):
    """Horizontal bar chart of the top-N risk drivers by |Spearman correlation|.

    Bars are colored red for risks positively correlated with a worse
    (higher) final value, and green for opportunities/risks correlated with
    a better (lower) one.
    """
    plot_df = drivers.head(top_n).sort_values("abs_spearman", ascending=True)
    colors = ["#c0392b" if s > 0 else "#1f8a4c" for s in plot_df["spearman"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(plot_df["risk_id"], plot_df["abs_spearman"], color=colors)
    fig.tight_layout()
    return fig