from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_histogram(final_cost, metrics, baseline_cost):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(final_cost, bins=45, color="#1f6f78", alpha=0.85)
    for p, c in zip((50, 80, 90), ("#d9822b", "#c0392b", "#7d3c98")):
        ax.axvline(metrics[f"p{p}"], linestyle="--", color=c, label=f"P{p}")
    ax.axvline(baseline_cost, color="black", linewidth=1, label="Baseline")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

def plot_s_curve(final_cost, budget=None):
    x = np.sort(final_cost)
    y = np.arange(1, len(x) + 1) / len(x) * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(x, y, color="#1f6f78")
    if budget is not None:
        ax.axvline(budget, linestyle="--", color="#c0392b")
    fig.tight_layout()
    return fig

def plot_risk_drivers(drivers, top_n=10):
    plot_df = drivers.head(top_n).sort_values("abs_spearman", ascending=True)
    colors = ["#c0392b" if s > 0 else "#1f8a4c" for s in plot_df["spearman"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh(plot_df["risk_id"], plot_df["abs_spearman"], color=colors)
    fig.tight_layout()
    return fig