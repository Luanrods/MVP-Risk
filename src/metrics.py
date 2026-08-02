from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def summary_metrics(final_cost, baseline_cost, budget=None):
    result = {"mean": float(np.mean(final_cost)), "std_dev": float(np.std(final_cost, ddof=1))}
    for p in (50, 80, 90, 95):
        result[f"p{p}"] = float(np.percentile(final_cost, p))
    result["p80_contingency"] = result["p80"] - baseline_cost
    if budget is not None:
        result["prob_within_budget"] = float(np.mean(final_cost <= budget))
    return result

def mean_impact(row, lambd: float = 4.0) -> float:
    a, m, b = float(row["min_impact"]), float(row["most_likely_impact"]), float(row["max_impact"])
    if row["distribution"] == "fixed":
        return m
    if row["distribution"] == "triangular":
        return (a + m + b) / 3.0
    if row["distribution"] == "pert":
        return (a + lambd * m + b) / (lambd + 2.0)
    raise ValueError(row["distribution"])

def add_emv(df):
    out = df.copy()
    out["mean_impact"] = out.apply(mean_impact, axis=1)
    sign = out["type"].map({"threat": 1.0, "opportunity": -1.0})
    out["signed_emv"] = out["probability"] * out["mean_impact"] * sign
    return out

def risk_driver_table(contribution_df, final_cost):
    rows = []
    for risk_id in contribution_df.columns:
        x = contribution_df[risk_id].to_numpy()
        corr = 0.0 if np.allclose(x, x[0]) else float(spearmanr(x, final_cost).statistic)
        if np.isnan(corr):
            corr = 0.0
        rows.append({"risk_id": risk_id, "spearman": corr, "abs_spearman": abs(corr)})
    return pd.DataFrame(rows).sort_values("abs_spearman", ascending=False)