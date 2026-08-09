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
    sign = out["type"].map({"risco": 1.0, "oportunidade": -1.0})
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


def convergence_report(
    simulate_fn,
    df: pd.DataFrame,
    baseline,
    n_values: list[int] | None = None,
    seeds: list[int] | None = None,
    percentiles: tuple[int, ...] = (50, 80),
) -> pd.DataFrame:
    """Check how stable percentile estimates are across sample sizes and seeds.

    Runs ``simulate_fn`` (e.g. ``simulate_cost_risk`` or
    ``simulate_schedule_risk``) once per (n_simulations, seed) combination
    and reports the spread of each requested percentile across seeds, for
    each sample size. A shrinking spread as n_simulations grows is evidence
    the chosen iteration count is enough for a stable answer; a spread that
    stays wide is a sign more iterations (or fewer, noisier seeds) are needed.

    Parameters
    ----------
    simulate_fn : callable
        A function with signature (df, baseline, n_simulations, seed) -> (values, contributions),
        i.e. ``simulate_cost_risk`` or ``simulate_schedule_risk``.
    df : DataFrame
        Risk register.
    baseline : float or dict
        Baseline value or distribution spec, passed through to simulate_fn.
    n_values : list[int], optional
        Sample sizes to test. Defaults to [1_000, 10_000, 50_000, 100_000].
    seeds : list[int], optional
        Seeds to repeat each sample size with. Defaults to [1, 2, 3, 4, 5].
    percentiles : tuple[int, ...]
        Percentiles to track. Defaults to (50, 80).

    Returns
    -------
    DataFrame with one row per n_simulations, and for each percentile: its
    mean across seeds, and the (max - min) spread across seeds — the
    practical "how much does the answer wobble" number.
    """
    n_values = n_values or [1_000, 10_000, 50_000, 100_000]
    seeds = seeds or [1, 2, 3, 4, 5]

    rows = []
    for n in n_values:
        seed_results = {p: [] for p in percentiles}
        for seed in seeds:
            values, _ = simulate_fn(df, baseline, n_simulations=n, seed=seed)
            for p in percentiles:
                seed_results[p].append(float(np.percentile(values, p)))

        row = {"n_simulations": n}
        for p in percentiles:
            arr = np.array(seed_results[p])
            row[f"p{p}_mean"] = float(arr.mean())
            row[f"p{p}_spread"] = float(arr.max() - arr.min())
        rows.append(row)

    return pd.DataFrame(rows)