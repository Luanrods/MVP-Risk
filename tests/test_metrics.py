import numpy as np
import pandas as pd

from src.metrics import summary_metrics, add_emv, risk_driver_table, convergence_report
from src.simulation import simulate_cost_risk


def test_summary_metrics_basic():
    values = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    metrics = summary_metrics(values, baseline_cost=100.0, budget=350.0)
    assert metrics["mean"] == 300.0
    assert metrics["p50"] == 300.0
    assert 0.0 <= metrics["prob_within_budget"] <= 1.0


def test_summary_metrics_without_budget_omits_prob():
    values = np.array([100.0, 200.0, 300.0])
    metrics = summary_metrics(values, baseline_cost=100.0)
    assert "prob_within_budget" not in metrics


def test_add_emv_signs_opportunities_negative():
    df = pd.DataFrame([
        {"id": "R01", "type": "risco", "probability": 0.5, "distribution": "fixed",
         "min_impact": 100.0, "most_likely_impact": 100.0, "max_impact": 100.0},
        {"id": "O01", "type": "oportunidade", "probability": 0.5, "distribution": "fixed",
         "min_impact": 100.0, "most_likely_impact": 100.0, "max_impact": 100.0},
    ])
    out = add_emv(df)
    risk_emv = out.loc[out["id"] == "R01", "signed_emv"].iloc[0]
    opp_emv = out.loc[out["id"] == "O01", "signed_emv"].iloc[0]
    assert risk_emv > 0
    assert opp_emv < 0


def test_risk_driver_table_ranks_higher_impact_risk_first():
    df = pd.DataFrame([
        {"id": "BIG", "type": "risco", "description": "Big risk", "probability": 0.5,
         "distribution": "fixed", "min_impact": 1000.0, "most_likely_impact": 1000.0, "max_impact": 1000.0},
        {"id": "SMALL", "type": "risco", "description": "Small risk", "probability": 0.5,
         "distribution": "fixed", "min_impact": 10.0, "most_likely_impact": 10.0, "max_impact": 10.0},
    ])
    final_cost, contributions = simulate_cost_risk(df, baseline_cost=0.0, n_simulations=5000, seed=1)
    drivers = risk_driver_table(contributions, final_cost)
    assert drivers.iloc[0]["risk_id"] == "BIG"


def test_convergence_report_shrinks_with_more_simulations():
    df = pd.DataFrame([
        {"id": "R01", "type": "risco", "description": "Test risk", "probability": 0.4,
         "distribution": "triangular", "min_impact": 50.0, "most_likely_impact": 100.0, "max_impact": 300.0},
    ])
    report = convergence_report(
        simulate_cost_risk, df, baseline=1000.0,
        n_values=[500, 20_000], seeds=[1, 2, 3, 4],
    )
    # p50 can land exactly on the "risk did not occur" baseline for every seed
    # when probability < 0.5 (discrete threshold), so it's not a reliable
    # convergence signal here — p80 is continuous and does the job.
    small_n_spread = report.loc[report["n_simulations"] == 500, "p80_spread"].iloc[0]
    large_n_spread = report.loc[report["n_simulations"] == 20_000, "p80_spread"].iloc[0]
    assert large_n_spread < small_n_spread
