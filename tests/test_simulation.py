import numpy as np
import pandas as pd
from src.simulation import simulate_cost_risk

def one_risk(**overrides):
    row = {
        "id": "R01", "type": "threat", "description": "Test risk",
        "probability": 1.0, "distribution": "fixed",
        "min_impact": 100.0, "most_likely_impact": 100.0, "max_impact": 100.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])

def test_zero_probability_has_zero_effect():
    df = one_risk(probability=0.0)
    cost, _ = simulate_cost_risk(df, baseline_cost=1000.0, seed=1)
    assert np.allclose(cost, 1000.0)

def test_fixed_certain_threat_adds_cost():
    df = one_risk(probability=1.0)
    cost, _ = simulate_cost_risk(df, baseline_cost=1000.0, seed=1)
    assert np.allclose(cost, 1100.0)

def test_fixed_certain_opportunity_reduces_cost():
    df = one_risk(type="opportunity", probability=1.0)
    cost, _ = simulate_cost_risk(df, baseline_cost=1000.0, seed=1)
    assert np.allclose(cost, 900.0)

def test_seed_is_reproducible():
    df = one_risk(probability=0.4, distribution="triangular",
                   min_impact=50.0, most_likely_impact=100.0, max_impact=200.0)
    a, _ = simulate_cost_risk(df, 1000.0, seed=123)
    b, _ = simulate_cost_risk(df, 1000.0, seed=123)
    assert np.array_equal(a, b)