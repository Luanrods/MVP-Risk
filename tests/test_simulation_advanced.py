import numpy as np
import pandas as pd
import pytest

from src.simulation import simulate_cost_risk, simulate_schedule_risk


def one_risk(**overrides):
    row = {
        "id": "R01", "type": "risco", "description": "Test risk",
        "probability": 1.0, "distribution": "fixed",
        "min_impact": 100.0, "most_likely_impact": 100.0, "max_impact": 100.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def two_risks():
    return pd.DataFrame([
        {"id": "R01", "type": "risco", "description": "A", "probability": 0.5,
         "distribution": "fixed", "min_impact": 100.0, "most_likely_impact": 100.0, "max_impact": 100.0},
        {"id": "R02", "type": "risco", "description": "B", "probability": 0.5,
         "distribution": "fixed", "min_impact": 100.0, "most_likely_impact": 100.0, "max_impact": 100.0},
    ])


# --- Baseline uncertainty -------------------------------------------------

def test_scalar_baseline_still_works():
    df = one_risk(probability=0.0)
    cost, _ = simulate_cost_risk(df, baseline_cost=1000.0, seed=1)
    assert np.allclose(cost, 1000.0)


def test_dict_baseline_adds_variance():
    df = one_risk(probability=0.0)
    baseline_spec = {"distribution": "triangular", "min": 900.0, "most_likely": 1000.0, "max": 1200.0}
    cost, _ = simulate_cost_risk(df, baseline_cost=baseline_spec, n_simulations=5000, seed=1)
    assert cost.std() > 0
    assert 900.0 <= cost.min() and cost.max() <= 1200.0


def test_invalid_baseline_type_raises():
    df = one_risk()
    with pytest.raises(TypeError):
        simulate_cost_risk(df, baseline_cost="not a number", seed=1)


# --- Correlated risks ------------------------------------------------------

def test_correlation_matrix_wrong_shape_raises():
    df = two_risks()
    bad_matrix = np.eye(3)  # 2 risks, 3x3 matrix -> mismatch
    with pytest.raises(ValueError):
        simulate_cost_risk(df, baseline_cost=1000.0, n_simulations=100, seed=1,
                            correlation_matrix=bad_matrix)


def test_correlation_matrix_not_psd_raises():
    df = two_risks()
    invalid_matrix = np.array([[1.0, 2.0], [2.0, 1.0]])  # not a valid correlation matrix
    with pytest.raises(ValueError):
        simulate_cost_risk(df, baseline_cost=1000.0, n_simulations=100, seed=1,
                            correlation_matrix=invalid_matrix)


def test_positive_correlation_increases_joint_occurrence():
    """With strong positive correlation, both risks should occur together
    far more often than under independence (probability=0.5 each)."""
    df = two_risks()

    _, contrib_indep = simulate_cost_risk(
        df, baseline_cost=0.0, n_simulations=20_000, seed=1,
    )
    both_occur_indep = ((contrib_indep["R01"] != 0) & (contrib_indep["R02"] != 0)).mean()

    corr = np.array([[1.0, 0.9], [0.9, 1.0]])
    _, contrib_corr = simulate_cost_risk(
        df, baseline_cost=0.0, n_simulations=20_000, seed=1, correlation_matrix=corr,
    )
    both_occur_corr = ((contrib_corr["R01"] != 0) & (contrib_corr["R02"] != 0)).mean()

    # Independent: ~0.25. Correlated: should be noticeably higher.
    assert both_occur_corr > both_occur_indep + 0.1


# --- Schedule risk -----------------------------------------------------

def test_schedule_risk_uses_same_engine_semantics():
    df = one_risk(probability=1.0, min_impact=10.0, most_likely_impact=10.0, max_impact=10.0)
    duration, _ = simulate_schedule_risk(df, baseline_duration=180.0, seed=1)
    assert np.allclose(duration, 190.0)


def test_schedule_opportunity_reduces_duration():
    df = one_risk(type="oportunidade", probability=1.0,
                   min_impact=5.0, most_likely_impact=5.0, max_impact=5.0)
    duration, _ = simulate_schedule_risk(df, baseline_duration=180.0, seed=1)
    assert np.allclose(duration, 175.0)
