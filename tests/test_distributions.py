import numpy as np
import pytest

from src.distributions import sample_impact, sample_pert


def test_fixed_returns_constant():
    rng = np.random.default_rng(1)
    out = sample_impact(rng, "fixed", minimum=10, most_likely=50, maximum=90, size=100)
    assert np.all(out == 50)


def test_triangular_stays_within_bounds():
    rng = np.random.default_rng(1)
    out = sample_impact(rng, "triangular", minimum=10, most_likely=50, maximum=90, size=5000)
    assert out.min() >= 10 and out.max() <= 90


def test_triangular_degenerate_case_returns_constant():
    rng = np.random.default_rng(1)
    out = sample_impact(rng, "triangular", minimum=50, most_likely=50, maximum=50, size=100)
    assert np.all(out == 50)


def test_pert_stays_within_bounds():
    rng = np.random.default_rng(1)
    out = sample_impact(rng, "pert", minimum=10, most_likely=50, maximum=90, size=5000)
    assert out.min() >= 10 and out.max() <= 90


def test_pert_degenerate_case_returns_constant():
    rng = np.random.default_rng(1)
    out = sample_pert(rng, minimum=50, most_likely=50, maximum=50, size=100)
    assert np.all(out == 50)


def test_pert_mode_pulls_distribution_toward_most_likely():
    rng = np.random.default_rng(1)
    out = sample_impact(rng, "pert", minimum=0, most_likely=10, maximum=100, size=20000)
    # PERT concentrates mass near most_likely; mean should sit well below the midpoint (50).
    assert out.mean() < 30


def test_unsupported_distribution_raises():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        sample_impact(rng, "gaussian", minimum=0, most_likely=1, maximum=2, size=10)
