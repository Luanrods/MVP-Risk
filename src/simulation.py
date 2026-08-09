"""Monte Carlo engine for cost and schedule quantitative risk analysis (QCRA/QSRA).

The core routine, ``_simulate_generic_risk``, draws occurrence (Bernoulli) and
impact magnitude (fixed / triangular / PERT) for each risk in a register and
sums them on top of a baseline value. ``simulate_cost_risk`` and
``simulate_schedule_risk`` are thin, semantically-named wrappers around it —
the underlying maths for "money" and "days" is identical.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from .distributions import sample_impact
from .validation import validate_risks

# A baseline can be a fixed number, or a dict describing an uncertainty
# distribution to sample it from on every iteration, e.g.:
#   {"distribution": "triangular", "min": 38_000_000, "most_likely": 40_000_000, "max": 43_000_000}
BaselineSpec = Union[float, dict]


def _sample_baseline(rng: np.random.Generator, baseline: BaselineSpec, size: int) -> np.ndarray:
    """Resolve a baseline spec into an array of per-iteration baseline values.

    A plain float reproduces the original v0.1–v0.3 behaviour (constant
    baseline). A dict enables optional baseline uncertainty on top of the
    risk register.
    """
    if isinstance(baseline, (int, float)):
        return np.full(size, float(baseline), dtype=float)

    if isinstance(baseline, dict):
        distribution = baseline.get("distribution", "fixed")
        minimum = float(baseline.get("min", baseline.get("most_likely")))
        most_likely = float(baseline["most_likely"])
        maximum = float(baseline.get("max", baseline.get("most_likely")))
        return sample_impact(
            rng=rng, distribution=distribution,
            minimum=minimum, most_likely=most_likely, maximum=maximum, size=size,
        )

    raise TypeError(
        "baseline must be a number or a dict with keys "
        "{'distribution', 'min', 'most_likely', 'max'}."
    )


def _correlated_uniforms(
    rng: np.random.Generator, correlation_matrix: np.ndarray, n_risks: int, size: int,
) -> np.ndarray:
    """Draw correlated Uniform(0,1) variates via a Gaussian copula.

    Used to correlate whether risks *occur* (their Bernoulli draws), not the
    magnitude of their impact given occurrence — a common, tractable
    simplification for cost/schedule risk registers where co-occurrence
    ("if the port strike happens, the customs-clearance delay is more likely
    too") is the relationship that matters most in practice.
    """
    correlation_matrix = np.asarray(correlation_matrix, dtype=float)
    if correlation_matrix.shape != (n_risks, n_risks):
        raise ValueError(
            f"correlation_matrix must be {n_risks}x{n_risks} (one row/col per risk), "
            f"got {correlation_matrix.shape}."
        )
    try:
        cholesky = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "correlation_matrix must be symmetric positive semi-definite "
            "(a valid correlation matrix)."
        ) from exc

    z = rng.standard_normal(size=(size, n_risks))
    correlated_z = z @ cholesky.T
    from scipy.stats import norm
    return norm.cdf(correlated_z)


def _simulate_generic_risk(
    df: pd.DataFrame,
    baseline: BaselineSpec,
    n_simulations: int = 10_000,
    seed: int = 42,
    correlation_matrix: Optional[np.ndarray] = None,
):
    """Shared Monte Carlo core used by both cost and schedule simulation.

    Returns (final_value, contribution_df) where final_value is baseline +
    sum of signed risk contributions, and contribution_df holds each risk's
    per-iteration contribution (useful for driver/sensitivity analysis).
    """
    errors = validate_risks(df)
    if errors:
        raise ValueError(" | ".join(errors))

    rng = np.random.default_rng(seed)

    if len(df) == 0:
        final_value = _sample_baseline(rng, baseline, n_simulations)
        return final_value, pd.DataFrame(index=range(n_simulations))

    n_risks = len(df)
    contributions = np.zeros((n_simulations, n_risks), dtype=float)

    if correlation_matrix is not None:
        occurrence_draws = _correlated_uniforms(rng, correlation_matrix, n_risks, n_simulations)
    else:
        occurrence_draws = rng.random(size=(n_simulations, n_risks))

    for j, (_, risk) in enumerate(df.iterrows()):
        occurs = occurrence_draws[:, j] < float(risk["probability"])
        impacts = sample_impact(
            rng=rng, distribution=str(risk["distribution"]),
            minimum=float(risk["min_impact"]),
            most_likely=float(risk["most_likely_impact"]),
            maximum=float(risk["max_impact"]), size=n_simulations,
        )
        sign = 1.0 if risk["type"] == "risco" else -1.0
        contributions[:, j] = occurs * impacts * sign

    baseline_draws = _sample_baseline(rng, baseline, n_simulations)
    final_value = baseline_draws + contributions.sum(axis=1)
    contribution_df = pd.DataFrame(contributions, columns=df["id"].astype(str).tolist())
    return final_value, contribution_df


def simulate_cost_risk(
    df: pd.DataFrame,
    baseline_cost: BaselineSpec,
    n_simulations: int = 10_000,
    seed: int = 42,
    correlation_matrix: Optional[np.ndarray] = None,
):
    """Run a Monte Carlo cost risk simulation.

    Parameters
    ----------
    df : DataFrame
        Risk register (see src/validation.py for required columns).
    baseline_cost : float or dict
        Fixed baseline cost, or a dict describing a distribution for baseline
        uncertainty, e.g. {"distribution": "pert", "min": .., "most_likely": .., "max": ..}.
    n_simulations : int
        Number of Monte Carlo iterations.
    seed : int
        Seed for reproducibility.
    correlation_matrix : array-like, optional
        n_risks x n_risks correlation matrix (same row/col order as df) used
        to correlate *occurrence* of risks via a Gaussian copula. None (the
        default) reproduces the original independent-risks behaviour.

    Returns
    -------
    (final_cost, contribution_df)
    """
    return _simulate_generic_risk(
        df=df, baseline=baseline_cost, n_simulations=n_simulations,
        seed=seed, correlation_matrix=correlation_matrix,
    )


def simulate_schedule_risk(
    df: pd.DataFrame,
    baseline_duration: BaselineSpec,
    n_simulations: int = 10_000,
    seed: int = 42,
    correlation_matrix: Optional[np.ndarray] = None,
):
    """Run a Monte Carlo schedule (duration) risk simulation.

    Same engine and same risk-register schema as ``simulate_cost_risk`` —
    impacts are read as days instead of currency. This is a schedule-risk
    *overlay* (each row is an independent delay/acceleration driver), not a
    full Critical Path Method (CPM) network simulation: it does not model
    activity dependencies, float, or path convergence. Treat it as a QSRA-lite
    view of duration risk exposure, not a replacement for a scheduling tool.

    Parameters mirror ``simulate_cost_risk``, with ``baseline_duration`` (in
    days) in place of ``baseline_cost``.

    Returns
    -------
    (final_duration, contribution_df)
    """
    return _simulate_generic_risk(
        df=df, baseline=baseline_duration, n_simulations=n_simulations,
        seed=seed, correlation_matrix=correlation_matrix,
    )
