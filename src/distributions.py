"""Sampling helpers for risk impact magnitudes."""
from __future__ import annotations

import numpy as np


def sample_pert(
    rng: np.random.Generator,
    minimum: float,
    most_likely: float,
    maximum: float,
    size: int,
    lambd: float = 4.0,
) -> np.ndarray:
    """Sample from a Modified PERT distribution via a rescaled Beta.

    ``lambd`` controls how sharply the distribution concentrates around
    ``most_likely`` (4.0 is the standard PERT shape parameter).
    """
    if maximum == minimum:
        return np.full(size, minimum, dtype=float)
    alpha = 1.0 + lambd * (most_likely - minimum) / (maximum - minimum)
    beta = 1.0 + lambd * (maximum - most_likely) / (maximum - minimum)
    draws = rng.beta(alpha, beta, size=size)
    return minimum + draws * (maximum - minimum)


def sample_impact(
    rng: np.random.Generator,
    distribution: str,
    minimum: float,
    most_likely: float,
    maximum: float,
    size: int,
) -> np.ndarray:
    """Draw ``size`` impact magnitudes for one risk under the given distribution.

    Supported distributions: "fixed" (always ``most_likely``), "triangular",
    and "pert". Raises ValueError for anything else.
    """
    if distribution == "fixed":
        return np.full(size, most_likely, dtype=float)
    if distribution == "triangular":
        if minimum == maximum:
            return np.full(size, minimum, dtype=float)
        return rng.triangular(minimum, most_likely, maximum, size=size)
    if distribution == "pert":
        return sample_pert(rng, minimum, most_likely, maximum, size)
    raise ValueError(f"Unsupported distribution: {distribution}")
