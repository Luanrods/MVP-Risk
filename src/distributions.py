from __future__ import annotations
import numpy as np

def sample_pert(rng, minimum, most_likely, maximum, size, lambd=4.0):
    if maximum == minimum:
        return np.full(size, minimum, dtype=float)
    alpha = 1.0 + lambd * (most_likely - minimum) / (maximum - minimum)
    beta = 1.0 + lambd * (maximum - most_likely) / (maximum - minimum)
    draws = rng.beta(alpha, beta, size=size)
    return minimum + draws * (maximum - minimum)

def sample_impact(rng, distribution, minimum, most_likely, maximum, size):
    if distribution == "fixed":
        return np.full(size, most_likely, dtype=float)
    if distribution == "triangular":
        if minimum == maximum:
            return np.full(size, minimum, dtype=float)
        return rng.triangular(minimum, most_likely, maximum, size=size)
    if distribution == "pert":
        return sample_pert(rng, minimum, most_likely, maximum, size)
    raise ValueError(f"Unsupported distribution: {distribution}")