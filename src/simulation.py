from __future__ import annotations
import numpy as np
import pandas as pd
from .distributions import sample_impact
from .validation import validate_risks

def simulate_cost_risk(df, baseline_cost, n_simulations=10_000, seed=42):
    errors = validate_risks(df)
    if errors:
        raise ValueError(" | ".join(errors))

    if len(df) == 0:
        final_cost = np.full(n_simulations, baseline_cost, dtype=float)
        return final_cost, pd.DataFrame(index=range(n_simulations))

    rng = np.random.default_rng(seed)
    contributions = np.zeros((n_simulations, len(df)), dtype=float)

    for j, (_, risk) in enumerate(df.iterrows()):
        occurs = rng.random(n_simulations) < float(risk["probability"])
        impacts = sample_impact(
            rng=rng, distribution=str(risk["distribution"]),
            minimum=float(risk["min_impact"]),
            most_likely=float(risk["most_likely_impact"]),
            maximum=float(risk["max_impact"]), size=n_simulations,
        )
        sign = 1.0 if risk["type"] == "risco" else -1.0
        contributions[:, j] = occurs * impacts * sign

    final_cost = baseline_cost + contributions.sum(axis=1)
    contribution_df = pd.DataFrame(contributions, columns=df["id"].astype(str).tolist())
    return final_cost, contribution_df