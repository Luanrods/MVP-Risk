# src/validation.py
from __future__ import annotations
import pandas as pd


# O REQUIRED_COLUMNS is a set of column names that must be present in the input DataFrame 
# for risk validation. Se o dataframe tiver mais colunas, não há problema, mas se faltar alguma
# dessas colunas, a validação falhará.

REQUIRED_COLUMNS = {
    "id", "type", "description", "probability", "distribution",
    "min_impact", "most_likely_impact", "max_impact",
}

def validate_risks(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return [f"Missing columns: {sorted(missing)}"]

    if df["id"].duplicated().any():
        errors.append("Risk IDs must be unique.")
    if (~df["probability"].between(0, 1)).any():
        errors.append("Probability must be between 0 and 1.")
    if (~df["type"].isin(["threat", "opportunity"])).any():
        errors.append("Type must be threat or opportunity.")
    if (~df["distribution"].isin(["fixed", "triangular", "pert"])).any():
        errors.append("Unsupported distribution found.")

    bad_order = ~(
        (df["min_impact"] <= df["most_likely_impact"])
        & (df["most_likely_impact"] <= df["max_impact"])
    )
    if bad_order.any():
        errors.append("Each row must satisfy min <= most_likely <= max.")

    impact_cols = ["min_impact", "most_likely_impact", "max_impact"]
    if (df[impact_cols] < 0).any().any():
        errors.append("Enter positive impact magnitudes; type controls the sign.")

    pert_rows = df[df["distribution"] == "pert"]
    if not pert_rows.empty:
        inconsistent = (
            (pert_rows["min_impact"] == pert_rows["max_impact"])
            & (pert_rows["most_likely_impact"] != pert_rows["min_impact"])
        )
        if inconsistent.any():
            bad_ids = pert_rows.loc[inconsistent, "id"].tolist()
            errors.append(
                f"PERT rows with min_impact == max_impact must also have "
                f"most_likely_impact equal to that value: {bad_ids}"
            )

    return errors