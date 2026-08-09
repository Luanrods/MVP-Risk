import pandas as pd

from src.validation import validate_risks


def valid_df(**overrides):
    row = {
        "id": "R01", "type": "risco", "description": "Test risk",
        "probability": 0.5, "distribution": "triangular",
        "min_impact": 10.0, "most_likely_impact": 20.0, "max_impact": 30.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_valid_register_has_no_errors():
    assert validate_risks(valid_df()) == []


def test_missing_column_detected():
    df = valid_df().drop(columns=["probability"])
    errors = validate_risks(df)
    assert any("Missing columns" in e for e in errors)


def test_duplicate_ids_detected():
    df = pd.concat([valid_df(), valid_df()], ignore_index=True)
    errors = validate_risks(df)
    assert any("unique" in e for e in errors)


def test_probability_out_of_range_detected():
    df = valid_df(probability=1.5)
    errors = validate_risks(df)
    assert any("Probability" in e for e in errors)


def test_invalid_type_detected():
    df = valid_df(type="ameaca")
    errors = validate_risks(df)
    assert any("Type must be" in e for e in errors)


def test_invalid_distribution_detected():
    df = valid_df(distribution="normal")
    errors = validate_risks(df)
    assert any("Unsupported distribution" in e for e in errors)


def test_impact_order_violation_detected():
    df = valid_df(min_impact=50.0, most_likely_impact=20.0, max_impact=30.0)
    errors = validate_risks(df)
    assert any("min <= most_likely <= max" in e for e in errors)


def test_negative_impact_detected():
    df = valid_df(min_impact=-10.0)
    errors = validate_risks(df)
    assert any("positive impact" in e for e in errors)


def test_pert_min_equals_max_requires_matching_most_likely():
    df = valid_df(distribution="pert", min_impact=50.0, most_likely_impact=20.0, max_impact=50.0)
    errors = validate_risks(df)
    assert any("PERT rows" in e for e in errors)
