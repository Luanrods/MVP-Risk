import pandas as pd
from src.simulation import simulate_cost_risk

df = pd.DataFrame([
    {"id": "R01", "type": "risco", "description": "t", "probability": 1.0, "distribution": "fixed",
     "min_impact": 100.0, "most_likely_impact": 100.0, "max_impact": 100.0},
    {"id": "O01", "type": "oportunidade", "description": "t", "probability": 1.0, "distribution": "fixed",
     "min_impact": 40.0, "most_likely_impact": 40.0, "max_impact": 40.0},
])
cost, _ = simulate_cost_risk(df, baseline_cost=1000.0, n_simulations=10000, seed=42)
print("Esperado: 1060.0 | Obtido:", cost.mean())