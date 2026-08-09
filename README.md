# MVP-Risk

A lightweight Monte Carlo engine for **Quantitative Cost Risk Analysis (QCRA)** on capital projects. Feed it a risk register, get back a full probabilistic cost report — P50/P80/P90, an S-curve, and Spearman-based risk driver ranking — as a formatted PDF.

Built for EPC / infrastructure projects, where "add 10% contingency" is still the norm and a proper probabilistic view is rarely available.

## What it does

1. Reads a risk register (CSV): each row is a risk or opportunity with a probability of occurrence and an impact range.
2. Runs a Monte Carlo simulation (default: 100,000 iterations) combining:
   - **Bernoulli sampling** for whether each risk occurs
   - **Triangular or PERT distributions** for impact magnitude when it does
3. Aggregates results into a final cost distribution and computes:
   - Mean, standard deviation, P50/P80/P90/P95
   - P80 contingency vs. baseline
   - Probability of finishing within budget
   - Risk driver ranking (Spearman correlation between each risk's contribution and total cost)
   - EMV (Expected Monetary Value) as a deterministic reference point
4. Exports three charts (histogram, S-curve, driver ranking) and assembles everything into a PDF report.

## Project structure

```
MVP-Risk/
├── data/
│   └── example_risk_register.csv   # sample risk register
├── src/
│   ├── simulation.py                # Monte Carlo engine
│   ├── distributions.py             # PERT / triangular / fixed sampling
│   ├── metrics.py                   # percentiles, EMV, risk drivers
│   ├── charts.py                    # matplotlib figures
│   └── validation.py                # risk register schema checks
├── tests/
│   └── test_simulation.py           # pytest suite
├── outputs/                         # generated charts + PDF land here
├── build_report.py                  # entry point — run this
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/Luanrods/MVP-Risk.git
cd MVP-Risk
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows (PowerShell)
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Run the sample case:

```bash
python build_report.py
```

This reads `data/example_risk_register.csv`, runs the simulation, and writes `outputs/MVP-Risk.pdf` plus the three supporting PNGs.

To analyze your own project, replace `data/example_risk_register.csv` with your own risk register (same columns) and adjust `BASELINE_COST`, `BUDGET`, `N_SIMULATIONS`, and `SEED` at the top of `build_report.py`.

## Risk register format

| Column | Description |
|---|---|
| `id` | Unique risk ID |
| `type` | `risco` (adds cost) or `oportunidade` (reduces cost) |
| `description` | Free text |
| `probability` | Probability of occurrence, 0–1 |
| `distribution` | `fixed`, `triangular`, or `pert` |
| `min_impact` / `most_likely_impact` / `max_impact` | Impact magnitude (always positive — `type` controls the sign) |

## Tests

```bash
pytest
```

## Model assumptions & limitations (v0.1)

- Risks are simulated as independent — no correlation between events yet.
- Probability is the chance of occurrence within the analysis horizon; the distribution describes impact magnitude *conditional* on occurrence.
- Baseline cost uncertainty is not modeled — only the risk register's events.
- A fixed seed makes the run reproducible, not predictive.
- Results are only as good as the probability/impact estimates that go into the register.

## Status

Early-stage MVP (v0.1), built and maintained by [Luan Garcia Rodrigues](https://www.linkedin.com/) as part of the **Grid PM** project — open-source tooling for data-driven project & risk management in infrastructure. See the [roadmap](ROADMAP.md) for what's next.

## License

MIT (or your preferred license — not yet added, see roadmap).
