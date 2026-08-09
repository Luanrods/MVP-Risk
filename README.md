# MVP-Risk

[![Tests](https://github.com/Luanrods/MVP-Risk/actions/workflows/tests.yml/badge.svg)](https://github.com/Luanrods/MVP-Risk/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An open-source Monte Carlo engine for **Quantitative Cost & Schedule Risk Analysis (QCRA/QSRA)**. Feed it a risk register, get back a full probabilistic view — P50/P80/P90, an S-curve, and risk driver ranking — as a PDF report or in an interactive Streamlit app.

Built for EPC / infrastructure projects, where "add 10% contingency" is still the norm and a proper probabilistic view is rarely available.

## What it does

1. Reads a risk register (CSV): each row is a risk or opportunity with a probability of occurrence and an impact range.
2. Runs a Monte Carlo simulation combining:
   - **Bernoulli sampling** for whether each risk occurs
   - **Triangular or PERT distributions** for impact magnitude when it does
   - Optionally, **correlated occurrence** between related risks (Gaussian copula)
   - Optionally, **baseline uncertainty** (the baseline itself as a distribution, not just a fixed number)
3. Aggregates results into a final distribution and computes:
   - Mean, standard deviation, P50/P80/P90/P95
   - P80 contingency vs. baseline
   - Probability of finishing within budget / target duration
   - Risk driver ranking (Spearman correlation between each risk's contribution and the final result)
   - EMV (Expected Monetary Value) as a deterministic reference point
   - A convergence diagnostic to check whether your chosen iteration count is stable enough
4. The same engine drives two use cases: **cost risk** (money) and **schedule risk** (days) — see [Schedule risk](#schedule-risk-qsra-lite) below.
5. Outputs either a formatted PDF report (`build_report.py`) or an interactive web app (`app.py`).

## Project structure

```
MVP-Risk/
├── data/
│   ├── example_risk_register.csv      # sample cost risk register
│   └── example_schedule_register.csv  # sample schedule risk register
├── src/
│   ├── simulation.py                   # Monte Carlo engine (cost + schedule)
│   ├── distributions.py                # PERT / triangular / fixed sampling
│   ├── metrics.py                      # percentiles, EMV, drivers, convergence
│   ├── charts.py                       # matplotlib figures
│   └── validation.py                   # risk register schema checks
├── tests/                              # pytest suite (33 tests)
├── examples/
│   └── scratch_manual_check.py         # manual sanity check, not part of pytest
├── .github/workflows/tests.yml         # CI: runs pytest on every push/PR
├── outputs/                            # generated charts + PDF land here
├── app.py                              # Streamlit interactive app
├── build_report.py                     # CLI — generates the PDF report
├── pyproject.toml                      # makes the project pip-installable
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

### Option A — PDF report (CLI)

```bash
python build_report.py
```

Reads `data/example_risk_register.csv`, runs the simulation, and writes `outputs/MVP-Risk.pdf` plus three PNGs.

To analyze your own project, pass your own risk register and assumptions as CLI arguments — no need to edit any source file:

```bash
python build_report.py \
  --input data/my_project_risks.csv \
  --baseline-cost 40000000 \
  --budget 44000000 \
  --n-simulations 100000 \
  --seed 42 \
  --output-dir outputs
```

Run `python build_report.py --help` for the full list of options.

### Option B — Interactive app (Streamlit)

```bash
streamlit run app.py
```

Upload a risk register (or use the bundled example), adjust assumptions in the sidebar — including baseline uncertainty and correlation between two risks — and get live charts and KPIs. Switch between **Cost (QCRA)** and **Schedule (QSRA-lite)** mode from the sidebar.

### Option C — Install as a package

```bash
pip install -e .
mvp-risk --help
```

## Risk register format

| Column | Description |
|---|---|
| `id` | Unique risk ID |
| `type` | `risco` (adds cost/time) or `oportunidade` (reduces cost/time) |
| `description` | Free text |
| `probability` | Probability of occurrence, 0–1 |
| `distribution` | `fixed`, `triangular`, or `pert` |
| `min_impact` / `most_likely_impact` / `max_impact` | Impact magnitude (always positive — `type` controls the sign). Currency for cost analysis, days for schedule analysis. |

## Schedule risk (QSRA-lite)

`simulate_schedule_risk` reuses the exact same Monte Carlo engine as cost risk, just reading impacts as days instead of currency (`data/example_schedule_register.csv` is a ready-to-use example). This is a **schedule-risk overlay** — each row is treated as an independent delay/acceleration driver — not a full Critical Path Method (CPM) network simulation. It does not model activity dependencies, float, or path convergence. Useful as a fast probabilistic view of duration exposure; not a replacement for a scheduling tool.

## Correlated risks & baseline uncertainty

By default risks are independent and the baseline is a fixed number — matching a classic risk register. Two opt-in features go further:

- **Correlated occurrence**: pass a `correlation_matrix` (n_risks × n_risks) to `simulate_cost_risk` / `simulate_schedule_risk` to correlate whether related risks occur together, via a Gaussian copula. The Streamlit app exposes this for a pair of risks in the sidebar.
- **Baseline uncertainty**: pass a dict instead of a number for `baseline_cost` / `baseline_duration`, e.g. `{"distribution": "pert", "min": 38_000_000, "most_likely": 40_000_000, "max": 43_000_000}`, to sample the baseline itself instead of holding it fixed.

## Convergence diagnostics

`src/metrics.convergence_report()` runs the simulation across multiple sample sizes and seeds and reports how much your P50/P80 estimates wobble at each size — evidence for whether your chosen `--n-simulations` is actually enough:

```python
from src.metrics import convergence_report
from src.simulation import simulate_cost_risk
import pandas as pd

risks = pd.read_csv("data/example_risk_register.csv")
report = convergence_report(simulate_cost_risk, risks, baseline=40_000_000.0)
print(report)
```

## Tests

```bash
pytest
```

33 tests covering the simulation engine (independence, correlation, baseline uncertainty, schedule mode), distributions, validation, and metrics. CI runs this matrix (Python 3.11/3.12) on every push via GitHub Actions.

## Model assumptions & limitations

- Without an explicit `correlation_matrix`, risks are simulated as independent.
- Probability is the chance of occurrence within the analysis horizon; the distribution describes impact magnitude *conditional* on occurrence.
- Baseline uncertainty is off by default — only the risk register's events are modeled unless you opt in.
- Schedule mode is an overlay of independent duration drivers, not a CPM network.
- A fixed seed makes a run reproducible, not predictive.
- Results are only as good as the probability/impact estimates that go into the register.

## Status

MVP under active development, built and maintained by [Luan Garcia Rodrigues](https://www.linkedin.com/) as part of the **Grid PM** project — open-source tooling for data-driven project & risk management in infrastructure. See the [roadmap](ROADMAP.md) for what's next.

## License

[MIT](LICENSE)
