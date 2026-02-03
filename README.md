# Stochastic Lotka-Volterra Dynamics in Macroeconomics: A Bounded Goodwin Wage-Employment Cycle

Port of Goodwin/LV bounded model with deterministic & stochastic simulation,
MLE parameter estimation, diagnostics and plotting.

## Structure
- `goodwin/core.py` — main functions: transforms, sims, estimation, plotting
- `scripts/run_analysis.py` — example runner (reads CSVs from `data/`)
- `requirements.txt` — Python deps

## Quickstart
1. Create virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate       # macOS/Linux
   .venv\Scripts\activate          # Windows (PowerShell)
   pip install -r requirements.txt


