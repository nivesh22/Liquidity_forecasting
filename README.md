Liquidity: Forecasting Pipelines & Dashboard
==========================================

Overview
--------
- End-to-end Kedro project that ingests dummy datasets (clients, EOD balances, remuneration, account types), processes and aggregates deposits, and produces a simple country-level forecast using interest rates.
- A Streamlit dashboard renders the processed aggregates and forecast results with a custom theme and consistent styling.

Highlights
----------
- Modular, production-style Kedro pipelines with parameters.
- Forecasting at country level based on historical deposits and provided/observed interest rates (simple elasticity model).
- Streamlit dashboard with sidebar filters, sectioned charts, and consistent theme.
- Ruff-compliant code with docstrings and unit tests targeting 100% coverage for all node functions.

Project Layout
--------------
- `src/liquidity/`: Source package, pipelines, and registry.
- `conf/base/`: Kedro catalog and parameters.
- `data/`: Dummy data (raw) and pipeline outputs.
- `app/streamlit_app.py`: Dashboard app.
- `.streamlit/config.toml`: Theme config.
- `tests/`: Unit tests for nodes with full coverage.

Getting Started
---------------
1) Install dependencies (suggested):
   - Create a virtual env, then: `pip install -e .` or `pip install -r requirements.txt` (if you generate one).
   - Ensure Python >= 3.9.

2) Run the Kedro pipeline:
   - `kedro run` (from project root). Outputs will be written to `data/08_reporting`.

3) Launch the Streamlit app:
   - `streamlit run app/streamlit_app.py`

Parameters
----------
- Edit `conf/base/parameters.yml` to tweak forecast horizon and default average interest rate (per-country overrides supported).

Notes
-----
- The forecasting logic uses a simple linear elasticity model on log-deposits vs. interest rate to keep dependencies light and code transparent. It’s suitable for showcasing workflow and structure.
- Replace dummy CSVs in `data/01_raw` with real datasets to run the project with your data.
