# Algothon 2026

Research, strategy development, backtesting, and visualisation for the SIG × UNSW FinTech Society Algothon 2026.

## Structure

```text
.
├── 01-data/prices.txt
├── 02-analysis/{da.ipynb, utils.py}
├── 03-strategy/strategy.py
├── 04-backtest/{backtester.py, dashboard.py, eval.py}
├── 05-submission/submission.py
├── .venv/
├── requirements-dev.txt
└── README.md
```

## Setup

```bash
python3.12 -m venv .venv  # Use python3 if Python 3.12 is already the default.
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn matplotlib plotly streamlit
```

## Run the backtester

```bash
python 04-backtest/backtester.py --prices 01-data/prices.txt --strategy 03-strategy/strategy.py --start-day 1 --end-day 499
```

## Run the dashboard

```bash
python -m streamlit run 04-backtest/dashboard.py
```

Run all commands from the repository root. The dashboard opens in your browser.
