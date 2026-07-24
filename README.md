# Algothon 2026

Research, strategy development, backtesting, and visualisation for the SIG × UNSW FinTech Society Algothon 2026.

## Setup on macOS

### 1. Check Python

Python 3.12 is recommended for matching the competition environment.

```bash
python3 --version
```

If Python is not installed, install it with Homebrew:

```bash
brew install python@3.12
```

### 2. Create a virtual environment

From the repository's root directory, run:

```bash
python3.12 -m venv .venv
```

If `python3.12` is unavailable but `python3` reports an appropriate version, use:

```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

Your terminal prompt should now begin with `(.venv)`.

### 4. Install the development packages

First update `pip`, then install the standard analysis and dashboard packages:

```bash
python -m pip install --upgrade pip
python -m pip install numpy matplotlib scipy scikit-learn pandas streamlit
```

The package is installed as `scikit-learn`, but imported in Python as `sklearn`.

### 5. Verify the installation

```bash
python -c "import numpy, matplotlib, scipy, sklearn, pandas, streamlit; print('Setup complete')"
```

## Running the project

Run the official evaluator from the repository root:

```bash
python backtesting/eval.py
```

Launch the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

Streamlit should open the dashboard in your default browser. If it does not, open the local URL printed in the terminal.

## Leaving and returning to the environment

Deactivate the environment when finished:

```bash
deactivate
```

Activate it again in a new terminal session with:

```bash
source .venv/bin/activate
```

## Optional: save exact package versions

After the environment is working, record its installed versions:

```bash
python -m pip freeze > requirements-dev.txt
```

The competition submission should contain only the self-contained team Python file and, if required, a separate `requirements.txt` listing packages outside the competition's standard environment.

IF in root folder, run:
python3 04-backtest/backtester.py --prices 01-data/prices.txt --strategy 03-strategy/strategy.py --start-day 1 --end-day 499