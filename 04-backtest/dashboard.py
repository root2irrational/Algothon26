#!/usr/bin/env python3
"""Single-file, evaluator-compatible Algothon 2026 Streamlit dashboard.

This module deliberately preserves the official evaluator's accounting:

* ``getMyPosition(prcSoFar)`` receives a NumPy array shaped
  ``(n_instruments, n_history_days)``.
* The strategy trades at the latest observed price.
* Positions are clipped to integer share limits derived from dollar limits.
* Commission created by a trade is charged at the *next* valuation step.
* The first strategy decision is made one day before the first scored PnL day.
* The final loop iteration only marks existing positions; it does not trade.
* Daily PnL volatility uses ``numpy.std(..., ddof=0)``.

Compared with the official evaluator, this implementation additionally records
per-instrument PnL, orders, positions, commissions, exposures, drawdowns,
strategy timings, clipping events, warnings, and reproducibility metadata.

Run from the repository root:

    streamlit run dashboard.py

The application executes trusted local strategy Python code.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Pandas 3 defaults text columns and indexes to Arrow-backed string arrays when
# PyArrow is installed.  Some current pandas/PyArrow combinations can crash at
# native level while a long-lived Streamlit process repeatedly reindexes those
# arrays.  Dashboard labels are small, so the stable Python-object representation
# is the safer choice here.
try:
    pd.options.future.infer_string = False
except (AttributeError, KeyError, ValueError):
    pass
try:
    pd.options.mode.string_storage = "python"
except (AttributeError, KeyError, ValueError):
    pass


BACKTESTER_VERSION = "2.1.0"
TRADING_DAYS_PER_YEAR = 250
SCORE_DEFAULT_PARAM = 1.0

DEFAULT_COMMISSION_RATE = 0.0001
INSTRUMENT_0_COMMISSION_RATE = 0.00002
DEFAULT_DOLLAR_POSITION_LIMIT = 10_000.0
INSTRUMENT_0_DOLLAR_POSITION_LIMIT = 100_000.0

POSITION_FUNCTION_CANDIDATES = ("getMyPosition", "getMyPos", "getPosition")
RESET_HOOK_CANDIDATES = ("reset_strategy", "resetStrategy", "reset")
CONFIGURE_HOOK_CANDIDATES = (
    "configure_strategy",
    "configureStrategy",
    "set_parameters",
    "setParameters",
)

PositionFunction = Callable[[np.ndarray], np.ndarray]
InstrumentSelector = str | int


class BacktestError(RuntimeError):
    """Base class for expected, user-facing backtest failures."""


class DataValidationError(BacktestError):
    """Raised when prices or configuration values are invalid."""


class StrategyLoadError(BacktestError):
    """Raised when a strategy module or callback cannot be loaded."""


class StrategyExecutionError(BacktestError):
    """Raised when strategy execution fails or returns invalid positions."""


@dataclass(frozen=True)
class BacktestConfig:
    """Immutable settings for one backtest run.

    Day indices refer to zero-based rows in ``prices.txt``. ``start_day`` and
    ``end_day`` are inclusive PnL days. If ``start_day`` is omitted, the final
    ``num_test_days`` ending at ``end_day`` are scored.

    ``include_instruments`` and ``exclude_instruments`` accept instrument names
    and/or zero-based indices and are mutually exclusive.

    Strategy parameters are passed only through an optional strategy hook such
    as ``configure_strategy(**params)``. The official callback signature remains
    ``getMyPosition(prcSoFar)``.
    """

    prices_path: Path = Path("prices.txt")
    strategy_module: str = "teamName"
    strategy_file: Path | None = None
    position_function: str | None = None

    start_day: int | None = None
    end_day: int | None = None
    num_test_days: int = 250

    include_instruments: tuple[InstrumentSelector, ...] | None = None
    exclude_instruments: tuple[InstrumentSelector, ...] | None = None

    score_param: float = SCORE_DEFAULT_PARAM
    ranking_count: int = 5

    default_commission_rate: float = DEFAULT_COMMISSION_RATE
    instrument_0_commission_rate: float = INSTRUMENT_0_COMMISSION_RATE
    default_dollar_position_limit: float = DEFAULT_DOLLAR_POSITION_LIMIT
    instrument_0_dollar_position_limit: float = INSTRUMENT_0_DOLLAR_POSITION_LIMIT
    commission_overrides: Mapping[str, float] = field(default_factory=dict)
    position_limit_overrides: Mapping[str, float] = field(default_factory=dict)

    strategy_parameters: Mapping[str, Any] = field(default_factory=dict)
    readonly_history: bool = True
    ranking_metric: str = "score_2026"
    quiet: bool = False

    def validate(self) -> None:
        """Validate scalar and mutually exclusive configuration values."""
        if self.num_test_days <= 0:
            raise DataValidationError("num_test_days must be positive.")
        if self.ranking_count <= 0:
            raise DataValidationError("ranking_count must be positive.")
        if self.score_param < 0:
            raise DataValidationError("score_param must be non-negative.")
        if self.include_instruments is not None and self.exclude_instruments is not None:
            raise DataValidationError(
                "Use either include_instruments or exclude_instruments, not both."
            )
        if self.default_commission_rate < 0 or self.instrument_0_commission_rate < 0:
            raise DataValidationError("Commission rates must be non-negative.")
        if self.default_dollar_position_limit <= 0:
            raise DataValidationError("Default dollar position limit must be positive.")
        if self.instrument_0_dollar_position_limit <= 0:
            raise DataValidationError("Instrument 0 dollar position limit must be positive.")
        if self.ranking_metric not in {"score_2025", "score_2026"}:
            raise DataValidationError(
                "ranking_metric must be 'score_2025' or 'score_2026'."
            )
        for name, value in self.commission_overrides.items():
            if not isinstance(name, str) or not name:
                raise DataValidationError("Commission override keys must be symbols or indices.")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"Commission override for {name!r} must be numeric."
                ) from exc
            if not np.isfinite(numeric_value) or numeric_value < 0:
                raise DataValidationError(
                    f"Commission override for {name!r} must be finite and non-negative."
                )
        for name, value in self.position_limit_overrides.items():
            if not isinstance(name, str) or not name:
                raise DataValidationError("Position-limit override keys must be symbols or indices.")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"Position-limit override for {name!r} must be numeric."
                ) from exc
            if not np.isfinite(numeric_value) or numeric_value <= 0:
                raise DataValidationError(
                    f"Position-limit override for {name!r} must be finite and positive."
                )
        try:
            json.dumps(dict(self.strategy_parameters), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                "strategy_parameters must be JSON-serialisable."
            ) from exc


@dataclass(frozen=True)
class StrategyAdapter:
    """Freshly loaded strategy callback plus reproducibility metadata.

    ``owned_module_key`` is set only for strategy files loaded under a temporary
    module name.  Calling :meth:`close` removes that temporary entry from
    ``sys.modules`` so repeated dashboard runs do not leak strategy modules.
    """

    module: ModuleType
    position_function: PositionFunction
    function_name: str
    source_name: str
    source_path: str | None
    source_hash: str | None
    owned_module_key: str | None = None

    def close(self) -> None:
        """Release temporary module registrations created for file strategies."""
        if self.owned_module_key is not None:
            sys.modules.pop(self.owned_module_key, None)

    def prepare(self, parameters: Mapping[str, Any]) -> None:
        """Reset and configure the strategy when optional hooks are provided."""
        reset_hook = _first_callable(self.module, RESET_HOOK_CANDIDATES)
        if reset_hook is not None:
            try:
                reset_hook()
            except Exception as exc:  # strategy code is untrusted local code
                raise StrategyExecutionError(
                    f"Strategy reset hook {reset_hook.__name__} failed: {exc}"
                ) from exc

        if not parameters:
            return

        configure_hook = _first_callable(self.module, CONFIGURE_HOOK_CANDIDATES)
        if configure_hook is None:
            names = ", ".join(CONFIGURE_HOOK_CANDIDATES)
            raise StrategyExecutionError(
                "Strategy parameters were supplied, but the strategy exposes no "
                f"configuration hook. Add one of: {names}."
            )

        try:
            signature = inspect.signature(configure_hook)
            parameters_spec = list(signature.parameters.values())
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters_spec)
            positional = [
                p
                for p in parameters_spec
                if p.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if has_var_kw or len(positional) != 1:
                configure_hook(**dict(parameters))
            else:
                configure_hook(dict(parameters))
        except Exception as exc:  # strategy code is untrusted local code
            raise StrategyExecutionError(
                f"Strategy configuration hook {configure_hook.__name__} failed: {exc}"
            ) from exc


@dataclass(frozen=True)
class BacktestResult:
    """Complete result of one backtest run.

    Wide daily tables use scored day numbers as their index. ``daily_positions``
    contains post-trade target positions. ``daily_held_positions`` contains the
    positions that generated that day's market PnL before the new order.
    """

    symbols: tuple[str, ...]
    active_symbols: tuple[str, ...]
    start_day: int
    end_day: int

    portfolio_summary: pd.DataFrame
    instrument_summary: pd.DataFrame
    best_2025: pd.DataFrame
    worst_2025: pd.DataFrame
    best_2026: pd.DataFrame
    worst_2026: pd.DataFrame

    price_history: pd.DataFrame
    daily_prices: pd.DataFrame
    daily_pnl: pd.DataFrame
    daily_gross_pnl: pd.DataFrame
    daily_equity: pd.DataFrame
    daily_held_positions: pd.DataFrame
    daily_positions: pd.DataFrame
    daily_requested_positions: pd.DataFrame
    daily_orders: pd.DataFrame
    daily_dollar_volume: pd.DataFrame
    daily_commission_charged: pd.DataFrame
    daily_commission_generated: pd.DataFrame
    daily_limit_utilisation: pd.DataFrame
    daily_clipped: pd.DataFrame

    portfolio_daily: pd.DataFrame
    trade_log: pd.DataFrame
    strategy_calls: pd.DataFrame

    warnings: tuple[str, ...]
    audit: Mapping[str, Any]
    timings: Mapping[str, float]

    def summary_value(self, name: str) -> float:
        """Return a scalar metric from ``portfolio_summary``."""
        if name not in self.portfolio_summary.index:
            raise KeyError(f"Unknown portfolio metric: {name}")
        return float(self.portfolio_summary.at[name, "value"])


# ---------------------------------------------------------------------------
# File and strategy loading
# ---------------------------------------------------------------------------


def file_sha256(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Return a streaming SHA-256 digest for a local file."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"File not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prices(path: Path | str) -> tuple[list[str], np.ndarray]:
    """Load and validate a whitespace-delimited price file.

    Returns a symbol list and a C-contiguous ``float64`` array shaped
    ``(n_instruments, n_days)``.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Prices file not found: {resolved}")

    try:
        with resolved.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip().split()
    except OSError as exc:
        raise DataValidationError(f"Could not read prices header: {exc}") from exc

    if not header:
        raise DataValidationError(f"Prices file has no header: {resolved}")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise DataValidationError(
            f"Duplicate instrument names are not allowed: {', '.join(duplicates)}"
        )

    try:
        frame = pd.read_csv(resolved, sep=r"\s+", header=0, index_col=None)
    except Exception as exc:
        raise DataValidationError(f"Could not parse prices file: {exc}") from exc

    if frame.empty or frame.shape[1] == 0:
        raise DataValidationError(f"Prices file contains no usable data: {resolved}")
    if len(frame.columns) != len(header):
        raise DataValidationError(
            "The parsed column count does not match the header column count."
        )

    try:
        numeric = frame.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Every price observation must be numeric.") from exc

    values = numeric.to_numpy(dtype=np.float64, copy=False)
    if values.shape[0] < 2:
        raise DataValidationError("At least two price rows are required.")
    if not np.isfinite(values).all():
        bad_count = int(np.size(values) - np.isfinite(values).sum())
        raise DataValidationError(
            f"Prices contain {bad_count} NaN or infinite observation(s)."
        )
    if (values <= 0).any():
        row, col = np.argwhere(values <= 0)[0]
        raise DataValidationError(
            f"Prices must be strictly positive; found {values[row, col]!r} "
            f"at day {row}, instrument {numeric.columns[col]!r}."
        )

    prices = np.ascontiguousarray(values.T, dtype=np.float64)
    return [str(column) for column in numeric.columns], prices


def _first_callable(module: ModuleType, names: Iterable[str]) -> Callable[..., Any] | None:
    for name in names:
        value = getattr(module, name, None)
        if callable(value):
            return value
    return None


def _load_module_from_file(path: Path) -> tuple[ModuleType, str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Strategy file not found: {resolved}")
    if resolved.suffix.lower() != ".py":
        raise StrategyLoadError("Strategy file must have a .py extension.")

    source_hash = file_sha256(resolved)
    module_name = (
        f"_algothon_strategy_{resolved.stem}_{source_hash[:12]}_{time.time_ns()}"
    )
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise StrategyLoadError(
            f"Could not create an import specification for {resolved}."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    parent = str(resolved.parent)
    inserted = parent not in sys.path
    if inserted:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise StrategyLoadError(f"Could not import strategy file {resolved}: {exc}") from exc
    finally:
        if inserted:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass
    return module, source_hash, module_name


def load_strategy(
    strategy_module: str = "teamName",
    strategy_file: Path | None = None,
    position_function: str | None = None,
) -> StrategyAdapter:
    """Load a fresh strategy module and resolve its position callback."""
    try:
        if strategy_file is not None:
            resolved = Path(strategy_file).expanduser().resolve()
            module, source_hash, owned_module_key = _load_module_from_file(resolved)
            source_name = resolved.name
            source_path: str | None = str(resolved)
        else:
            if not strategy_module.strip():
                raise StrategyLoadError("Strategy module name cannot be empty.")
            importlib.invalidate_caches()
            if strategy_module in sys.modules:
                module = importlib.reload(sys.modules[strategy_module])
            else:
                module = importlib.import_module(strategy_module)
            source_name = strategy_module
            origin = getattr(getattr(module, "__spec__", None), "origin", None)
            source_path = str(Path(origin).resolve()) if origin and origin != "built-in" else None
            source_hash = file_sha256(source_path) if source_path and Path(source_path).is_file() else None
            owned_module_key = None
    except (BacktestError, FileNotFoundError):
        raise
    except Exception as exc:
        source = str(strategy_file) if strategy_file is not None else strategy_module
        raise StrategyLoadError(f"Could not import strategy {source!r}: {exc}") from exc

    candidates: Iterable[str] = (
        (position_function,) if position_function else POSITION_FUNCTION_CANDIDATES
    )
    for name in candidates:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return StrategyAdapter(
                module=module,
                position_function=candidate,
                function_name=name,
                source_name=source_name,
                source_path=source_path,
                source_hash=source_hash,
                owned_module_key=owned_module_key,
            )

    expected = position_function or ", ".join(POSITION_FUNCTION_CANDIDATES)
    if owned_module_key is not None:
        sys.modules.pop(owned_module_key, None)
    raise StrategyLoadError(
        f"No valid position function found in {module.__name__!r}. Expected: {expected}."
    )


def load_position_function(
    strategy_module: str = "teamName",
    strategy_file: Path | None = None,
    position_function: str | None = None,
) -> PositionFunction:
    """Backward-compatible helper returning only the strategy callback."""
    return load_strategy(strategy_module, strategy_file, position_function).position_function


# ---------------------------------------------------------------------------
# Scoring and metrics
# ---------------------------------------------------------------------------


def score_2025(mean_pnl: float, stddev_pnl: float) -> float:
    """Return the 2025 score: mean PnL minus 10% of PnL volatility."""
    return float(mean_pnl - 0.1 * stddev_pnl)


def score_2026(
    mean_pnl: float,
    stddev_pnl: float,
    param: float = SCORE_DEFAULT_PARAM,
) -> float:
    """Return the official 2026 evaluator score."""
    if mean_pnl <= 0 or stddev_pnl < 1e-10:
        return float(mean_pnl)
    sharpe = np.sqrt(TRADING_DAYS_PER_YEAR) * mean_pnl / stddev_pnl
    fraction = sharpe**2 / (sharpe**2 + param**2)
    return float(mean_pnl * fraction)


def annualised_sharpe(mean_pnl: float, stddev_pnl: float) -> float:
    """Calculate annualised Sharpe using 250 trading days."""
    if stddev_pnl <= 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * mean_pnl / stddev_pnl)


def _drawdown_details(equity: np.ndarray, days: np.ndarray) -> dict[str, Any]:
    values = np.asarray(equity, dtype=float)
    day_values = np.asarray(days, dtype=int)
    if values.size == 0:
        return {
            "max_drawdown": 0.0,
            "drawdown_start_day": None,
            "drawdown_trough_day": None,
            "drawdown_recovery_day": None,
            "drawdown_duration": 0,
            "current_drawdown": 0.0,
        }

    augmented_values = np.concatenate(([0.0], values))
    augmented_days = np.concatenate(([day_values[0] - 1], day_values))
    peaks = np.maximum.accumulate(augmented_values)
    drawdowns = peaks - augmented_values
    trough_index = int(np.argmax(drawdowns))
    max_dd = float(drawdowns[trough_index])

    peak_value = peaks[trough_index]
    peak_candidates = np.flatnonzero(augmented_values[: trough_index + 1] == peak_value)
    peak_index = int(peak_candidates[-1])

    recovery_index: int | None = None
    for index in range(trough_index + 1, augmented_values.size):
        if augmented_values[index] >= peak_value:
            recovery_index = index
            break

    current_dd = float(peaks[-1] - augmented_values[-1])
    duration_end = recovery_index if recovery_index is not None else augmented_values.size - 1
    return {
        "max_drawdown": max_dd,
        "drawdown_start_day": int(augmented_days[peak_index]),
        "drawdown_trough_day": int(augmented_days[trough_index]),
        "drawdown_recovery_day": (
            int(augmented_days[recovery_index]) if recovery_index is not None else None
        ),
        "drawdown_duration": int(duration_end - peak_index),
        "current_drawdown": current_dd,
    }


def max_drawdown(equity: np.ndarray) -> float:
    """Return maximum peak-to-trough drawdown as a positive dollar amount."""
    values = np.asarray(equity, dtype=float)
    if values.size == 0:
        return 0.0
    days = np.arange(values.size, dtype=int)
    return float(_drawdown_details(values, days)["max_drawdown"])


def _longest_streak(values: np.ndarray, positive: bool) -> int:
    best = 0
    current = 0
    for value in np.asarray(values, dtype=float):
        matches = value > 0 if positive else value < 0
        current = current + 1 if matches else 0
        best = max(best, current)
    return best


def _loss_risk_metrics(pnl: np.ndarray) -> tuple[float, float]:
    losses = -np.asarray(pnl, dtype=float)
    if losses.size == 0:
        return 0.0, 0.0
    var_95 = max(0.0, float(np.quantile(losses, 0.95)))
    tail = losses[losses >= var_95]
    expected_shortfall = max(0.0, float(np.mean(tail))) if tail.size else 0.0
    return var_95, expected_shortfall


def resolve_day_range(
    n_days: int,
    start_day: int | None,
    end_day: int | None,
    num_test_days: int,
) -> tuple[int, int]:
    """Resolve and validate the inclusive scored day range."""
    if n_days < 2:
        raise DataValidationError("At least two price rows are required.")
    if num_test_days <= 0:
        raise DataValidationError("num_test_days must be positive.")

    end = n_days - 1 if end_day is None else int(end_day)
    start = max(1, end - num_test_days + 1) if start_day is None else int(start_day)

    if start < 1:
        raise DataValidationError(
            "start_day must be at least 1 because day 0 supplies prior history."
        )
    if end >= n_days:
        raise DataValidationError(
            f"end_day={end} exceeds the final valid day {n_days - 1}."
        )
    if start > end:
        raise DataValidationError(
            f"start_day={start} must not exceed end_day={end}."
        )
    return start, end


# ---------------------------------------------------------------------------
# Universe and market configuration
# ---------------------------------------------------------------------------


def _parse_selector(value: str) -> str | int:
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("Instrument selectors cannot be empty.")
    try:
        return int(stripped)
    except ValueError:
        return stripped


def _selector_to_index(selector: InstrumentSelector | str, symbols: Sequence[str]) -> int:
    if isinstance(selector, bool):
        raise DataValidationError(f"Invalid instrument selector: {selector!r}")
    if isinstance(selector, int):
        index = selector
    else:
        text = str(selector).strip()
        if text.lstrip("+-").isdigit():
            index = int(text)
        else:
            try:
                return symbols.index(text)
            except ValueError as exc:
                raise DataValidationError(
                    f"Unknown instrument {text!r}. Valid symbols: {', '.join(symbols)}"
                ) from exc
    if index < 0 or index >= len(symbols):
        raise DataValidationError(
            f"Instrument index {index} is outside 0..{len(symbols) - 1}."
        )
    return index


def _normalise_selectors(
    selectors: Sequence[InstrumentSelector],
    symbols: Sequence[str],
) -> set[int]:
    return {_selector_to_index(selector, symbols) for selector in selectors}


def build_active_mask(
    symbols: Sequence[str],
    include: Sequence[InstrumentSelector] | None,
    exclude: Sequence[InstrumentSelector] | None,
) -> np.ndarray:
    """Resolve include/exclude selectors into a Boolean trading mask."""
    if include is not None and exclude is not None:
        raise DataValidationError(
            "Use either include_instruments or exclude_instruments, not both."
        )
    if include is not None:
        active = _normalise_selectors(include, symbols)
    else:
        active = set(range(len(symbols)))
        if exclude is not None:
            active -= _normalise_selectors(exclude, symbols)
    mask = np.zeros(len(symbols), dtype=bool)
    if active:
        mask[list(active)] = True
    return mask


def build_market_settings(
    symbols: Sequence[str], config: BacktestConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Create commission-rate and dollar-position-limit vectors."""
    n_instruments = len(symbols)
    commission_rates = np.full(
        n_instruments, config.default_commission_rate, dtype=np.float64
    )
    dollar_limits = np.full(
        n_instruments, config.default_dollar_position_limit, dtype=np.float64
    )
    if n_instruments:
        commission_rates[0] = config.instrument_0_commission_rate
        dollar_limits[0] = config.instrument_0_dollar_position_limit

    for selector, value in config.commission_overrides.items():
        commission_rates[_selector_to_index(selector, symbols)] = float(value)
    for selector, value in config.position_limit_overrides.items():
        dollar_limits[_selector_to_index(selector, symbols)] = float(value)
    return commission_rates, dollar_limits


def _validated_target_positions(
    raw_positions: Any,
    n_instruments: int,
    active_mask: np.ndarray,
    current_prices: np.ndarray,
    dollar_limits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate, universe-filter, clip, and integerise strategy positions."""
    positions = np.asarray(raw_positions)
    if positions.shape != (n_instruments,):
        raise StrategyExecutionError(
            f"Strategy returned shape {positions.shape}; expected {(n_instruments,)}."
        )
    try:
        requested = positions.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise StrategyExecutionError("Strategy positions must be numeric.") from exc
    if not np.isfinite(requested).all():
        bad = np.flatnonzero(~np.isfinite(requested))
        raise StrategyExecutionError(
            f"Strategy returned NaN or infinite positions at indices {bad.tolist()}."
        )

    requested[~active_mask] = 0.0
    share_limits = (dollar_limits / current_prices).astype(np.int64)
    clipped_float = np.clip(requested, -share_limits, share_limits)
    clipped_flags = np.abs(requested) > share_limits
    targets = clipped_float.astype(np.int64)
    utilisation = np.divide(
        np.abs(targets) * current_prices,
        dollar_limits,
        out=np.zeros(n_instruments, dtype=np.float64),
        where=dollar_limits > 0,
    )
    return requested, targets, clipped_flags, utilisation


def _summary_frame(metrics: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(dict(metrics), orient="index", columns=["value"])


def _canonical_config(config: BacktestConfig) -> dict[str, Any]:
    data = asdict(config)
    data["prices_path"] = str(Path(config.prices_path).expanduser().resolve())
    data["strategy_file"] = (
        str(Path(config.strategy_file).expanduser().resolve())
        if config.strategy_file is not None
        else None
    )
    data["commission_overrides"] = dict(config.commission_overrides)
    data["position_limit_overrides"] = dict(config.position_limit_overrides)
    data["strategy_parameters"] = dict(config.strategy_parameters)
    return data


# ---------------------------------------------------------------------------
# Backtest execution
# ---------------------------------------------------------------------------


def run_backtest(
    symbols: Sequence[str],
    prices: np.ndarray,
    get_position: PositionFunction | StrategyAdapter,
    config: BacktestConfig,
) -> BacktestResult:
    """Execute an evaluator-compatible backtest with full diagnostics."""
    total_started = time.perf_counter()
    config.validate()

    symbols = [str(symbol) for symbol in symbols]
    prices = np.asarray(prices, dtype=np.float64)
    if prices.ndim != 2:
        raise DataValidationError(
            "prices must be a 2D array shaped (instruments, days)."
        )
    n_instruments, n_days = prices.shape
    if len(symbols) != n_instruments:
        raise DataValidationError("The symbol count does not match the price matrix.")
    if len(set(symbols)) != len(symbols):
        raise DataValidationError("Instrument symbols must be unique.")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise DataValidationError("All prices must be finite and strictly positive.")

    if isinstance(get_position, StrategyAdapter):
        adapter = get_position
        adapter.prepare(config.strategy_parameters)
        position_function = adapter.position_function
    else:
        adapter = None
        position_function = get_position
        if not callable(position_function):
            raise StrategyLoadError("get_position must be callable.")
        if config.strategy_parameters:
            raise StrategyExecutionError(
                "strategy_parameters require a StrategyAdapter loaded with load_strategy()."
            )

    start_day, end_day = resolve_day_range(
        n_days, config.start_day, config.end_day, config.num_test_days
    )
    active_mask = build_active_mask(
        symbols, config.include_instruments, config.exclude_instruments
    )
    commission_rates, dollar_limits = build_market_settings(symbols, config)

    n_scored = end_day - start_day + 1
    shape = (n_scored, n_instruments)
    scored_days = np.arange(start_day, end_day + 1, dtype=np.int64)

    daily_prices_array = np.empty(shape, dtype=np.float64)
    pnl_array = np.empty(shape, dtype=np.float64)
    gross_pnl_array = np.empty(shape, dtype=np.float64)
    equity_array = np.empty(shape, dtype=np.float64)
    held_position_array = np.empty(shape, dtype=np.int64)
    target_position_array = np.empty(shape, dtype=np.int64)
    requested_position_array = np.empty(shape, dtype=np.float64)
    order_array = np.empty(shape, dtype=np.int64)
    volume_array = np.empty(shape, dtype=np.float64)
    commission_charged_array = np.empty(shape, dtype=np.float64)
    commission_generated_array = np.empty(shape, dtype=np.float64)
    utilisation_array = np.empty(shape, dtype=np.float64)
    clipped_array = np.empty(shape, dtype=bool)

    cash_by_instrument = np.zeros(n_instruments, dtype=np.float64)
    current_positions = np.zeros(n_instruments, dtype=np.int64)
    equity_by_instrument = np.zeros(n_instruments, dtype=np.float64)
    previous_commission = np.zeros(n_instruments, dtype=np.float64)
    total_volume_by_instrument = np.zeros(n_instruments, dtype=np.float64)
    total_commission_by_instrument = np.zeros(n_instruments, dtype=np.float64)

    trade_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    strategy_seconds = 0.0
    accounting_seconds = 0.0

    for t in range(start_day, end_day + 2):
        current_prices = prices[:, t - 1]
        held_positions = current_positions.copy()
        commission_charged = previous_commission.copy()

        requested_positions = held_positions.astype(np.float64)
        target_positions = held_positions.copy()
        clipped_flags = np.zeros(n_instruments, dtype=bool)
        utilisation = np.divide(
            np.abs(target_positions) * current_prices,
            dollar_limits,
            out=np.zeros(n_instruments, dtype=np.float64),
            where=dollar_limits > 0,
        )
        call_elapsed = 0.0

        if t <= end_day:
            history = prices[:, :t].view()
            if config.readonly_history:
                history.setflags(write=False)
            call_started = time.perf_counter()
            try:
                raw_positions = position_function(history)
            except Exception as exc:
                decision_day = t - 1
                raise StrategyExecutionError(
                    f"Strategy failed while deciding positions at day {decision_day} "
                    f"with {t} days of history: {exc}"
                ) from exc
            call_elapsed = time.perf_counter() - call_started
            strategy_seconds += call_elapsed

            requested_positions, target_positions, clipped_flags, utilisation = (
                _validated_target_positions(
                    raw_positions,
                    n_instruments,
                    active_mask,
                    current_prices,
                    dollar_limits,
                )
            )
            call_rows.append(
                {
                    "decision_day": t - 1,
                    "history_days": t,
                    "elapsed_seconds": call_elapsed,
                    "requested_nonzero": int(np.count_nonzero(requested_positions)),
                    "target_nonzero": int(np.count_nonzero(target_positions)),
                    "clipped_instruments": int(np.count_nonzero(clipped_flags)),
                }
            )

        accounting_started = time.perf_counter()
        position_change = target_positions - held_positions
        cash_by_instrument -= current_prices * position_change + commission_charged

        traded_volume = current_prices * np.abs(position_change)
        total_volume_by_instrument += traded_volume
        total_commission_by_instrument += commission_charged
        next_commission = traded_volume * commission_rates

        current_positions = target_positions
        new_equity = cash_by_instrument + current_positions * current_prices
        pnl_by_instrument = new_equity - equity_by_instrument
        gross_pnl_by_instrument = pnl_by_instrument + commission_charged
        equity_by_instrument = new_equity
        previous_commission = next_commission

        if t <= end_day:
            nonzero_orders = np.flatnonzero(position_change)
            for instrument_index in nonzero_orders:
                order_size = int(position_change[instrument_index])
                trade_rows.append(
                    {
                        "decision_day": t - 1,
                        "instrument_id": int(instrument_index),
                        "symbol": symbols[instrument_index],
                        "price": float(current_prices[instrument_index]),
                        "previous_position": int(held_positions[instrument_index]),
                        "requested_position": float(requested_positions[instrument_index]),
                        "target_position": int(target_positions[instrument_index]),
                        "order_size": order_size,
                        "side": "BUY" if order_size > 0 else "SELL",
                        "dollar_volume": float(traded_volume[instrument_index]),
                        "commission_generated": float(next_commission[instrument_index]),
                        "position_limit_utilisation": float(utilisation[instrument_index]),
                        "clipped": bool(clipped_flags[instrument_index]),
                    }
                )

        if t > start_day:
            row = t - start_day - 1
            daily_prices_array[row] = current_prices
            pnl_array[row] = pnl_by_instrument
            gross_pnl_array[row] = gross_pnl_by_instrument
            equity_array[row] = equity_by_instrument
            held_position_array[row] = held_positions
            target_position_array[row] = target_positions
            requested_position_array[row] = requested_positions
            order_array[row] = position_change
            volume_array[row] = traded_volume
            commission_charged_array[row] = commission_charged
            commission_generated_array[row] = next_commission
            utilisation_array[row] = utilisation
            clipped_array[row] = clipped_flags

        accounting_seconds += time.perf_counter() - accounting_started

    # The final mark charges the last generated commission, so total charged
    # commission is complete at this point.
    total_commission_by_instrument = np.sum(commission_charged_array, axis=0)

    index = pd.Index(scored_days, name="day")
    price_index = pd.Index(np.arange(start_day - 1, end_day + 1), name="day")
    price_history = pd.DataFrame(
        prices[:, start_day - 1 : end_day + 1].T,
        index=price_index,
        columns=symbols,
    )

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=index, columns=symbols)

    daily_prices = frame(daily_prices_array)
    daily_pnl = frame(pnl_array)
    daily_gross_pnl = frame(gross_pnl_array)
    daily_equity = frame(equity_array)
    daily_held_positions = frame(held_position_array)
    daily_positions = frame(target_position_array)
    daily_requested_positions = frame(requested_position_array)
    daily_orders = frame(order_array)
    daily_dollar_volume = frame(volume_array)
    daily_commission_charged = frame(commission_charged_array)
    daily_commission_generated = frame(commission_generated_array)
    daily_limit_utilisation = frame(utilisation_array)
    daily_clipped = frame(clipped_array)

    pnl_matrix = pnl_array
    equity_matrix = equity_array
    mean_by_instrument = np.mean(pnl_matrix, axis=0)
    std_by_instrument = np.std(pnl_matrix, axis=0, ddof=0)
    cumulative_by_instrument = np.sum(pnl_matrix, axis=0)
    gross_by_instrument = np.sum(gross_pnl_array, axis=0)
    sharpe_by_instrument = np.divide(
        np.sqrt(TRADING_DAYS_PER_YEAR) * mean_by_instrument,
        std_by_instrument,
        out=np.zeros(n_instruments, dtype=np.float64),
        where=std_by_instrument > 0,
    )
    return_by_instrument = np.divide(
        cumulative_by_instrument,
        total_volume_by_instrument,
        out=np.zeros(n_instruments, dtype=np.float64),
        where=total_volume_by_instrument > 0,
    )

    instrument_drawdowns = [
        _drawdown_details(equity_matrix[:, i], scored_days) for i in range(n_instruments)
    ]
    portfolio_total_pnl = np.sum(pnl_matrix, axis=1)
    portfolio_gross_pnl = np.sum(gross_pnl_array, axis=1)
    portfolio_equity = np.sum(equity_matrix, axis=1)
    portfolio_drawdown = _drawdown_details(portfolio_equity, scored_days)

    cumulative_instrument_pnl = np.cumsum(pnl_matrix, axis=0)
    rolling_five = (
        pd.DataFrame(pnl_matrix)
        .rolling(window=min(5, n_scored), min_periods=min(5, n_scored))
        .sum()
        .min(axis=0)
        .to_numpy(dtype=float)
    )

    total_portfolio_pnl = float(np.sum(portfolio_total_pnl))
    abs_contribution = np.abs(cumulative_by_instrument)
    abs_contribution_total = float(np.sum(abs_contribution))
    abs_contribution_share = np.divide(
        abs_contribution,
        abs_contribution_total,
        out=np.zeros(n_instruments, dtype=np.float64),
        where=abs_contribution_total > 0,
    )
    contribution_pct = np.divide(
        cumulative_by_instrument,
        total_portfolio_pnl,
        out=np.zeros(n_instruments, dtype=np.float64),
        where=abs(total_portfolio_pnl) > 1e-12,
    )

    instrument_summary = pd.DataFrame(
        {
            "instrument_id": np.arange(n_instruments, dtype=int),
            "symbol": symbols,
            "active": active_mask,
            "mean_pnl": mean_by_instrument,
            "gross_pnl": gross_by_instrument,
            "commission": total_commission_by_instrument,
            "cumulative_pnl": cumulative_by_instrument,
            "stddev_pnl": std_by_instrument,
            "sharpe": sharpe_by_instrument,
            "return": return_by_instrument,
            "score_2025": mean_by_instrument - 0.1 * std_by_instrument,
            "score_2026": [
                score_2026(mu, sigma, config.score_param)
                for mu, sigma in zip(mean_by_instrument, std_by_instrument)
            ],
            "max_drawdown": [item["max_drawdown"] for item in instrument_drawdowns],
            "current_drawdown": [
                item["current_drawdown"] for item in instrument_drawdowns
            ],
            "dollar_volume": total_volume_by_instrument,
            "profitable_day_pct": np.mean(pnl_matrix > 0, axis=0),
            "worst_one_day": np.min(pnl_matrix, axis=0),
            "worst_five_day": rolling_five,
            "average_abs_position": np.mean(np.abs(target_position_array), axis=0),
            "maximum_abs_position": np.max(np.abs(target_position_array), axis=0),
            "final_position": current_positions,
            "number_of_orders": np.count_nonzero(order_array, axis=0)
            + np.array(
                [
                    sum(
                        1
                        for row in trade_rows
                        if row["instrument_id"] == i
                        and row["decision_day"] == start_day - 1
                    )
                    for i in range(n_instruments)
                ],
                dtype=int,
            ),
            "limit_hit_pct": np.mean(utilisation_array >= 0.999, axis=0),
            "contribution_pct": contribution_pct,
            "absolute_contribution_share": abs_contribution_share,
        }
    )

    daily_total_volume = np.sum(volume_array, axis=1)
    initial_volume = float(
        sum(row["dollar_volume"] for row in trade_rows if row["decision_day"] == start_day - 1)
    )
    daily_commission = np.sum(commission_charged_array, axis=1)
    daily_generated_commission = np.sum(commission_generated_array, axis=1)
    post_trade_notional = target_position_array * daily_prices_array
    gross_exposure = np.sum(np.abs(post_trade_notional), axis=1)
    net_exposure = np.sum(post_trade_notional, axis=1)
    long_exposure = np.sum(np.maximum(post_trade_notional, 0.0), axis=1)
    short_exposure = np.sum(np.abs(np.minimum(post_trade_notional, 0.0)), axis=1)
    portfolio_cumulative = np.cumsum(portfolio_total_pnl)
    running_peaks = np.maximum.accumulate(np.concatenate(([0.0], portfolio_cumulative)))[1:]
    drawdown_series = running_peaks - portfolio_cumulative

    strategy_seconds_by_day = np.zeros(n_scored, dtype=np.float64)
    call_time_by_day = {int(row["decision_day"]): row["elapsed_seconds"] for row in call_rows}
    for row, day in enumerate(scored_days):
        strategy_seconds_by_day[row] = float(call_time_by_day.get(int(day), 0.0))

    portfolio_daily = pd.DataFrame(
        {
            "daily_pnl": portfolio_total_pnl,
            "gross_pnl": portfolio_gross_pnl,
            "cumulative_pnl": portfolio_cumulative,
            "drawdown": drawdown_series,
            "daily_dollar_volume": daily_total_volume,
            "cumulative_dollar_volume": initial_volume + np.cumsum(daily_total_volume),
            "commission_charged": daily_commission,
            "commission_generated": daily_generated_commission,
            "cumulative_commission": np.cumsum(daily_commission),
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "active_positions": np.count_nonzero(target_position_array, axis=1),
            "long_positions": np.sum(target_position_array > 0, axis=1),
            "short_positions": np.sum(target_position_array < 0, axis=1),
            "flat_positions": np.sum(target_position_array == 0, axis=1),
            "number_of_orders": np.count_nonzero(order_array, axis=1),
            "strategy_seconds": strategy_seconds_by_day,
        },
        index=index,
    )

    mean_pnl = float(np.mean(portfolio_total_pnl))
    stddev_pnl = float(np.std(portfolio_total_pnl, ddof=0))
    cumulative_pnl = total_portfolio_pnl
    gross_pnl = float(np.sum(portfolio_gross_pnl))
    commission = float(np.sum(total_commission_by_instrument))
    dollar_volume = float(np.sum(total_volume_by_instrument))
    portfolio_return = cumulative_pnl / dollar_volume if dollar_volume > 0 else 0.0
    positive_days = portfolio_total_pnl[portfolio_total_pnl > 0]
    negative_days = portfolio_total_pnl[portfolio_total_pnl < 0]
    profit_factor = (
        float(np.sum(positive_days) / abs(np.sum(negative_days)))
        if negative_days.size
        else (float("inf") if positive_days.size else 0.0)
    )
    var_95, expected_shortfall_95 = _loss_risk_metrics(portfolio_total_pnl)
    rolling_window = min(5, n_scored)
    worst_five_day = float(
        pd.Series(portfolio_total_pnl)
        .rolling(rolling_window, min_periods=rolling_window)
        .sum()
        .min()
    )

    sorted_abs_shares = np.sort(abs_contribution_share)[::-1]
    top_1_concentration = float(sorted_abs_shares[0]) if sorted_abs_shares.size else 0.0
    top_5_concentration = float(np.sum(sorted_abs_shares[:5]))
    pnl_hhi = float(np.sum(abs_contribution_share**2))
    average_daily_volume = float(np.mean(daily_total_volume))
    average_gross_exposure = float(np.mean(gross_exposure))
    holding_period_approx = (
        average_gross_exposure / average_daily_volume if average_daily_volume > 0 else 0.0
    )

    portfolio_metrics: dict[str, Any] = {
        "mean_pnl": mean_pnl,
        "gross_pnl": gross_pnl,
        "commission": commission,
        "cumulative_pnl": cumulative_pnl,
        "stddev_pnl": stddev_pnl,
        "sharpe": annualised_sharpe(mean_pnl, stddev_pnl),
        "return": portfolio_return,
        "score_2025": score_2025(mean_pnl, stddev_pnl),
        "score_2026": score_2026(mean_pnl, stddev_pnl, config.score_param),
        **portfolio_drawdown,
        "dollar_volume": dollar_volume,
        "n_days": n_scored,
        "start_day": start_day,
        "end_day": end_day,
        "profitable_day_pct": float(np.mean(portfolio_total_pnl > 0)),
        "average_winning_day": float(np.mean(positive_days)) if positive_days.size else 0.0,
        "average_losing_day": float(np.mean(negative_days)) if negative_days.size else 0.0,
        "profit_factor": profit_factor,
        "largest_winning_day": float(np.max(portfolio_total_pnl)),
        "largest_losing_day": float(np.min(portfolio_total_pnl)),
        "worst_five_day": worst_five_day,
        "longest_winning_streak": _longest_streak(portfolio_total_pnl, positive=True),
        "longest_losing_streak": _longest_streak(portfolio_total_pnl, positive=False),
        "var_95": var_95,
        "expected_shortfall_95": expected_shortfall_95,
        "skewness": float(pd.Series(portfolio_total_pnl).skew()),
        "excess_kurtosis": float(pd.Series(portfolio_total_pnl).kurt()),
        "average_daily_turnover": average_daily_volume,
        "average_gross_exposure": average_gross_exposure,
        "holding_period_approx_days": holding_period_approx,
        "top_1_absolute_pnl_concentration": top_1_concentration,
        "top_5_absolute_pnl_concentration": top_5_concentration,
        "pnl_concentration_hhi": pnl_hhi,
        "active_instruments": int(np.count_nonzero(active_mask)),
        "profitable_instruments": int(np.sum(cumulative_by_instrument > 0)),
        "losing_instruments": int(np.sum(cumulative_by_instrument < 0)),
        "number_of_orders": int(len(trade_rows)),
    }
    portfolio_summary = _summary_frame(portfolio_metrics)

    ranking_columns = [
        "instrument_id",
        "symbol",
        "score_2025",
        "score_2026",
        "cumulative_pnl",
        "sharpe",
        "max_drawdown",
        "dollar_volume",
        "contribution_pct",
    ]
    rankable = instrument_summary.loc[instrument_summary["active"]].copy()
    n_ranked = min(config.ranking_count, len(rankable))

    def ranked(score_column: str, ascending: bool) -> pd.DataFrame:
        return (
            rankable.sort_values(
                [score_column, "symbol"],
                ascending=[ascending, True],
                kind="stable",
            )
            .head(n_ranked)[ranking_columns]
            .reset_index(drop=True)
        )

    trade_log = pd.DataFrame(trade_rows)
    if trade_log.empty:
        trade_log = pd.DataFrame(
            columns=[
                "decision_day",
                "instrument_id",
                "symbol",
                "price",
                "previous_position",
                "requested_position",
                "target_position",
                "order_size",
                "side",
                "dollar_volume",
                "commission_generated",
                "position_limit_utilisation",
                "clipped",
                "next_day_pnl",
                "next_day_cumulative_pnl",
            ]
        )
    else:
        next_day_pnl: list[float] = []
        next_day_cumulative: list[float] = []
        symbol_to_index = {symbol: i for i, symbol in enumerate(symbols)}
        for row in trade_rows:
            next_day = int(row["decision_day"]) + 1
            instrument_index = symbol_to_index[str(row["symbol"])]
            if start_day <= next_day <= end_day:
                pnl_row = next_day - start_day
                next_day_pnl.append(float(pnl_matrix[pnl_row, instrument_index]))
                next_day_cumulative.append(
                    float(cumulative_instrument_pnl[pnl_row, instrument_index])
                )
            else:
                next_day_pnl.append(float("nan"))
                next_day_cumulative.append(float("nan"))
        trade_log["next_day_pnl"] = next_day_pnl
        trade_log["next_day_cumulative_pnl"] = next_day_cumulative
        trade_log = trade_log.sort_values(
            ["decision_day", "instrument_id"], kind="stable"
        ).reset_index(drop=True)

    strategy_calls = pd.DataFrame(call_rows)

    warnings: list[str] = []
    portfolio_sharpe = float(portfolio_metrics["sharpe"])
    if dollar_volume == 0:
        warnings.append("The strategy generated no trades.")
    if stddev_pnl <= 1e-12:
        warnings.append("Portfolio daily PnL volatility is zero or numerically negligible.")
    if abs(portfolio_sharpe) >= 10:
        warnings.append(
            f"Annualised Sharpe is unusually large ({portfolio_sharpe:.2f}); "
            "check for leakage, stale prices, or a very small evaluation sample."
        )
    if negative_days.size == 0 and n_scored > 1:
        warnings.append("There are no losing portfolio days; verify the strategy for leakage.")
    if top_1_concentration >= 0.80:
        warnings.append(
            f"One instrument contributes {top_1_concentration:.1%} of absolute PnL; "
            "portfolio performance is highly concentrated."
        )
    limit_hit_pct = float(np.mean(utilisation_array >= 0.999))
    if limit_hit_pct >= 0.50:
        warnings.append(
            f"Positions are at or near their dollar limits on {limit_hit_pct:.1%} "
            "of instrument-days."
        )
    avg_strategy_seconds = strategy_seconds / len(call_rows) if call_rows else 0.0
    if avg_strategy_seconds >= 0.1:
        warnings.append(
            f"Average strategy call time is {avg_strategy_seconds:.3f}s; this may be "
            "slow for repeated research runs."
        )

    total_seconds = time.perf_counter() - total_started
    timings = {
        "total_seconds": total_seconds,
        "strategy_seconds": strategy_seconds,
        "accounting_seconds": accounting_seconds,
        "average_strategy_call_seconds": avg_strategy_seconds,
        "strategy_call_count": float(len(call_rows)),
    }

    prices_path = Path(config.prices_path).expanduser().resolve()
    prices_hash = file_sha256(prices_path) if prices_path.is_file() else None
    config_payload = _canonical_config(config)
    identity_payload = {
        "backtester_version": BACKTESTER_VERSION,
        "prices_hash": prices_hash,
        "strategy_hash": adapter.source_hash if adapter is not None else None,
        "config": config_payload,
    }
    run_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    audit: dict[str, Any] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backtester_version": BACKTESTER_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "prices_path": str(prices_path),
        "prices_sha256": prices_hash,
        "strategy_source": adapter.source_name if adapter is not None else "callable",
        "strategy_path": adapter.source_path if adapter is not None else None,
        "strategy_sha256": adapter.source_hash if adapter is not None else None,
        "position_function": adapter.function_name if adapter is not None else getattr(position_function, "__name__", "callable"),
        "config": config_payload,
        "timings": timings,
    }

    return BacktestResult(
        symbols=tuple(symbols),
        active_symbols=tuple(np.asarray(symbols)[active_mask].tolist()),
        start_day=start_day,
        end_day=end_day,
        portfolio_summary=portfolio_summary,
        instrument_summary=instrument_summary,
        best_2025=ranked("score_2025", ascending=False),
        worst_2025=ranked("score_2025", ascending=True),
        best_2026=ranked("score_2026", ascending=False),
        worst_2026=ranked("score_2026", ascending=True),
        price_history=price_history,
        daily_prices=daily_prices,
        daily_pnl=daily_pnl,
        daily_gross_pnl=daily_gross_pnl,
        daily_equity=daily_equity,
        daily_held_positions=daily_held_positions,
        daily_positions=daily_positions,
        daily_requested_positions=daily_requested_positions,
        daily_orders=daily_orders,
        daily_dollar_volume=daily_dollar_volume,
        daily_commission_charged=daily_commission_charged,
        daily_commission_generated=daily_commission_generated,
        daily_limit_utilisation=daily_limit_utilisation,
        daily_clipped=daily_clipped,
        portfolio_daily=portfolio_daily,
        trade_log=trade_log,
        strategy_calls=strategy_calls,
        warnings=tuple(warnings),
        audit=audit,
        timings=timings,
    )


def run_backtest_from_config(config: BacktestConfig) -> BacktestResult:
    """Load prices and a fresh strategy, then run one backtest.

    Temporary strategy modules are always unregistered, including when strategy
    execution fails.  This matters for long-lived Streamlit processes where
    every button press reruns the application in the same Python interpreter.
    """
    symbols, prices = load_prices(config.prices_path)
    adapter = load_strategy(
        strategy_module=config.strategy_module,
        strategy_file=config.strategy_file,
        position_function=config.position_function,
    )
    try:
        return run_backtest(symbols, prices, adapter, config)
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# Terminal reporting and CLI
# ---------------------------------------------------------------------------


def _format_number(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if np.isposinf(value):
            return "∞"
        if np.isneginf(value):
            return "-∞"
        if np.isnan(value):
            return "NaN"
        return f"{float(value):,.4f}"
    return str(value)


def print_report(result: BacktestResult) -> None:
    """Print a compact terminal report."""
    print("\n" + "=" * 96)
    print(
        f"BACKTEST {result.audit['run_id']} | days {result.start_day}..{result.end_day} "
        f"| active {len(result.active_symbols)}/{len(result.symbols)}"
    )
    print("=" * 96)

    headline = [
        "score_2026",
        "score_2025",
        "cumulative_pnl",
        "sharpe",
        "max_drawdown",
        "dollar_volume",
        "commission",
        "number_of_orders",
    ]
    print("\nPORTFOLIO SUMMARY")
    print(
        result.portfolio_summary.loc[headline].to_string(
            formatters={"value": _format_number}
        )
    )

    ranking_formatters = {
        column: _format_number
        for column in [
            "score_2025",
            "score_2026",
            "cumulative_pnl",
            "sharpe",
            "max_drawdown",
            "dollar_volume",
            "contribution_pct",
        ]
    }
    for title, table in (
        ("BEST — SCORE 2026", result.best_2026),
        ("WORST — SCORE 2026", result.worst_2026),
        ("BEST — SCORE 2025", result.best_2025),
        ("WORST — SCORE 2025", result.worst_2025),
    ):
        print(f"\n{title}")
        print(
            "No active instruments."
            if table.empty
            else table.to_string(index=False, formatters=ranking_formatters)
        )

    if result.warnings:
        print("\nWARNINGS")
        for warning in result.warnings:
            print(f"* {warning}")
    print(
        f"\nRuntime: {result.timings['total_seconds']:.3f}s total; "
        f"{result.timings['strategy_seconds']:.3f}s in strategy."
    )
    print("=" * 96 + "\n")


def _json_object(value: str, option_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} must be valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{option_name} must decode to a JSON object.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run an evaluator-compatible Algothon backtest with diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prices", type=Path, default=Path("prices.txt"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--strategy-module", default="teamName")
    source.add_argument("--strategy-file", type=Path)
    parser.add_argument(
        "--position-function",
        help="Exact callback name; otherwise common names are tried.",
    )
    parser.add_argument("--start-day", type=int)
    parser.add_argument("--end-day", type=int)
    parser.add_argument("--num-test-days", type=int, default=250)

    universe = parser.add_mutually_exclusive_group()
    universe.add_argument("--include", nargs="+", type=_parse_selector)
    universe.add_argument("--exclude", nargs="+", type=_parse_selector)

    parser.add_argument("--score-param", type=float, default=SCORE_DEFAULT_PARAM)
    parser.add_argument("--ranking-count", type=int, default=5)
    parser.add_argument("--default-commission-rate", type=float, default=DEFAULT_COMMISSION_RATE)
    parser.add_argument(
        "--instrument-0-commission-rate",
        type=float,
        default=INSTRUMENT_0_COMMISSION_RATE,
    )
    parser.add_argument(
        "--default-position-limit",
        type=float,
        default=DEFAULT_DOLLAR_POSITION_LIMIT,
    )
    parser.add_argument(
        "--instrument-0-position-limit",
        type=float,
        default=INSTRUMENT_0_DOLLAR_POSITION_LIMIT,
    )
    parser.add_argument("--strategy-parameters", default="{}")
    parser.add_argument("--commission-overrides", default="{}")
    parser.add_argument("--position-limit-overrides", default="{}")
    parser.add_argument("--writable-history", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = BacktestConfig(
            prices_path=args.prices,
            strategy_module=args.strategy_module or "teamName",
            strategy_file=args.strategy_file,
            position_function=args.position_function,
            start_day=args.start_day,
            end_day=args.end_day,
            num_test_days=args.num_test_days,
            include_instruments=tuple(args.include) if args.include is not None else None,
            exclude_instruments=tuple(args.exclude) if args.exclude is not None else None,
            score_param=args.score_param,
            ranking_count=args.ranking_count,
            default_commission_rate=args.default_commission_rate,
            instrument_0_commission_rate=args.instrument_0_commission_rate,
            default_dollar_position_limit=args.default_position_limit,
            instrument_0_dollar_position_limit=args.instrument_0_position_limit,
            strategy_parameters=_json_object(args.strategy_parameters, "--strategy-parameters"),
            commission_overrides=_json_object(args.commission_overrides, "--commission-overrides"),
            position_limit_overrides=_json_object(
                args.position_limit_overrides, "--position-limit-overrides"
            ),
            readonly_history=not args.writable_history,
            quiet=args.quiet,
        )
        result = run_backtest_from_config(config)
    except (BacktestError, FileNotFoundError) as exc:
        parser.exit(2, f"backtester: error: {exc}\n")

    if not config.quiet:
        print_report(result)
    return 0

@dataclass(frozen=True)
class ComparisonResult:
    """Aligned tables used by the dashboard's run-comparison page."""

    summary: pd.DataFrame
    cumulative_pnl: pd.DataFrame
    drawdown: pd.DataFrame
    instrument_differences: pd.DataFrame


def rolling_sharpe(values: pd.Series, window: int) -> pd.Series:
    """Annualised rolling Sharpe with population standard deviation."""
    if window <= 1:
        raise ValueError("window must be greater than 1")
    series = values.astype(float)
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return np.sqrt(TRADING_DAYS_PER_YEAR) * mean.div(std.where(std > 0))


def rolling_volatility(values: pd.Series, window: int) -> pd.Series:
    """Rolling population standard deviation."""
    if window <= 1:
        raise ValueError("window must be greater than 1")
    return values.astype(float).rolling(window, min_periods=window).std(ddof=0)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Positive dollar drawdown from the running peak, including initial zero."""
    values = equity.astype(float)
    if values.empty:
        return pd.Series(dtype=float, name="drawdown", index=values.index)
    augmented = pd.concat(
        [pd.Series([0.0], index=[int(values.index.min()) - 1]), values]
    )
    result = augmented.cummax() - augmented
    return result.iloc[1:].rename("drawdown")


def correlation_matrix(result: BacktestResult, kind: str) -> pd.DataFrame:
    """Return one of the dashboard's instrument correlation matrices."""
    normalized = kind.strip().lower()
    if normalized == "returns":
        data = result.daily_prices.pct_change(fill_method=None)
    elif normalized == "pnl":
        data = result.daily_pnl
    elif normalized == "positions":
        data = result.daily_positions
    elif normalized in {"trade direction", "trade_direction", "orders"}:
        data = np.sign(result.daily_orders)
    else:
        raise ValueError(f"Unknown correlation kind: {kind}")

    frame = pd.DataFrame(data, index=result.daily_pnl.index, columns=result.symbols)
    nonconstant = frame.columns[frame.nunique(dropna=True) > 1]
    if len(nonconstant) == 0:
        return pd.DataFrame()
    return frame[nonconstant].corr().fillna(0.0)


def instrument_cumulative_pnl(result: BacktestResult) -> pd.DataFrame:
    """Return cumulative PnL by instrument."""
    return result.daily_pnl.cumsum()


def instrument_drawdowns(result: BacktestResult) -> pd.DataFrame:
    """Return positive drawdown by instrument."""
    equity = result.daily_equity
    augmented = pd.concat(
        [
            pd.DataFrame(
                [np.zeros(len(result.symbols))],
                index=[result.start_day - 1],
                columns=result.symbols,
            ),
            equity,
        ]
    )
    return (augmented.cummax() - augmented).iloc[1:]


def normalized_prices(result: BacktestResult, symbols: Iterable[str]) -> pd.DataFrame:
    """Rebase selected prices to 100 at the beginning of displayed history."""
    selected = [str(symbol) for symbol in symbols]
    if not selected:
        return pd.DataFrame(index=result.price_history.index)
    missing = [symbol for symbol in selected if symbol not in result.price_history.columns]
    if missing:
        raise KeyError(f"Unknown instrument(s): {', '.join(missing)}")
    frame = result.price_history.loc[:, selected].astype(float)
    if frame.empty:
        return frame
    first = frame.iloc[0]
    if (first == 0).any():
        zero_symbols = first.index[first == 0].tolist()
        raise ValueError(
            "Cannot normalise instruments with a zero initial price: "
            + ", ".join(map(str, zero_symbols))
        )
    return frame.div(first).mul(100.0)


def selected_day_table(
    result: BacktestResult,
    day: int,
    *,
    orders_only: bool = True,
) -> pd.DataFrame:
    """Build the complete instrument snapshot for one scored day."""
    if day not in result.daily_pnl.index:
        raise KeyError(
            f"Day {day} is outside the scored range {result.start_day}..{result.end_day}."
        )

    symbols = list(result.symbols)
    cumulative = result.daily_pnl.loc[:day].sum(axis=0)
    table = pd.DataFrame(
        {
            "instrument_id": np.arange(len(symbols), dtype=int),
            "symbol": symbols,
            "price": result.daily_prices.loc[day].to_numpy(dtype=float),
            "previous_position": result.daily_held_positions.loc[day].to_numpy(dtype=int),
            "target_position": result.daily_positions.loc[day].to_numpy(dtype=int),
            "requested_position": result.daily_requested_positions.loc[day].to_numpy(dtype=float),
            "order_size": result.daily_orders.loc[day].to_numpy(dtype=int),
            "dollar_volume": result.daily_dollar_volume.loc[day].to_numpy(dtype=float),
            "commission_generated": result.daily_commission_generated.loc[day].to_numpy(dtype=float),
            "daily_pnl": result.daily_pnl.loc[day].to_numpy(dtype=float),
            "cumulative_pnl": cumulative.to_numpy(dtype=float),
            "position_limit_utilisation": result.daily_limit_utilisation.loc[day].to_numpy(dtype=float),
            "clipped": result.daily_clipped.loc[day].to_numpy(dtype=bool),
        }
    )
    table["side"] = np.select(
        [table["order_size"] > 0, table["order_size"] < 0],
        ["BUY", "SELL"],
        default="—",
    )
    if orders_only:
        table = table.loc[table["order_size"] != 0]
    return table.reset_index(drop=True)


def selected_day_snapshot(result: BacktestResult, day: int) -> dict[str, Any]:
    """Return portfolio-level metrics for a selected scored day."""
    if day not in result.portfolio_daily.index:
        raise KeyError(f"Unknown scored day: {day}")
    row = result.portfolio_daily.loc[day]
    pnl_row = result.daily_pnl.loc[day]
    position_row = result.daily_positions.loc[day]
    abs_notional = np.abs(
        position_row.to_numpy(dtype=float)
        * result.daily_prices.loc[day].to_numpy(dtype=float)
    )
    largest_position_index = int(np.argmax(abs_notional)) if abs_notional.size else 0
    return {
        "daily_pnl": float(row["daily_pnl"]),
        "number_of_orders": int(row["number_of_orders"]),
        "dollar_volume": float(row["daily_dollar_volume"]),
        "gross_exposure": float(row["gross_exposure"]),
        "net_exposure": float(row["net_exposure"]),
        "long_positions": int(row["long_positions"]),
        "short_positions": int(row["short_positions"]),
        "largest_position": result.symbols[largest_position_index],
        "largest_contributor": str(pnl_row.idxmax()),
        "largest_detractor": str(pnl_row.idxmin()),
    }


def compare_results(
    current: BacktestResult,
    baseline: BacktestResult,
    *,
    current_name: str = "Current",
    baseline_name: str = "Baseline",
) -> ComparisonResult:
    """Align and compare two completed backtests."""
    metrics = [
        "score_2026",
        "score_2025",
        "cumulative_pnl",
        "sharpe",
        "max_drawdown",
        "dollar_volume",
        "commission",
        "profitable_day_pct",
        "active_instruments",
        "number_of_orders",
    ]
    def metric_values(result: BacktestResult) -> pd.Series:
        values = {
            metric: (
                result.portfolio_summary.at[metric, "value"]
                if metric in result.portfolio_summary.index
                else float("nan")
            )
            for metric in metrics
        }
        return pd.Series(values, dtype=float)

    current_values = metric_values(current)
    baseline_values = metric_values(baseline)
    summary = pd.DataFrame(
        {
            baseline_name: baseline_values,
            current_name: current_values,
            "difference": current_values - baseline_values,
        },
        index=pd.Index(metrics, dtype=object, name="metric"),
    )

    cumulative = pd.concat(
        [
            baseline.portfolio_daily["cumulative_pnl"].rename(baseline_name),
            current.portfolio_daily["cumulative_pnl"].rename(current_name),
        ],
        axis=1,
    ).sort_index()
    drawdown = pd.concat(
        [
            baseline.portfolio_daily["drawdown"].rename(baseline_name),
            current.portfolio_daily["drawdown"].rename(current_name),
        ],
        axis=1,
    ).sort_index()

    columns = [
        "cumulative_pnl",
        "score_2026",
        "score_2025",
        "sharpe",
        "max_drawdown",
        "dollar_volume",
        "commission",
    ]
    baseline_instruments = baseline.instrument_summary.set_index("symbol")[columns]
    current_instruments = current.instrument_summary.set_index("symbol")[columns]
    common = [
        symbol
        for symbol in baseline_instruments.index.tolist()
        if symbol in current_instruments.index
    ]
    difference = current_instruments.loc[common] - baseline_instruments.loc[common]
    difference.columns = [f"delta_{name}" for name in difference.columns]
    difference = difference.sort_values("delta_cumulative_pnl", ascending=False)

    return ComparisonResult(
        summary=summary,
        cumulative_pnl=cumulative,
        drawdown=drawdown,
        instrument_differences=difference,
    )


APP_TITLE = "Algothon Backtester"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRICES_PATH = PROJECT_ROOT / "01-data" / "prices.txt"
DEFAULT_STRATEGY_PATH = PROJECT_ROOT / "03-strategy" / "strategy.py"
DEFAULT_PAGE_SIZE = 75
LOW_MEMORY_PAGE_SIZE = 40
MAX_CORRELATION_INSTRUMENTS = 51
LOW_MEMORY_CORRELATION_INSTRUMENTS = 20
PLOT_CONFIG = {
    "displaylogo": False,
    "scrollZoom": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 2rem;}
      [data-testid="stMetricValue"] {font-size: 1.35rem;}
      [data-testid="stSidebar"] {min-width: 320px; max-width: 410px;}
      div[role="radiogroup"] {gap: 0.35rem;}
      div[role="radiogroup"] label {padding: 0.25rem 0.55rem; border-radius: 0.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Compatibility, state, and defensive display helpers
# ---------------------------------------------------------------------------


def _supported_kwargs(function: Callable[..., Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Return only keyword arguments accepted by the installed Streamlit API.

    Streamlit has renamed a few sizing arguments across releases.  Filtering at
    runtime keeps this dashboard usable across a wider set of supported
    versions without version-string branching.
    """
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return dict(values)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        filtered = dict(values)
    else:
        filtered = {
            key: value for key, value in values.items() if key in signature.parameters
        }
    # Current Streamlit uses ``width="stretch"``.  Older supported releases
    # use ``use_container_width=True``.  Never pass both because newer releases
    # warn and future releases may reject the deprecated argument.
    if "width" in signature.parameters and "width" in filtered:
        filtered.pop("use_container_width", None)
    elif "use_container_width" in signature.parameters:
        filtered.pop("width", None)
    return filtered


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "current_result": None,
        "current_name": "Current run",
        "baseline_result": None,
        "baseline_name": "Baseline",
        "last_error": None,
        "nav_page": "Overview",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_large_session_objects(*, clear_baseline: bool = False) -> None:
    """Drop large session-owned objects and request prompt garbage collection."""
    st.session_state["current_result"] = None
    st.session_state["current_name"] = "Current run"
    if clear_baseline:
        st.session_state["baseline_result"] = None
        st.session_state["baseline_name"] = "Baseline"
    gc.collect()


def _safe_scalar(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def _summary_value(result: BacktestResult, name: str, default: float = float("nan")) -> float:
    """Read a summary metric without allowing a missing row to crash a page."""
    try:
        return _safe_scalar(result.portfolio_summary.at[name, "value"], default)
    except (KeyError, TypeError, ValueError):
        return default


def _money(value: Any) -> str:
    number = _safe_scalar(value)
    if math.isnan(number):
        return "—"
    if math.isinf(number):
        return "∞" if number > 0 else "-∞"
    sign = "-" if number < 0 else ""
    return f"{sign}${abs(number):,.2f}"


def _number(value: Any, decimals: int = 2) -> str:
    number = _safe_scalar(value)
    if math.isnan(number):
        return "—"
    if math.isinf(number):
        return "∞" if number > 0 else "-∞"
    return f"{number:,.{decimals}f}"


def _percent(value: Any, decimals: int = 1) -> str:
    number = _safe_scalar(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{decimals}%}"


def _metric_row(items: Sequence[tuple[str, str, str | None]]) -> None:
    """Render metrics in small groups to avoid extremely wide frontend nodes."""
    if not items:
        return
    for offset in range(0, len(items), 4):
        group = items[offset : offset + 4]
        columns = st.columns(len(group))
        for column, (label, value, help_text) in zip(columns, group):
            column.metric(label, value, help=help_text)


def _standard_layout(
    figure: go.Figure,
    *,
    height: int = 430,
    hovermode: str = "x unified",
) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=50, b=25),
        hovermode=hovermode,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    return figure


def _show_plot(figure: go.Figure, key: str) -> None:
    """Render one Plotly figure with cross-version sizing arguments."""
    kwargs = _supported_kwargs(
        st.plotly_chart,
        {
            "use_container_width": True,
            "width": "stretch",
            "config": PLOT_CONFIG,
            "key": key,
        },
    )
    st.plotly_chart(figure, **kwargs)


def _jsonish(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _display_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create an Arrow-friendly presentation copy.

    Mixed Python objects in a pandas ``object`` column can cause conversion
    failures in ``st.dataframe``.  Only problematic values are stringified;
    normal strings and missing values remain unchanged.
    """
    display = frame.copy(deep=False)
    for column in display.columns:
        if display[column].dtype != object:
            continue
        display[column] = display[column].map(
            lambda value: (
                _jsonish(value)
                if isinstance(value, (dict, list, tuple, set, Path))
                else value
            )
        )
    return display


def _dataframe(
    frame: pd.DataFrame,
    *,
    height: int = 420,
    hide_index: bool = True,
    column_config: Mapping[str, Any] | None = None,
    key: str | None = None,
) -> None:
    """Render a bounded HTML table without Streamlit's PyArrow conversion.

    ``st.dataframe`` serialises pandas objects through PyArrow. That path is
    implicated in the observed process-level crash. A plain HTML table is less
    feature-rich but stays in Python and the browser, which is deliberately
    safer here.

    ``column_config`` and ``key`` are accepted for call-site compatibility but
    are not used by the HTML renderer.
    """
    del column_config, key
    display = _display_safe_frame(frame)

    def format_float(value: float) -> str:
        if pd.isna(value):
            return ""
        if np.isposinf(value):
            return "∞"
        if np.isneginf(value):
            return "-∞"
        magnitude = abs(float(value))
        if magnitude >= 100_000:
            return f"{value:,.0f}"
        if magnitude >= 100:
            return f"{value:,.2f}"
        return f"{value:,.4f}"

    table_html = display.to_html(
        index=not hide_index,
        border=0,
        classes=["algothon-table"],
        escape=True,
        na_rep="",
        float_format=format_float,
    )
    st.markdown(
        f"""
        <div class="algothon-table-wrap" style="max-height:{int(height)}px; overflow:auto;">
          {table_html}
        </div>
        <style>
          .algothon-table-wrap {{
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 0.35rem;
          }}
          .algothon-table {{
            border-collapse: collapse;
            width: max-content;
            min-width: 100%;
            font-size: 0.84rem;
          }}
          .algothon-table th, .algothon-table td {{
            border-bottom: 1px solid rgba(128,128,128,0.18);
            padding: 0.32rem 0.5rem;
            text-align: right;
            white-space: nowrap;
          }}
          .algothon-table th {{
            position: sticky;
            top: 0;
            z-index: 1;
            background: var(--background-color, white);
            font-weight: 600;
          }}
          .algothon-table th:first-child, .algothon-table td:first-child {{
            text-align: left;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _paginated_frame(
    frame: pd.DataFrame,
    *,
    key: str,
    page_size: int,
    height: int = 420,
    hide_index: bool = True,
    column_config: Mapping[str, Any] | None = None,
) -> None:
    """Send only one bounded slice of a potentially large table to the browser."""
    total_rows = int(len(frame))
    if total_rows == 0:
        st.info("No rows to display.")
        return

    page_size = max(1, int(page_size))
    page_count = max(1, math.ceil(total_rows / page_size))
    if page_count == 1:
        _dataframe(
            frame,
            height=height,
            hide_index=hide_index,
            column_config=column_config,
            key=f"{key}-table",
        )
        return

    page = st.number_input(
        "Page",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
        key=f"{key}-page",
    )
    start = (int(page) - 1) * page_size
    stop = min(start + page_size, total_rows)
    st.caption(f"Rows {start + 1:,}–{stop:,} of {total_rows:,}")
    _dataframe(
        frame.iloc[start:stop],
        height=height,
        hide_index=hide_index,
        column_config=column_config,
        key=f"{key}-table",
    )


# ---------------------------------------------------------------------------
# Small cached metadata only — never cache a full BacktestResult here
# ---------------------------------------------------------------------------


def _path_signature(path_text: str) -> tuple[str, int, int]:
    path = Path(path_text).expanduser().resolve()
    stat = path.stat()
    return str(path), int(stat.st_size), int(stat.st_mtime_ns)


@st.cache_data(show_spinner=False, max_entries=16)
def _preview_prices(
    path_text: str,
    size: int,
    mtime_ns: int,
) -> tuple[tuple[str, ...], int, int]:
    del size, mtime_ns
    symbols, prices = load_prices(Path(path_text))
    return tuple(symbols), int(prices.shape[1]), int(prices.shape[0])


def _parse_json_object(text: str, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text.strip() or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{label} is invalid JSON: {exc.msg} at character {exc.pos}."
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def _strategy_source_hash(strategy_file: str | None, strategy_module: str) -> str:
    if strategy_file:
        return file_sha256(Path(strategy_file))
    specification = importlib.util.find_spec(strategy_module)
    origin = getattr(specification, "origin", None) if specification is not None else None
    if origin and origin != "built-in" and Path(origin).is_file():
        return file_sha256(Path(origin))
    return f"module:{strategy_module}"


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _portfolio_pnl_figure(result: BacktestResult, mode: str) -> go.Figure:
    daily = result.portfolio_daily
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    if mode in {"Daily and cumulative PnL", "Daily PnL"}:
        figure.add_trace(
            go.Bar(x=daily.index, y=daily["daily_pnl"], name="Daily PnL"),
            secondary_y=False,
        )
    if mode in {"Daily and cumulative PnL", "Cumulative PnL"}:
        figure.add_trace(
            go.Scatter(
                x=daily.index,
                y=daily["cumulative_pnl"],
                name="Cumulative PnL",
                mode="lines",
            ),
            secondary_y=(mode == "Daily and cumulative PnL"),
        )
    figure.update_xaxes(title_text="Day")
    figure.update_yaxes(title_text="PnL ($)", secondary_y=False)
    if mode == "Daily and cumulative PnL":
        figure.update_yaxes(title_text="Cumulative PnL ($)", secondary_y=True)
    figure.update_layout(title=mode)
    return _standard_layout(figure)


def _pnl_drawdown_figure(result: BacktestResult) -> go.Figure:
    daily = result.portfolio_daily
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
    )
    figure.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["cumulative_pnl"],
            name="Cumulative PnL",
            mode="lines",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["drawdown"],
            name="Drawdown",
            mode="lines",
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="PnL ($)", row=1, col=1)
    figure.update_yaxes(title_text="Drawdown ($)", row=2, col=1)
    figure.update_xaxes(title_text="Day", row=2, col=1)
    figure.update_layout(title="Portfolio PnL and drawdown")
    return _standard_layout(figure, height=580)


def _exposure_figure(result: BacktestResult) -> go.Figure:
    daily = result.portfolio_daily
    figure = go.Figure()
    for column, label in (
        ("gross_exposure", "Gross exposure"),
        ("net_exposure", "Net exposure"),
        ("long_exposure", "Long exposure"),
        ("short_exposure", "Short exposure"),
    ):
        if column in daily.columns:
            figure.add_trace(
                go.Scatter(x=daily.index, y=daily[column], mode="lines", name=label)
            )
    figure.update_layout(title="Portfolio exposure")
    figure.update_xaxes(title="Day")
    figure.update_yaxes(title="Dollar exposure")
    return _standard_layout(figure)


def _turnover_figure(result: BacktestResult) -> go.Figure:
    daily = result.portfolio_daily
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Bar(
            x=daily.index,
            y=daily["daily_dollar_volume"],
            name="Daily dollar volume",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=daily.index,
            y=daily["cumulative_commission"],
            name="Cumulative commission",
            mode="lines",
        ),
        secondary_y=True,
    )
    figure.update_xaxes(title_text="Day")
    figure.update_yaxes(title_text="Daily volume ($)", secondary_y=False)
    figure.update_yaxes(title_text="Commission ($)", secondary_y=True)
    figure.update_layout(title="Turnover and costs")
    return _standard_layout(figure)


def _rolling_risk_figure(result: BacktestResult, window: int) -> go.Figure:
    pnl = result.portfolio_daily["daily_pnl"]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=pnl.index,
            y=rolling_sharpe(pnl, window),
            name=f"{window}-day rolling Sharpe",
            mode="lines",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=pnl.index,
            y=rolling_volatility(pnl, window),
            name=f"{window}-day PnL volatility",
            mode="lines",
        ),
        secondary_y=True,
    )
    figure.update_xaxes(title_text="Day")
    figure.update_yaxes(title_text="Annualised Sharpe", secondary_y=False)
    figure.update_yaxes(title_text="PnL volatility ($)", secondary_y=True)
    figure.update_layout(title="Rolling risk")
    return _standard_layout(figure)



def _instrument_price_position_figure(result: BacktestResult, symbol: str) -> go.Figure:
    """Price chart with scaled buy/sell markers and target position."""
    prices = result.price_history[symbol]
    positions = result.daily_positions[symbol]
    trades = result.trade_log.loc[result.trade_log["symbol"] == symbol].copy()

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices,
            name="Price",
            mode="lines",
            hovertemplate="Day %{x}<br>Price: %{y:,.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    if not trades.empty:
        maximum_volume = max(float(trades["dollar_volume"].max()), 1.0)
        trades["marker_size"] = 8.0 + 18.0 * np.sqrt(
            trades["dollar_volume"].astype(float) / maximum_volume
        )
        for side, marker_symbol in (("BUY", "triangle-up"), ("SELL", "triangle-down")):
            subset = trades.loc[trades["side"] == side]
            if subset.empty:
                continue
            custom = np.column_stack(
                [
                    subset["order_size"],
                    subset["target_position"],
                    subset["dollar_volume"],
                    subset["commission_generated"],
                    subset["next_day_pnl"],
                    subset["next_day_cumulative_pnl"],
                    subset["clipped"],
                ]
            )
            figure.add_trace(
                go.Scatter(
                    x=subset["decision_day"],
                    y=subset["price"],
                    name=side.title(),
                    mode="markers",
                    marker={"symbol": marker_symbol, "size": subset["marker_size"]},
                    customdata=custom,
                    hovertemplate=(
                        "Day %{x}<br>Price: %{y:,.4f}<br>Order: %{customdata[0]:,.0f}"
                        "<br>New position: %{customdata[1]:,.0f}"
                        "<br>Dollar volume: $%{customdata[2]:,.2f}"
                        "<br>Commission: $%{customdata[3]:,.4f}"
                        "<br>Next-day PnL: $%{customdata[4]:,.2f}"
                        "<br>Next-day cumulative PnL: $%{customdata[5]:,.2f}"
                        "<br>Clipped: %{customdata[6]}<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

    figure.add_trace(
        go.Scatter(
            x=positions.index,
            y=positions,
            name="Target position",
            mode="lines",
            line_shape="hv",
            opacity=0.6,
            hovertemplate="Day %{x}<br>Target position: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.update_xaxes(title_text="Day")
    figure.update_yaxes(title_text="Price", secondary_y=False)
    figure.update_yaxes(title_text="Shares", secondary_y=True)
    figure.update_layout(title=f"{symbol}: price, orders, and target position")
    return _standard_layout(figure, height=520)


def _instrument_pnl_figure(result: BacktestResult, symbol: str) -> go.Figure:
    pnl = result.daily_pnl[symbol]
    cumulative = pnl.cumsum()
    drawdown = instrument_drawdowns(result)[symbol]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
    )
    figure.add_trace(go.Bar(x=pnl.index, y=pnl, name="Daily PnL"), row=1, col=1)
    figure.add_trace(
        go.Scatter(x=cumulative.index, y=cumulative, name="Cumulative PnL"),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name="Drawdown",
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="PnL ($)", row=1, col=1)
    figure.update_yaxes(title_text="Drawdown ($)", row=2, col=1)
    figure.update_xaxes(title_text="Day", row=2, col=1)
    figure.update_layout(title=f"{symbol}: PnL")
    return _standard_layout(figure, height=540)


def _contribution_figure(result: BacktestResult, mode: str) -> go.Figure:
    summary = result.instrument_summary.copy()
    if mode == "Net PnL":
        value_column = "cumulative_pnl"
        title = "Instrument net PnL contribution"
        y_title = "Cumulative PnL ($)"
    else:
        value_column = "absolute_contribution_share"
        title = "Absolute contribution share"
        y_title = "Share of absolute PnL"

    summary = summary.sort_values(value_column, ascending=False)
    figure = go.Figure(
        go.Bar(
            x=summary["symbol"],
            y=summary[value_column],
            customdata=np.column_stack(
                [
                    summary["score_2026"],
                    summary["sharpe"],
                    summary["max_drawdown"],
                    summary["dollar_volume"],
                ]
            ),
            hovertemplate=(
                "%{x}<br>Value: %{y:,.4f}<br>Score 2026: %{customdata[0]:,.4f}"
                "<br>Sharpe: %{customdata[1]:,.2f}"
                "<br>Max drawdown: $%{customdata[2]:,.2f}"
                "<br>Dollar volume: $%{customdata[3]:,.0f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(title=title)
    figure.update_xaxes(title="Instrument")
    figure.update_yaxes(title=y_title, tickformat=".0%" if mode != "Net PnL" else None)
    return _standard_layout(figure, height=490, hovermode="closest")


def _correlation_figure(matrix: pd.DataFrame, title: str) -> go.Figure:
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=matrix.columns,
            y=matrix.index,
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": "Correlation"},
            hovertemplate="%{x} vs %{y}<br>%{z:.3f}<extra></extra>",
        )
    )
    figure.update_layout(title=title)
    return _standard_layout(figure, height=650, hovermode="closest")


# ---------------------------------------------------------------------------
# Pages — exactly one of these is executed on each rerun
# ---------------------------------------------------------------------------



def _overview_page(result: BacktestResult, low_memory: bool) -> None:
    del low_memory
    st.subheader("Portfolio overview")
    _metric_row(
        [
            ("Score 2026", _number(_summary_value(result, "score_2026"), 4), "Official 2026 score"),
            ("Score 2025", _number(_summary_value(result, "score_2025"), 4), "Mean PnL minus 10% of volatility"),
            ("Cumulative PnL", _money(_summary_value(result, "cumulative_pnl")), None),
            ("Annualised Sharpe", _number(_summary_value(result, "sharpe"), 2), None),
            ("Maximum drawdown", _money(_summary_value(result, "max_drawdown")), None),
            ("Return on volume", _percent(_summary_value(result, "return"), 3), None),
            ("Dollar volume", _money(_summary_value(result, "dollar_volume")), None),
            ("Commission", _money(_summary_value(result, "commission")), None),
            ("Profitable days", _percent(_summary_value(result, "profitable_day_pct")), None),
            ("Profitable instruments", f"{int(_summary_value(result, 'profitable_instruments', 0))}", None),
            ("Orders", f"{int(_summary_value(result, 'number_of_orders', 0)):,}", None),
            ("Runtime", f"{_safe_scalar(result.timings.get('total_seconds'), 0):.3f}s", None),
        ]
    )

    if result.warnings:
        with st.expander(f"Run warnings ({len(result.warnings)})", expanded=True):
            for warning in result.warnings:
                st.warning(warning)

    _show_plot(_pnl_drawdown_figure(result), "overview-pnl-drawdown")

    ranking_metric = st.radio(
        "Rank instruments by",
        ["score_2026", "score_2025"],
        horizontal=True,
        format_func=lambda value: value.replace("_", " ").title(),
        key="overview-ranking-metric",
    )
    best = result.best_2026 if ranking_metric == "score_2026" else result.best_2025
    worst = result.worst_2026 if ranking_metric == "score_2026" else result.worst_2025
    left, right = st.columns(2)
    with left:
        st.markdown("#### Best instruments")
        _dataframe(best, height=300, key="overview-best")
    with right:
        st.markdown("#### Worst instruments")
        _dataframe(worst, height=300, key="overview-worst")


def _portfolio_page(result: BacktestResult, low_memory: bool) -> None:
    st.subheader("Portfolio analysis")
    chart = st.selectbox(
        "Chart",
        [
            "Daily and cumulative PnL",
            "Daily PnL",
            "Cumulative PnL",
            "Exposure",
            "Turnover and costs",
            "Rolling risk",
        ],
        key="portfolio-chart",
    )
    if chart in {"Daily and cumulative PnL", "Daily PnL", "Cumulative PnL"}:
        _show_plot(_portfolio_pnl_figure(result, chart), "portfolio-pnl")
    elif chart == "Exposure":
        _show_plot(_exposure_figure(result), "portfolio-exposure")
    elif chart == "Turnover and costs":
        _show_plot(_turnover_figure(result), "portfolio-turnover")
    else:
        max_window = min(100, len(result.portfolio_daily))
        if max_window < 2:
            st.info("At least two scored days are required for rolling risk.")
        else:
            window = st.slider(
                "Rolling window",
                min_value=2,
                max_value=max_window,
                value=min(20, max_window),
                key="portfolio-risk-window",
            )
            _show_plot(_rolling_risk_figure(result, int(window)), "portfolio-risk")

    recovery = (
        result.portfolio_summary.at["drawdown_recovery_day", "value"]
        if "drawdown_recovery_day" in result.portfolio_summary.index
        else None
    )
    if pd.isna(recovery):
        recovery_text = "Not recovered"
    else:
        recovery_text = str(int(float(recovery)))
    _metric_row(
        [
            ("Peak day", _number(_summary_value(result, "drawdown_start_day"), 0), None),
            ("Trough day", _number(_summary_value(result, "drawdown_trough_day"), 0), None),
            ("Recovery day", recovery_text, None),
            ("Duration", f"{int(_summary_value(result, 'drawdown_duration', 0))} days", None),
            ("Current drawdown", _money(_summary_value(result, "current_drawdown")), None),
        ]
    )

    st.markdown("#### Daily portfolio data")
    page_size = LOW_MEMORY_PAGE_SIZE if low_memory else DEFAULT_PAGE_SIZE
    _paginated_frame(
        result.portfolio_daily.reset_index(),
        key="portfolio-daily",
        page_size=page_size,
        height=430,
    )



def _instrument_page(result: BacktestResult, low_memory: bool) -> None:
    st.subheader("Instrument explorer")
    symbol = st.selectbox("Instrument", list(result.symbols), key="instrument-symbol")
    summary = result.instrument_summary.set_index("symbol").loc[symbol]
    _metric_row(
        [
            ("Cumulative PnL", _money(summary.get("cumulative_pnl")), None),
            ("Mean daily PnL", _money(summary.get("mean_pnl")), None),
            ("Daily PnL volatility", _money(summary.get("stddev_pnl")), None),
            ("Sharpe", _number(summary.get("sharpe"), 2), None),
            ("Score 2026", _number(summary.get("score_2026"), 4), None),
            ("Score 2025", _number(summary.get("score_2025"), 4), None),
            ("Maximum drawdown", _money(summary.get("max_drawdown")), None),
            ("Dollar volume", _money(summary.get("dollar_volume")), None),
            ("Commission", _money(summary.get("commission")), None),
            ("Profitable days", _percent(summary.get("profitable_day_pct")), None),
            ("Average |position|", _number(summary.get("average_abs_position"), 1), None),
            ("Maximum |position|", _number(summary.get("maximum_abs_position"), 0), None),
        ]
    )

    _show_plot(_instrument_price_position_figure(result, str(symbol)), "instrument-price-position")
    _show_plot(_instrument_pnl_figure(result, str(symbol)), "instrument-pnl")

    st.markdown("#### Instrument comparison")
    default_symbols = list(
        result.instrument_summary.nlargest(3, "absolute_contribution_share")["symbol"]
    )
    max_choices = 8 if low_memory else 10
    selected = st.multiselect(
        "Compare instruments",
        list(result.symbols),
        default=default_symbols,
        max_selections=max_choices,
        key="instrument-compare-symbols",
    )
    metric = st.selectbox(
        "Comparison metric",
        ["Normalised price", "Cumulative PnL", "Rolling Sharpe", "Position", "Drawdown"],
        key="instrument-compare-metric",
    )
    if not selected:
        return

    if metric == "Normalised price":
        comparison = normalized_prices(result, selected)
    elif metric == "Cumulative PnL":
        comparison = result.daily_pnl[selected].cumsum()
    elif metric == "Position":
        comparison = result.daily_positions[selected]
    elif metric == "Drawdown":
        comparison = instrument_drawdowns(result)[selected]
    else:
        max_window = min(100, len(result.daily_pnl))
        if max_window < 2:
            st.info("At least two scored days are required for rolling Sharpe.")
            return
        window = st.slider(
            "Comparison rolling window",
            min_value=2,
            max_value=max_window,
            value=min(20, max_window),
            key="instrument-compare-window",
        )
        comparison = pd.DataFrame(
            {name: rolling_sharpe(result.daily_pnl[name], int(window)) for name in selected}
        )

    figure = go.Figure()
    for name in comparison.columns:
        figure.add_trace(go.Scatter(x=comparison.index, y=comparison[name], name=str(name), mode="lines"))
    figure.update_layout(title=f"Instrument comparison: {metric}")
    figure.update_xaxes(title="Day")
    _show_plot(_standard_layout(figure, height=500), "instrument-comparison")



def _trade_page(result: BacktestResult, low_memory: bool) -> None:
    st.subheader("Trade explorer")
    day = st.slider(
        "Scored day",
        min_value=int(result.start_day),
        max_value=int(result.end_day),
        value=int(result.end_day),
        step=1,
        key="trade-day",
    )
    orders_only = st.checkbox("Show only instruments with orders", value=True, key="trade-orders-only")
    snapshot = selected_day_snapshot(result, int(day))
    _metric_row(
        [
            ("Daily portfolio PnL", _money(snapshot["daily_pnl"]), None),
            ("Orders", f"{snapshot['number_of_orders']:,}", None),
            ("Dollar volume", _money(snapshot["dollar_volume"]), None),
            ("Gross exposure", _money(snapshot["gross_exposure"]), None),
            ("Net exposure", _money(snapshot["net_exposure"]), None),
            ("Long / short", f"{snapshot['long_positions']} / {snapshot['short_positions']}", None),
        ]
    )
    st.caption(
        f"Largest position: **{snapshot['largest_position']}** · "
        f"Largest contributor: **{snapshot['largest_contributor']}** · "
        f"Largest detractor: **{snapshot['largest_detractor']}**"
    )

    table = selected_day_table(result, int(day), orders_only=orders_only).copy()
    if "position_limit_utilisation" in table.columns:
        table["position_limit_utilisation"] *= 100.0
    _dataframe(table, height=570, key="trade-day-table")

    with st.expander("Complete executed trade log", expanded=False):
        st.caption(
            "The full log is paginated to avoid the native Arrow conversion path that caused the crash."
        )
        page_size = LOW_MEMORY_PAGE_SIZE if low_memory else DEFAULT_PAGE_SIZE
        trade_log_display = result.trade_log.copy()
        if "position_limit_utilisation" in trade_log_display.columns:
            trade_log_display["position_limit_utilisation"] *= 100.0
        _paginated_frame(
            trade_log_display,
            key="trade-full-log",
            page_size=page_size,
            height=540,
        )
        st.caption(
            f"The initial position is opened on decision day {result.start_day - 1}, "
            "one day before the first scored PnL day, matching the official evaluator."
        )



def _diagnostics_page(result: BacktestResult, low_memory: bool) -> None:
    st.subheader("Diagnostics")
    mode = st.radio(
        "Contribution view",
        ["Net PnL", "Absolute PnL share"],
        horizontal=True,
        key="diagnostics-contribution-mode",
    )
    _show_plot(_contribution_figure(result, mode), "diagnostics-contribution")

    _metric_row(
        [
            ("Top-1 concentration", _percent(_summary_value(result, "top_1_absolute_pnl_concentration")), None),
            ("Top-5 concentration", _percent(_summary_value(result, "top_5_absolute_pnl_concentration")), None),
            ("PnL concentration HHI", _number(_summary_value(result, "pnl_concentration_hhi"), 3), None),
            ("Profit factor", _number(_summary_value(result, "profit_factor"), 2), None),
            ("95% VaR", _money(_summary_value(result, "var_95")), None),
            ("95% expected shortfall", _money(_summary_value(result, "expected_shortfall_95")), None),
        ]
    )

    st.markdown("#### Correlation heatmap")
    kind = st.selectbox(
        "Correlation type",
        ["PnL", "Returns", "Positions", "Trade direction"],
        key="diagnostics-correlation-kind",
    )
    correlation_max = min(MAX_CORRELATION_INSTRUMENTS, len(result.symbols))
    if correlation_max >= 2:
        default_count = min(
            LOW_MEMORY_CORRELATION_INSTRUMENTS if low_memory else correlation_max,
            correlation_max,
        )
        requested = st.slider(
            "Maximum instruments in heatmap",
            min_value=2,
            max_value=correlation_max,
            value=default_count,
            key="diagnostics-correlation-count",
        )
        if st.button("Build / refresh correlation heatmap", key="diagnostics-build-correlation"):
            matrix = correlation_matrix(result, kind)
            if matrix.empty:
                st.info("There are not enough non-constant series for this matrix.")
            else:
                active_order = list(
                    result.instrument_summary.sort_values(
                        "absolute_contribution_share", ascending=False
                    )["symbol"]
                )
                selected = [name for name in active_order if name in matrix.columns][: int(requested)]
                matrix = matrix.loc[selected, selected]
                _show_plot(_correlation_figure(matrix, f"{kind} correlation"), "diagnostics-correlation")
    else:
        st.info("At least two instruments are required for a correlation matrix.")

    st.markdown("#### Daily PnL distribution")
    pnl = result.portfolio_daily["daily_pnl"].astype(float)
    left, right = st.columns([2, 1])
    with left:
        histogram = go.Figure(
            go.Histogram(
                x=pnl,
                nbinsx=min(40, max(10, int(np.sqrt(len(pnl)) * 2))),
                name="Daily PnL",
            )
        )
        histogram.update_layout(title="Daily portfolio PnL histogram")
        histogram.update_xaxes(title="Daily PnL ($)")
        histogram.update_yaxes(title="Days")
        _show_plot(_standard_layout(histogram, height=420, hovermode="closest"), "diagnostics-histogram")
    with right:
        box = go.Figure(go.Box(y=pnl, name="Daily PnL", boxpoints="outliers"))
        box.update_layout(title="PnL box plot", showlegend=False)
        box.update_yaxes(title="Daily PnL ($)")
        _show_plot(_standard_layout(box, height=420, hovermode="closest"), "diagnostics-box")

    stats = pd.DataFrame(
        {
            "Metric": [
                "Average winning day",
                "Average losing day",
                "Largest winning day",
                "Largest losing day",
                "Worst five-day PnL",
                "Longest winning streak",
                "Longest losing streak",
                "Skewness",
                "Excess kurtosis",
                "Approx. holding period",
            ],
            "Value": [
                _money(_summary_value(result, "average_winning_day")),
                _money(_summary_value(result, "average_losing_day")),
                _money(_summary_value(result, "largest_winning_day")),
                _money(_summary_value(result, "largest_losing_day")),
                _money(_summary_value(result, "worst_five_day")),
                f"{int(_summary_value(result, 'longest_winning_streak', 0))} days",
                f"{int(_summary_value(result, 'longest_losing_streak', 0))} days",
                _number(_summary_value(result, "skewness"), 3),
                _number(_summary_value(result, "excess_kurtosis"), 3),
                f"{_number(_summary_value(result, 'holding_period_approx_days'), 2)} days",
            ],
        }
    )
    _dataframe(stats, height=420, key="diagnostic-statistics")



def _comparison_page(result: BacktestResult, low_memory: bool) -> None:
    st.subheader("Strategy comparison")
    baseline: BacktestResult | None = st.session_state.get("baseline_result")
    if baseline is None:
        st.info("Save the current run as the baseline, then run another configuration.")
        return

    comparison = compare_results(
        result,
        baseline,
        current_name=str(st.session_state.get("current_name", "Current")),
        baseline_name=str(st.session_state.get("baseline_name", "Baseline")),
    )
    _dataframe(comparison.summary.reset_index(), height=350, key="comparison-summary")

    cumulative = go.Figure()
    for name in comparison.cumulative_pnl.columns:
        cumulative.add_trace(
            go.Scatter(
                x=comparison.cumulative_pnl.index,
                y=comparison.cumulative_pnl[name],
                name=str(name),
                mode="lines",
            )
        )
    cumulative.update_layout(title="Cumulative PnL comparison")
    cumulative.update_xaxes(title="Day")
    cumulative.update_yaxes(title="Cumulative PnL ($)")
    _show_plot(_standard_layout(cumulative, height=490), "comparison-cumulative")

    drawdown = go.Figure()
    for name in comparison.drawdown.columns:
        drawdown.add_trace(
            go.Scatter(
                x=comparison.drawdown.index,
                y=comparison.drawdown[name],
                name=str(name),
                mode="lines",
            )
        )
    drawdown.update_layout(title="Drawdown comparison")
    drawdown.update_xaxes(title="Day")
    drawdown.update_yaxes(title="Drawdown ($)")
    _show_plot(_standard_layout(drawdown, height=430), "comparison-drawdown")

    st.markdown("#### Per-instrument differences: current minus baseline")
    page_size = LOW_MEMORY_PAGE_SIZE if low_memory else DEFAULT_PAGE_SIZE
    _paginated_frame(
        comparison.instrument_differences.reset_index(),
        key="comparison-instruments",
        page_size=page_size,
        height=500,
    )



def _audit_page(result: BacktestResult, low_memory: bool) -> None:
    st.subheader("Run diagnostics")
    _metric_row(
        [
            ("Run ID", str(result.audit.get("run_id", "—")), None),
            ("Backtester", str(result.audit.get("backtester_version", "—")), None),
            ("Total runtime", f"{_safe_scalar(result.timings.get('total_seconds'), 0):.4f}s", None),
            ("Strategy time", f"{_safe_scalar(result.timings.get('strategy_seconds'), 0):.4f}s", None),
            (
                "Average strategy call",
                f"{_safe_scalar(result.timings.get('average_strategy_call_seconds'), 0) * 1000:.3f}ms",
                None,
            ),
        ]
    )

    st.markdown("#### Reproducibility metadata")
    with st.expander("Show run metadata", expanded=False):
        metadata = pd.DataFrame(
            {"Field": list(result.audit), "Value": [_jsonish(v) for v in result.audit.values()]}
        )
        _dataframe(metadata, key="audit-metadata", height=420)

    st.markdown("#### Strategy call performance")
    if result.strategy_calls.empty:
        st.info("No strategy calls were recorded.")
        return
    calls = result.strategy_calls.copy()
    calls["elapsed_ms"] = calls["elapsed_seconds"] * 1000.0
    timing = go.Figure(
        go.Scatter(
            x=calls["decision_day"],
            y=calls["elapsed_ms"],
            mode="lines+markers",
            name="Execution time",
        )
    )
    timing.update_layout(title="Strategy execution time by decision day")
    timing.update_xaxes(title="Decision day")
    timing.update_yaxes(title="Milliseconds")
    _show_plot(_standard_layout(timing, height=420), "audit-strategy-time")
    page_size = LOW_MEMORY_PAGE_SIZE if low_memory else DEFAULT_PAGE_SIZE
    _paginated_frame(calls, key="audit-calls", page_size=page_size, height=420)


PAGE_RENDERERS: dict[str, Callable[[BacktestResult, bool], None]] = {
    "Overview": _overview_page,
    "Portfolio Analysis": _portfolio_page,
    "Instrument Explorer": _instrument_page,
    "Trade Explorer": _trade_page,
    "Diagnostics": _diagnostics_page,
    "Comparison": _comparison_page,
    "Run Diagnostics": _audit_page,
}


# ---------------------------------------------------------------------------
# Sidebar configuration and execution
# ---------------------------------------------------------------------------


def _sidebar_configuration() -> tuple[dict[str, Any] | None, bool]:
    st.sidebar.title("Backtest configuration")
    st.sidebar.caption(
        f"Backtester {BACKTESTER_VERSION}. Only trusted local strategy code should be run."
    )

    prices_path = st.sidebar.text_input(
        "Prices file",
        value=str(DEFAULT_PRICES_PATH),
        key="config-prices-path",
    )
    source_mode = st.sidebar.radio(
        "Strategy source",
        ["Python file", "Importable module"],
        key="config-source-mode",
    )
    if source_mode == "Python file":
        strategy_file: str | None = st.sidebar.text_input(
            "Strategy file",
            value=str(DEFAULT_STRATEGY_PATH),
            key="config-strategy-file",
        )
        strategy_module = "teamName"
    else:
        strategy_file = None
        strategy_module = st.sidebar.text_input(
            "Strategy module",
            value="teamName",
            key="config-strategy-module",
        )

    symbols: tuple[str, ...] = ()
    n_days = 0
    preview_error: str | None = None
    try:
        resolved, size, mtime_ns = _path_signature(prices_path)
        symbols, n_days, n_instruments = _preview_prices(resolved, size, mtime_ns)
        st.sidebar.success(f"{n_instruments} instruments · {n_days} days")
    except Exception as exc:
        preview_error = str(exc)
        st.sidebar.warning(f"Price preview unavailable: {preview_error}")

    default_end = max(1, n_days - 1)
    default_start = max(1, default_end - 249)

    with st.sidebar.form("backtest-configuration-form", clear_on_submit=False):
        run_name = st.text_input("Run name", value="Current run")
        position_function = st.text_input(
            "Position function",
            value="",
            placeholder="Auto-detect getMyPosition / getMyPos",
        )
        start_day = st.number_input(
            "First scored day",
            min_value=1,
            max_value=default_end,
            value=default_start,
            step=1,
        )
        end_day = st.number_input(
            "Last scored day",
            min_value=1,
            max_value=default_end,
            value=default_end,
            step=1,
        )

        universe_mode = st.selectbox(
            "Universe",
            ["All instruments", "Include only", "Exclude"],
        )
        selected_instruments: list[str] = []
        if universe_mode != "All instruments":
            selected_instruments = st.multiselect(
                "Instruments",
                options=list(symbols),
                default=[],
            )

        with st.expander("Advanced settings", expanded=False):
            readonly_history = st.checkbox(
                "Protect price history from mutation",
                value=True,
            )
            score_param = st.number_input(
                "2026 score parameter",
                min_value=0.0,
                value=1.0,
            )
            ranking_count = st.number_input(
                "Best/worst count",
                min_value=1,
                value=5,
                step=1,
            )
            ranking_metric = st.selectbox(
                "Default ranking metric",
                ["score_2026", "score_2025"],
            )
            strategy_parameters_text = st.text_area(
                "Strategy parameters JSON",
                value="{}",
                height=90,
            )
            commission_overrides_text = st.text_area(
                "Commission overrides JSON",
                value="{}",
                height=75,
            )
            position_limit_overrides_text = st.text_area(
                "Position-limit overrides JSON",
                value="{}",
                height=75,
            )
            default_commission = st.number_input(
                "Default commission rate",
                min_value=0.0,
                value=0.0001,
                format="%.8f",
            )
            instrument0_commission = st.number_input(
                "Instrument 0 commission rate",
                min_value=0.0,
                value=0.00002,
                format="%.8f",
            )
            default_limit = st.number_input(
                "Default dollar position limit",
                min_value=1.0,
                value=10_000.0,
            )
            instrument0_limit = st.number_input(
                "Instrument 0 dollar position limit",
                min_value=1.0,
                value=100_000.0,
            )

        submitted = st.form_submit_button(
            "Run backtest",
            type="primary",
            disabled=preview_error is not None,
        )

    payload: dict[str, Any] | None = None
    if submitted:
        try:
            if int(start_day) > int(end_day):
                raise ValueError("First scored day must not exceed last scored day.")
            if universe_mode != "All instruments" and not selected_instruments:
                raise ValueError("Select at least one instrument for the chosen universe mode.")

            include = selected_instruments if universe_mode == "Include only" else None
            exclude = selected_instruments if universe_mode == "Exclude" else None
            payload = {
                "run_name": run_name.strip() or "Current run",
                "prices_path": str(Path(prices_path).expanduser().resolve()),
                "strategy_module": strategy_module.strip() or "teamName",
                "strategy_file": (
                    str(Path(strategy_file).expanduser().resolve())
                    if strategy_file
                    else None
                ),
                "position_function": position_function.strip() or None,
                "start_day": int(start_day),
                "end_day": int(end_day),
                "num_test_days": int(end_day) - int(start_day) + 1,
                "include_instruments": tuple(include) if include is not None else None,
                "exclude_instruments": tuple(exclude) if exclude is not None else None,
                "score_param": float(score_param),
                "ranking_count": int(ranking_count),
                "default_commission_rate": float(default_commission),
                "instrument_0_commission_rate": float(instrument0_commission),
                "default_dollar_position_limit": float(default_limit),
                "instrument_0_dollar_position_limit": float(instrument0_limit),
                "commission_overrides": _parse_json_object(
                    commission_overrides_text, "Commission overrides"
                ),
                "position_limit_overrides": _parse_json_object(
                    position_limit_overrides_text, "Position-limit overrides"
                ),
                "strategy_parameters": _parse_json_object(
                    strategy_parameters_text, "Strategy parameters"
                ),
                "readonly_history": bool(readonly_history),
                "ranking_metric": str(ranking_metric),
            }
        except ValueError as exc:
            st.sidebar.error(str(exc))

    st.sidebar.divider()
    low_memory = st.sidebar.checkbox(
        "Compact table mode",
        value=False,
        help="Optional: smaller table pages and heatmaps for low-memory machines.",
        key="display-low-memory",
    )

    current_result: BacktestResult | None = st.session_state.get("current_result")
    if current_result is not None:
        st.sidebar.caption("Use the page navigation shown above the dashboard content.")
        if st.sidebar.button("Save current as baseline", key="save-baseline"):
            st.session_state["baseline_result"] = current_result
            st.session_state["baseline_name"] = st.session_state.get(
                "current_name", "Baseline"
            )
            st.sidebar.success("Baseline saved.")
        if st.sidebar.button("Clear current run", key="clear-current"):
            _clear_large_session_objects(clear_baseline=False)
            st.rerun()
        if st.sidebar.button("Clear all runs", key="clear-all"):
            _clear_large_session_objects(clear_baseline=True)
            st.rerun()

    return payload, bool(low_memory)


def _execute_backtest(payload: Mapping[str, Any]) -> BacktestResult:
    """Build a validated BacktestConfig and execute one uncached run."""
    # Hashing before execution provides an early, clear file error and records
    # which exact source files were used.  The backtester independently records
    # the hashes in its audit metadata.
    file_sha256(payload["prices_path"])
    _strategy_source_hash(payload.get("strategy_file"), str(payload["strategy_module"]))

    config = BacktestConfig(
        prices_path=Path(str(payload["prices_path"])),
        strategy_module=str(payload["strategy_module"]),
        strategy_file=(
            Path(str(payload["strategy_file"]))
            if payload.get("strategy_file")
            else None
        ),
        position_function=(
            str(payload["position_function"])
            if payload.get("position_function")
            else None
        ),
        start_day=int(payload["start_day"]),
        end_day=int(payload["end_day"]),
        num_test_days=int(payload["num_test_days"]),
        include_instruments=payload.get("include_instruments"),
        exclude_instruments=payload.get("exclude_instruments"),
        score_param=float(payload["score_param"]),
        ranking_count=int(payload["ranking_count"]),
        default_commission_rate=float(payload["default_commission_rate"]),
        instrument_0_commission_rate=float(payload["instrument_0_commission_rate"]),
        default_dollar_position_limit=float(payload["default_dollar_position_limit"]),
        instrument_0_dollar_position_limit=float(
            payload["instrument_0_dollar_position_limit"]
        ),
        commission_overrides=dict(payload["commission_overrides"]),
        position_limit_overrides=dict(payload["position_limit_overrides"]),
        strategy_parameters=dict(payload["strategy_parameters"]),
        readonly_history=bool(payload["readonly_history"]),
        ranking_metric=str(payload["ranking_metric"]),
    )
    return run_backtest_from_config(config)


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _init_session_state()

    st.title(APP_TITLE)
    st.caption(
        "Evaluator-compatible scoring with all analysis pages restored, while only "
        "the selected page is rendered on each interaction."
    )

    payload, low_memory = _sidebar_configuration()
    if payload is not None:
        try:
            with st.spinner("Running strategy and building analytics..."):
                result = _execute_backtest(payload)
            st.session_state["current_result"] = result
            st.session_state["current_name"] = str(payload["run_name"])
            st.session_state["last_error"] = None
            st.success(
                f"Backtest complete. Score 2026: {result.summary_value('score_2026'):.4f}"
            )
        except (BacktestError, FileNotFoundError, ValueError, ImportError) as exc:
            st.session_state["last_error"] = str(exc)
            st.error(f"Backtest could not run: {exc}")
        except Exception as exc:
            st.session_state["last_error"] = str(exc)
            st.error(f"Unexpected backtest failure: {exc}")
            if st.checkbox("Show technical details", key="run-error-details"):
                st.code(traceback.format_exc())

    result: BacktestResult | None = st.session_state.get("current_result")
    if result is None:
        st.info(
            "Choose the price and strategy files in the sidebar, then submit the "
            "configuration form."
        )
        st.code(
            "def getMyPosition(prcSoFar):\n"
            "    # prcSoFar shape: (number_of_instruments, history_days)\n"
            "    return target_positions",
            language="python",
        )
        return

    st.caption(
        f"Run **{st.session_state.get('current_name', 'Current run')}** · "
        f"ID `{result.audit.get('run_id', 'unknown')}` · "
        f"days {result.start_day}–{result.end_day} · "
        f"{len(result.active_symbols)}/{len(result.symbols)} active instruments"
    )

    st.markdown("### Dashboard pages")
    selected_page = st.radio(
        "Dashboard page",
        options=list(PAGE_RENDERERS),
        horizontal=True,
        key="nav_page",
        label_visibility="collapsed",
    )
    st.divider()
    renderer = PAGE_RENDERERS.get(str(selected_page), _overview_page)
    try:
        renderer(result, low_memory)
    except Exception as exc:
        st.error(f"The {selected_page} page could not be rendered: {exc}")
        show_details = st.checkbox(
            "Show page error details",
            value=False,
            key="page-error-details",
        )
        if show_details:
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
