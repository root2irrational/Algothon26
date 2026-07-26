"""Reusable, side-effect-free utilities for exploratory market-data analysis.

All tabular inputs follow the pandas convention: rows are observations ordered
through time and columns are instruments. Public functions never mutate inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm

from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf

DataLike: TypeAlias = pd.Series | pd.DataFrame | np.ndarray
ReturnMethod: TypeAlias = Literal["simple", "log", "difference"]


def _as_frame(data: DataLike, *, name: str = "value") -> pd.DataFrame:
    """Return a numeric, finite, non-empty copy of data as a DataFrame."""
    if isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name or name)
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        array = np.asarray(data)
        if array.ndim == 1:
            frame = pd.DataFrame({name: array})
        elif array.ndim == 2:
            frame = pd.DataFrame(array)
        else:
            raise ValueError("data must be one- or two-dimensional")

    if frame.empty or frame.shape[1] == 0:
        raise ValueError("data must contain at least one observation and column")
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError("data must contain only numeric values") from exc
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("data must not contain infinite values")
    return frame


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _restore_type(frame: pd.DataFrame, original: DataLike) -> pd.Series | pd.DataFrame:
    return frame.iloc[:, 0].rename(frame.columns[0]) if isinstance(original, pd.Series) else frame


def instrument_name(data: DataLike, index: int) -> object:
    """Return the instrument column name at a zero-based position."""
    frame = _as_frame(data)
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise TypeError("index must be an integer")
    index = int(index)
    if index < 0 or index >= frame.shape[1]:
        raise IndexError(f"instrument index {index} is outside [0, {frame.shape[1] - 1}]")
    return frame.columns[index]


def instrument_index(data: DataLike, name: object) -> int:
    """Return the zero-based position of an instrument column name."""
    frame = _as_frame(data)
    if not frame.columns.is_unique:
        raise ValueError("instrument names must be unique")
    try:
        location = frame.columns.get_loc(name)
    except KeyError as exc:
        raise KeyError(f"unknown instrument name: {name!r}") from exc
    if not isinstance(location, (int, np.integer)):
        raise ValueError(f"instrument name {name!r} does not resolve to one column")
    return int(location)


def select_instruments(
    data: DataLike,
    instruments: Sequence[object],
) -> pd.DataFrame:
    """Select instruments by zero-based position, column name, or a mixture."""
    frame = _as_frame(data)
    if len(instruments) == 0:
        raise ValueError("instruments must not be empty")
    names = [
        instrument_name(frame, value)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        else value
        for value in instruments
    ]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"unknown instrument name(s): {missing}")
    return frame.loc[:, names].copy()


def calculate_returns(
    prices: DataLike,
    *,
    method: ReturnMethod = "simple",
    periods: int = 1,
    dropna: bool = True,
) -> pd.Series | pd.DataFrame:
    """Calculate simple, logarithmic, or absolute-difference returns."""
    frame = _as_frame(prices, name="price")
    periods = _positive_int(periods, name="periods")

    if method == "simple":
        result = frame.pct_change(periods=periods, fill_method=None)
    elif method == "log":
        if (frame <= 0).any().any():
            raise ValueError("log returns require strictly positive prices")
        result = np.log(frame).diff(periods=periods)
    elif method == "difference":
        result = frame.diff(periods=periods)
    else:
        raise ValueError("method must be 'simple', 'log', or 'difference'")

    if dropna:
        result = result.dropna(how="all")
    return _restore_type(result, prices)


def rolling_statistics(
    returns: DataLike,
    *,
    window: int = 20,
    annualization: int = 250,
    risk_free_rate: float = 0.0,
    min_periods: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Calculate rolling mean, volatility, and annualized Sharpe ratio."""
    frame = _as_frame(returns, name="return")
    window = _positive_int(window, name="window")
    annualization = _positive_int(annualization, name="annualization")
    min_periods = window if min_periods is None else _positive_int(min_periods, name="min_periods")
    if min_periods > window:
        raise ValueError("min_periods cannot exceed window")

    rolling = frame.rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    volatility = rolling.std(ddof=1)
    daily_risk_free = float(risk_free_rate) / annualization
    sharpe = np.sqrt(annualization) * (mean - daily_risk_free) / volatility
    sharpe = sharpe.where(volatility > np.finfo(float).eps)
    return {"mean": mean, "volatility": volatility, "sharpe": sharpe}


def summary_statistics(
    data: DataLike,
    *,
    input_type: Literal["prices", "returns"] = "prices",
    return_method: ReturnMethod = "simple",
    annualization: int = 250,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Return common distribution and risk statistics for each instrument."""
    annualization = _positive_int(annualization, name="annualization")
    if input_type == "prices":
        returns = _as_frame(calculate_returns(data, method=return_method), name="return")
    elif input_type == "returns":
        returns = _as_frame(data, name="return")
    else:
        raise ValueError("input_type must be 'prices' or 'returns'")

    mean = returns.mean()
    volatility = returns.std(ddof=1)
    daily_risk_free = float(risk_free_rate) / annualization
    sharpe = np.sqrt(annualization) * (mean - daily_risk_free) / volatility
    sharpe = sharpe.where(volatility > np.finfo(float).eps)

    result = pd.DataFrame(
        {
            "count": returns.count(),
            "mean": mean,
            "annualized_return": mean * annualization,
            "annualized_volatility": volatility * np.sqrt(annualization),
            "sharpe": sharpe,
            "skew": returns.skew(),
            "kurtosis": returns.kurt(),
            "minimum": returns.min(),
            "median": returns.median(),
            "maximum": returns.max(),
        }
    )
    result.insert(0, "instrument_index", np.arange(len(result), dtype=int))
    result.insert(1, "instrument_name", result.index.astype(str))
    result.index.name = "instrument"
    return result


def correlation_matrix(
    data: DataLike,
    *,
    input_type: Literal["prices", "returns"] = "returns",
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
) -> pd.DataFrame:
    """Calculate an instrument correlation matrix."""
    if input_type == "prices":
        frame = _as_frame(calculate_returns(data), name="return")
    elif input_type == "returns":
        frame = _as_frame(data, name="return")
    else:
        raise ValueError("input_type must be 'prices' or 'returns'")
    return frame.corr(method=method)


def autocorrelation(
    data: DataLike,
    *,
    lags: int | Sequence[int] = 20,
    squared: bool = False,
) -> pd.DataFrame:
    """Calculate per-instrument autocorrelation for one or more positive lags."""
    frame = _as_frame(data, name="value")
    lag_values = range(1, _positive_int(lags, name="lags") + 1) if isinstance(lags, int) else lags
    validated = [_positive_int(lag, name="lag") for lag in lag_values]
    if not validated:
        raise ValueError("lags must not be empty")
    values = frame.pow(2) if squared else frame
    return pd.DataFrame(
        {column: [values[column].autocorr(lag=lag) for lag in validated] for column in values},
        index=pd.Index(validated, name="lag"),
    )


def compare_periods(
    data: DataLike,
    periods: Mapping[str, tuple[object, object]],
    *,
    input_type: Literal["prices", "returns"] = "prices",
    annualization: int = 250,
) -> pd.DataFrame:
    """Compare summary statistics across inclusive index-label periods."""
    frame = _as_frame(data)
    if not periods:
        raise ValueError("periods must not be empty")

    results: list[pd.DataFrame] = []
    for label, bounds in periods.items():
        if len(bounds) != 2:
            raise ValueError(f"period {label!r} must contain (start, end)")
        sample = frame.loc[bounds[0] : bounds[1]]
        if sample.empty:
            raise ValueError(f"period {label!r} contains no observations")
        stats = summary_statistics(sample, input_type=input_type, annualization=annualization)
        stats.insert(0, "period", label)
        stats.insert(1, "instrument", stats.index)
        results.append(stats.reset_index(drop=True))
    return pd.concat(results, ignore_index=True).set_index(["period", "instrument"])


def plot_series(
    data: DataLike,
    *,
    normalize: bool = False,
    log_scale: bool = False,
    subplots: bool = False,
    n_cols: int = 4,
    title: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes | np.ndarray]:
    """Plot time series together or in a compact grid of subplots."""
    frame = _as_frame(data)

    if normalize:
        first = frame.apply(
            lambda column: (
                column.dropna().iloc[0]
                if column.notna().any()
                else np.nan
            )
        )
        if (first == 0).any():
            raise ValueError(
                "cannot normalize a series whose first valid value is zero"
            )
        frame = frame.divide(first).multiply(100)

    ylabel = "Normalized value (base 100)" if normalize else "Value"
    yscale = "log" if log_scale else "linear"

    if subplots:
        if ax is not None:
            raise ValueError("ax cannot be supplied when subplots=True")

        n_cols = _positive_int(n_cols, name="n_cols")
        n_instruments = frame.shape[1]
        n_cols = min(n_cols, n_instruments)
        n_rows = int(np.ceil(n_instruments / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4 * n_cols, 3.5 * n_rows),
            sharex=True,
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for axis, column in zip(flat_axes, frame.columns):
            frame[column].plot(
                ax=axis,
                legend=False,
                linewidth=1.2,
            )
            axis.set_title(str(column))
            axis.set_xlabel("Observation")
            axis.set_ylabel(ylabel)
            axis.set_yscale(yscale)
            axis.grid(alpha=0.25)

        # Hide unused panels in the final row.
        for axis in flat_axes[n_instruments:]:
            axis.set_visible(False)

        fig.suptitle(title or "Time Series", y=1.01)
        fig.tight_layout()
        return fig, axes

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    frame.plot(ax=ax)
    ax.set_title(title or "Time Series")
    ax.set_xlabel("Observation")
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    return fig, ax

def plot_series_indicators(
    prices: DataLike,
    *,
    ema_windows: Sequence[int] | None = (10, 30),
    volatility_window: int | None = 20,
    sharpe_window: int | None = 60,
    annualization: int = 250,
    normalize: bool = False,
    n_cols: int = 4,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """
    Plot each instrument and its selected indicators in a compact grid.

    Pass None to exclude a feature:
      - ema_windows=None
      - volatility_window=None
      - sharpe_window=None

    The returned axes array has shape:
        (instrument_rows, n_cols, panels_per_instrument)
    """
    frame = _as_frame(prices, name="price")

    annualization = _positive_int(
        annualization,
        name="annualization",
    )
    n_cols = _positive_int(
        n_cols,
        name="n_cols",
    )

    windows = (
        [
            _positive_int(window, name="ema_window")
            for window in ema_windows
        ]
        if ema_windows is not None
        else []
    )

    if volatility_window is not None:
        volatility_window = _positive_int(
            volatility_window,
            name="volatility_window",
        )

    if sharpe_window is not None:
        sharpe_window = _positive_int(
            sharpe_window,
            name="sharpe_window",
        )

    display = frame.copy()

    if normalize:
        first = display.apply(
            lambda column: (
                column.dropna().iloc[0]
                if column.notna().any()
                else np.nan
            )
        )

        if (first == 0).any():
            raise ValueError(
                "cannot normalize a series whose first valid value is zero"
            )

        display = display.divide(first).multiply(100)

    # Calculate returns only if a return-based indicator is enabled.
    returns = None

    if volatility_window is not None or sharpe_window is not None:
        returns = _as_frame(
            calculate_returns(frame),
            name="return",
        )

    volatility = None

    if volatility_window is not None:
        volatility = (
            rolling_statistics(
                returns,
                window=volatility_window,
                annualization=annualization,
            )["volatility"]
            * np.sqrt(annualization)
        )

    sharpe = None

    if sharpe_window is not None:
        sharpe = rolling_statistics(
            returns,
            window=sharpe_window,
            annualization=annualization,
        )["sharpe"]

    enabled_panels = ["price"]

    if volatility is not None:
        enabled_panels.append("volatility")

    if sharpe is not None:
        enabled_panels.append("sharpe")

    panel_count = len(enabled_panels)
    n_instruments = frame.shape[1]
    n_cols = min(n_cols, n_instruments)
    instrument_rows = int(np.ceil(n_instruments / n_cols))
    figure_rows = instrument_rows * panel_count

    fig, raw_axes = plt.subplots(
        figure_rows,
        n_cols,
        figsize=(4.5 * n_cols, 2.5 * figure_rows),
        squeeze=False,
    )

    # Reorganize axes as:
    # axes[instrument_row, instrument_column, indicator_panel]
    axes = np.empty(
        (instrument_rows, n_cols, panel_count),
        dtype=object,
    )

    for instrument_row in range(instrument_rows):
        for instrument_col in range(n_cols):
            for panel_index in range(panel_count):
                axes[
                    instrument_row,
                    instrument_col,
                    panel_index,
                ] = raw_axes[
                    instrument_row * panel_count + panel_index,
                    instrument_col,
                ]

    for instrument_position, column in enumerate(frame.columns):
        instrument_row, instrument_col = divmod(
            instrument_position,
            n_cols,
        )

        instrument_axes = axes[
            instrument_row,
            instrument_col,
        ]

        panel_index = 0

        # Price and optional EMAs
        price_axis = instrument_axes[panel_index]
        panel_index += 1

        price_axis.plot(
            display.index,
            display[column],
            label=str(column),
            linewidth=1.5,
        )

        for window in windows:
            ema = display[column].ewm(
                span=window,
                adjust=False,
                min_periods=window,
            ).mean()

            price_axis.plot(
                display.index,
                ema,
                label=f"EMA {window}",
                alpha=0.75,
            )

        price_axis.set_title(str(column))
        price_axis.set_ylabel(
            "Normalized price" if normalize else "Price"
        )
        price_axis.grid(alpha=0.25)

        if windows:
            price_axis.legend(fontsize="x-small")

        # Optional rolling volatility
        if volatility is not None:
            volatility_axis = instrument_axes[panel_index]
            panel_index += 1

            volatility_axis.plot(
                volatility.index,
                volatility[column],
                linewidth=1.2,
            )
            volatility_axis.set_ylabel("Ann. volatility")
            volatility_axis.grid(alpha=0.25)

        # Optional rolling Sharpe
        if sharpe is not None:
            sharpe_axis = instrument_axes[panel_index]
            panel_index += 1

            sharpe_axis.plot(
                sharpe.index,
                sharpe[column],
                linewidth=1.2,
            )
            sharpe_axis.axhline(
                0,
                color="black",
                linewidth=0.8,
                alpha=0.5,
            )
            sharpe_axis.set_ylabel("Rolling Sharpe")
            sharpe_axis.grid(alpha=0.25)

        # Align x-axes within each instrument group.
        for axis in instrument_axes[1:]:
            axis.sharex(price_axis)

        for axis in instrument_axes[:-1]:
            axis.tick_params(labelbottom=False)

        instrument_axes[-1].set_xlabel("Observation")

    # Hide unused instrument positions in the final row.
    total_positions = instrument_rows * n_cols

    for instrument_position in range(
        n_instruments,
        total_positions,
    ):
        instrument_row, instrument_col = divmod(
            instrument_position,
            n_cols,
        )

        for axis in axes[instrument_row, instrument_col]:
            axis.set_visible(False)

    fig.suptitle(
        title or "Prices and indicators",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    return fig, axes


def plot_correlation_matrix(
    data: DataLike,
    *,
    input_type: Literal["prices", "returns"] = "returns",
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    annotate: bool = False,
    ax: Axes | None = None,
) -> tuple[pd.DataFrame, Figure, Axes]:
    """Calculate and plot an instrument correlation matrix."""
    matrix = correlation_matrix(data, input_type=input_type, method=method)
    if ax is None:
        size = max(6.0, min(14.0, 0.45 * len(matrix)))
        fig, ax = plt.subplots(figsize=(size, size))
    else:
        fig = ax.figure

    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    positions = np.arange(len(matrix))
    ax.set_xticks(positions, matrix.columns, rotation=90)
    ax.set_yticks(positions, matrix.index)
    ax.set_title(f"{method.title()} correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if annotate:
        for row, column in np.ndindex(matrix.shape):
            ax.text(column, row, f"{matrix.iat[row, column]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    return matrix, fig, ax


__all__ = [
    "autocorrelation",
    "calculate_returns",
    "compare_periods",
    "correlation_matrix",
    "instrument_index",
    "instrument_name",
    "plot_correlation_matrix",
    "plot_series",
    "plot_series_indicators",
    "rolling_statistics",
    "select_instruments",
    "summary_statistics",
]

def pair_instruments(
    prices: pd.DataFrame,
    *,
    correlation_weight: float = 0.7,
    volatility_weight: float = 0.3,
    correlation_method: Literal["pearson", "spearman"] = "pearson",
    linkage_method: Literal["single", "complete", "average"] = "average",
    ax: Axes | None = None,
) -> list[tuple[object, object]]:
    """Plot a dendrogram and return disjoint pairs with minimum mixed distance."""
    if prices.shape[1] < 2:
        raise ValueError("prices must contain at least two instruments")
    if not prices.columns.is_unique:
        raise ValueError("instrument names must be unique")

    weights = np.asarray(
        [correlation_weight, volatility_weight],
        dtype=float,
    )
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() == 0:
        raise ValueError("weights must be finite, non-negative, and not both zero")
    weights /= weights.sum()

    returns = (
        prices.astype(float)
        .pct_change(fill_method=None)
        .dropna(how="any")
    )
    if len(returns) < 2:
        raise ValueError("insufficient complete return observations")

    correlation = returns.corr(method=correlation_method).to_numpy()
    volatility = returns.std(ddof=1).to_numpy()

    if not np.isfinite(correlation).all():
        raise ValueError("correlation contains invalid values")
    if not np.isfinite(volatility).all() or (volatility <= 0).any():
        raise ValueError("each instrument must have positive finite volatility")

    # High positive correlation produces a small distance.
    correlation_distance = np.sqrt(
        np.clip((1.0 - correlation) / 2.0, 0.0, 1.0)
    )

    # Instruments with similar proportional volatility have a small distance.
    log_volatility = np.log(volatility)
    volatility_distance = np.abs(
        log_volatility[:, None] - log_volatility[None, :]
    )
    if (scale := volatility_distance.max()) > 0:
        volatility_distance /= scale

    distance = (
        weights[0] * correlation_distance
        + weights[1] * volatility_distance
    )
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)

    # Dendrogram visualization.
    tree = linkage(
        squareform(distance, checks=True),
        method=linkage_method,
        optimal_ordering=True,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    dendrogram(
        tree,
        labels=[str(column) for column in prices.columns],
        leaf_rotation=90,
        leaf_font_size=9,
        ax=ax,
    )
    ax.set_title("Instrument Clusters")
    ax.set_xlabel("Instrument")
    ax.set_ylabel("Combined distance")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()

    # Select closest available pairs from the actual distance matrix.
    candidates = sorted(
        (
            (distance[i, j], i, j)
            for i in range(prices.shape[1])
            for j in range(i + 1, prices.shape[1])
        ),
        key=lambda candidate: candidate[0],
    )

    pairs: list[tuple[object, object]] = []
    used: set[int] = set()

    for _, first, second in candidates:
        if first not in used and second not in used:
            pairs.append(
                (prices.columns[first], prices.columns[second])
            )
            used.update((first, second))

    return pairs

def plot_instrument_pairs(
    prices: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    columns: int = 3,
    normalize: bool = True,
    figsize_per_plot: tuple[float, float] = (5.0, 3.5),
) -> tuple[Figure, np.ndarray]:
    """Plot each instrument pair in a separate subplot."""
    if not pairs:
        raise ValueError("pairs must not be empty")
    if columns < 1:
        raise ValueError("columns must be positive")

    missing = {
        instrument
        for pair in pairs
        for instrument in pair
        if instrument not in prices.columns
    }
    if missing:
        raise KeyError(f"unknown instruments: {sorted(missing)}")

    columns = min(columns, len(pairs))
    rows = int(np.ceil(len(pairs) / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(figsize_per_plot[0] * columns, figsize_per_plot[1] * rows),
        squeeze=False,
        sharex=True,
    )

    for axis, (first, second) in zip(axes.flat, pairs):
        pair_prices = prices.loc[:, [first, second]].astype(float)

        if normalize:
            initial = pair_prices.apply(lambda series: series.dropna().iloc[0])
            if (initial == 0).any():
                raise ValueError(f"cannot normalize pair {(first, second)}")
            pair_prices = pair_prices.divide(initial).multiply(100)

        pair_prices.plot(ax=axis, linewidth=1.3)
        axis.set_title(f"{first} vs {second}")
        axis.set_xlabel("Observation")
        axis.set_ylabel("Normalized price" if normalize else "Price")
        axis.grid(alpha=0.25)
        axis.legend()

    for axis in axes.flat[len(pairs):]:
        axis.set_visible(False)

    figure.tight_layout()
    return figure, axes

def plot_rolling_spreads(
    prices: pd.DataFrame,
    pairs: Sequence[tuple[object, object]],
    *,
    rolling_window: int = 60,
    columns: int = 3,
) -> tuple[pd.DataFrame, Figure, np.ndarray]:
    """
    Plot rolling OLS residuals for instrument pairs.

    For each pair (y, x), estimates:
        y_t = alpha_t + beta_t * x_t + residual_t

    Coefficients are shifted by one observation to avoid look-ahead bias.
    """
    if rolling_window < 2:
        raise ValueError("rolling_window must be at least 2")
    if not pairs:
        raise ValueError("pairs must not be empty")

    missing = {
        instrument
        for pair in pairs
        for instrument in pair
        if instrument not in prices.columns
    }
    if missing:
        raise KeyError(f"unknown instruments: {sorted(missing, key=str)}")

    columns = min(columns, len(pairs))
    rows = int(np.ceil(len(pairs) / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.5 * rows),
        squeeze=False,
        sharex=True,
    )

    spreads: dict[tuple[object, object], pd.Series] = {}

    for axis, (y_name, x_name) in zip(axes.flat, pairs):
        pair = prices.loc[:, [y_name, x_name]].astype(float)
        y, x = pair[y_name], pair[x_name]

        rolling = x.rolling(rolling_window, min_periods=rolling_window)
        beta = (
            y.rolling(rolling_window, min_periods=rolling_window).cov(x)
            / rolling.var()
        )
        alpha = (
            y.rolling(rolling_window, min_periods=rolling_window).mean()
            - beta * rolling.mean()
        )

        # Use only coefficients known before the current observation.
        spread = y - alpha.shift(1) - beta.shift(1) * x
        spreads[(y_name, x_name)] = spread

        axis.plot(spread.index, spread, linewidth=1)
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(f"{y_name} − β·{x_name}")
        axis.set_ylabel("OLS residual")
        axis.grid(alpha=0.25)

    for axis in axes.flat[len(pairs):]:
        axis.set_visible(False)

    figure.tight_layout()

    residuals = pd.concat(spreads, axis=1)
    residuals.columns = pd.MultiIndex.from_tuples(
        residuals.columns,
        names=["dependent", "independent"],
    )

    return residuals, figure, axes

def plot_residual_distributions(
    residuals: pd.DataFrame,
    *,
    bins: int = 30,
    columns: int = 3,
) -> tuple[pd.DataFrame, Figure, np.ndarray]:
    """Fit and plot a normal distribution for each residual series."""
    if residuals.empty:
        raise ValueError("residuals must not be empty")
    if bins < 2 or columns < 1:
        raise ValueError("bins must be at least 2 and columns must be positive")

    columns = min(columns, residuals.shape[1])
    rows = int(np.ceil(residuals.shape[1] / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.5 * rows),
        squeeze=False,
    )

    fitted_parameters = []

    for axis, name in zip(axes.flat, residuals.columns):
        values = residuals[name].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        if values.size < 2 or np.std(values) == 0:
            raise ValueError(f"insufficient variation in residuals for {name}")

        mean, standard_deviation = norm.fit(values)

        lower, upper = values.min(), values.max()
        x = np.linspace(lower, upper, 300)
        fitted_density = norm.pdf(
            x,
            loc=mean,
            scale=standard_deviation,
        )

        axis.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.6,
            color="steelblue",
            label="Residuals",
        )
        axis.plot(
            x,
            fitted_density,
            color="darkred",
            linewidth=2,
            label="Fitted normal",
        )
        axis.axvline(mean, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{name}\nμ={mean:.4f}, σ={standard_deviation:.4f}")
        axis.set_xlabel("Residual")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
        axis.legend()

        fitted_parameters.append(
            {
                "pair": name,
                "mean": mean,
                "standard_deviation": standard_deviation,
                "observations": values.size,
            }
        )

    for axis in axes.flat[residuals.shape[1]:]:
        axis.set_visible(False)

    figure.tight_layout()

    fits = pd.DataFrame(fitted_parameters).set_index("pair")
    return fits, figure, axes


def find_mean_reverting_assets(
    prices: pd.DataFrame,
    *,
    window: int = 40,
    threshold: float = 2.0,
    min_signals: int = 10,
) -> pd.DataFrame:
    """
    Find assets exhibiting one-day return mean reversion.

    A signal occurs when today's return is at least `threshold`
    rolling standard deviations from its rolling mean.

    Mean reversion is considered successful when:
        positive deviation today -> negative return tomorrow
        negative deviation today -> positive return tomorrow

    Parameters
    ----------
    prices:
        DataFrame with dates as rows and asset names as columns.
    window:
        Number of prior returns used to estimate mean and volatility.
    threshold:
        Absolute z-score needed to generate a signal.
    min_signals:
        Minimum number of signals required for an asset to qualify.

    Returns
    -------
    pd.DataFrame
        Assets ranked by mean-reversion hit rate.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    if window < 2:
        raise ValueError("window must be at least 2")

    if threshold <= 0:
        raise ValueError("threshold must be positive")

    if min_signals < 1:
        raise ValueError("min_signals must be positive")

    prices = prices.astype(float)

    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive")

    returns = prices.pct_change(fill_method=None)

    # Shift by one so today's return is not included in the
    # distribution against which it is being tested.
    rolling_mean = returns.rolling(
        window=window,
        min_periods=window,
    ).mean().shift(1)

    rolling_std = returns.rolling(
        window=window,
        min_periods=window,
    ).std(ddof=1).shift(1)

    z_scores = (
        returns - rolling_mean
    ) / rolling_std.replace(0, np.nan)

    # Tomorrow's return, aligned with today's signal.
    next_returns = returns.shift(-1)

    results = []

    for asset in prices.columns:
        asset_z = z_scores[asset]
        asset_next_return = next_returns[asset]

        signal_mask = (
            asset_z.abs() >= threshold
        ) & asset_next_return.notna()

        signal_z = asset_z[signal_mask]
        following_returns = asset_next_return[signal_mask]

        signal_count = int(signal_mask.sum())

        if signal_count < min_signals:
            continue

        # Contrarian position:
        # positive z -> short
        # negative z -> long
        direction = -np.sign(signal_z)

        strategy_returns = direction * following_returns

        # Success means that tomorrow's raw return has the
        # opposite sign to today's deviation.
        successful_reversals = (
            np.sign(following_returns)
            == direction
        )

        hit_rate = successful_reversals.mean()
        average_strategy_return = strategy_returns.mean()
        strategy_volatility = strategy_returns.std(ddof=1)

        if (
            np.isfinite(strategy_volatility)
            and strategy_volatility > np.finfo(float).eps
        ):
            t_statistic = (
                average_strategy_return
                / strategy_volatility
                * np.sqrt(signal_count)
            )
        else:
            t_statistic = np.nan

        positive_signals = signal_z > 0
        negative_signals = signal_z < 0

        positive_hit_rate = (
            (following_returns[positive_signals] < 0).mean()
            if positive_signals.any()
            else np.nan
        )

        negative_hit_rate = (
            (following_returns[negative_signals] > 0).mean()
            if negative_signals.any()
            else np.nan
        )

        results.append(
            {
                "asset": asset,
                "signals": signal_count,
                "hit_rate": hit_rate,
                "positive_deviation_hit_rate": positive_hit_rate,
                "negative_deviation_hit_rate": negative_hit_rate,
                "average_next_day_profit": average_strategy_return,
                "median_next_day_profit": strategy_returns.median(),
                "t_statistic": t_statistic,
            }
        )

    if not results:
        return pd.DataFrame(
            columns=[
                "signals",
                "hit_rate",
                "positive_deviation_hit_rate",
                "negative_deviation_hit_rate",
                "average_next_day_profit",
                "median_next_day_profit",
                "t_statistic",
            ]
        )

    return (
        pd.DataFrame(results)
        .set_index("asset")
        .sort_values(
            ["hit_rate", "average_next_day_profit"],
            ascending=False,
        )
    )


def majority_return_forecast(prices, target, plot=True):
    """
    Predict the target's return every day using the majority of peer
    instruments' returns from the previous day.

    Rows must be ordered oldest to newest. A date index is not required.
    """
    if target not in prices.columns:
        raise ValueError(f"{target!r} is not in prices.columns.")

    if len(prices) < 3:
        raise ValueError("At least three price observations are required.")

    returns = prices.pct_change(fill_method=None)
    peers = returns.drop(columns=target)

    def majority_prediction(peer_returns):
        peer_returns = peer_returns.dropna()

        if peer_returns.empty:
            return np.nan

        positive = peer_returns[peer_returns > 0]
        negative = peer_returns[peer_returns < 0]

        if len(positive) > len(negative):
            return positive.median()

        if len(negative) > len(positive):
            return negative.median()

        return 0.0

    # Prediction made from each day's peer returns.
    signals = peers.apply(majority_prediction, axis=1)

    # Shift forward: peers on day t predict the target on day t+1.
    daily_predictions = signals.shift(1)

    results = pd.DataFrame({
        "prediction": daily_predictions,
        "actual": returns[target],
    }).dropna()

    results["predicted_direction"] = np.sign(results["prediction"])
    results["actual_direction"] = np.sign(results["actual"])

    # Ignore tied votes when evaluating directional accuracy.
    active = results["predicted_direction"] != 0

    results["correct"] = np.nan
    results.loc[active, "correct"] = (
        results.loc[active, "predicted_direction"]
        == results.loc[active, "actual_direction"]
    ).astype(float)

    results["strategy_return"] = (
        results["predicted_direction"] * results["actual"]
    )
    results["strategy_growth"] = (
        1 + results["strategy_return"]
    ).cumprod()
    results["target_growth"] = (
        1 + results["actual"]
    ).cumprod()

    results["rolling_accuracy"] = (
        results["correct"]
        .rolling(20, min_periods=5)
        .mean()
    )

    # Latest peers predict the unobserved next day.
    tomorrow_forecast = signals.iloc[-1]

    directional_accuracy = results["correct"].mean()

    diagnostics = {
        "tomorrow_forecast": tomorrow_forecast,
        "tomorrow_direction": np.sign(tomorrow_forecast),
        "directional_accuracy": directional_accuracy,
        "correlation": results["prediction"].corr(results["actual"]),
        "mae": (
            results["prediction"] - results["actual"]
        ).abs().mean(),
        "number_of_predictions": len(results),
        "results": results,
    }

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        results[["prediction", "actual"]].plot(
            ax=axes[0, 0],
            alpha=0.7,
        )
        axes[0, 0].axhline(0, color="grey", linewidth=0.8)
        axes[0, 0].set_title("Daily predicted vs actual returns")
        axes[0, 0].set_ylabel("Return")

        results["rolling_accuracy"].plot(
            ax=axes[0, 1],
            color="navy",
        )
        axes[0, 1].axhline(
            0.5,
            color="grey",
            linestyle="--",
            label="50% reference",
        )
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title(
            f"20-day directional accuracy\n"
            f"Overall: {directional_accuracy:.2%}"
        )
        axes[0, 1].legend()

        results[["strategy_growth", "target_growth"]].plot(
            ax=axes[1, 0]
        )
        axes[1, 0].set_title("Majority strategy vs target")
        axes[1, 0].set_ylabel("Growth of $1")

        axes[1, 1].scatter(
            results["prediction"],
            results["actual"],
            alpha=0.4,
        )
        axes[1, 1].axhline(0, color="grey", linewidth=0.8)
        axes[1, 1].axvline(0, color="grey", linewidth=0.8)
        axes[1, 1].set_title(
            f"Predicted vs actual\n"
            f"Correlation: {diagnostics['correlation']:.3f}"
        )
        axes[1, 1].set_xlabel("Predicted return")
        axes[1, 1].set_ylabel("Actual return")

        fig.suptitle(
            f"{target} — tomorrow forecast: "
            f"{tomorrow_forecast:.3%}"
        )
        fig.tight_layout()
        plt.show()

        diagnostics["figure"] = fig

    return tomorrow_forecast, diagnostics

def adaptive_majority_return_forecast(
    prices,
    target,
    lookback=20,
    min_history=None,
    plot=True,
):
    """
    Predict the target's daily return from the previous day's peer returns.

    If the base majority model's trailing directional accuracy is below 50%,
    invert the prediction. Otherwise, retain the usual prediction.

    Rows must be ordered from oldest to newest.
    """
    if target not in prices.columns:
        raise ValueError(f"{target!r} is not in prices.columns.")

    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    if min_history is None:
        min_history = lookback

    returns = prices.pct_change(fill_method=None)
    peers = returns.drop(columns=target)

    def majority_prediction(peer_returns):
        peer_returns = peer_returns.dropna()

        if peer_returns.empty:
            return np.nan

        positive = peer_returns[peer_returns > 0]
        negative = peer_returns[peer_returns < 0]

        if len(positive) > len(negative):
            return float(positive.median())

        if len(negative) > len(positive):
            return float(negative.median())

        return 0.0

    # Peer returns on row t create the base forecast for row t+1.
    peer_signals = peers.apply(majority_prediction, axis=1)
    base_prediction = peer_signals.shift(1)

    results = pd.DataFrame({
        "actual": returns[target],
        "base_prediction": base_prediction,
    }).dropna()

    results["actual_direction"] = np.sign(results["actual"])
    results["base_direction"] = np.sign(results["base_prediction"])

    # Evaluate the unmodified majority model.
    results["base_correct"] = np.where(
        results["base_direction"] != 0,
        (
            results["base_direction"]
            == results["actual_direction"]
        ).astype(float),
        np.nan,
    )

    # Shift by one so today's inversion decision never uses today's outcome.
    results["prior_base_accuracy"] = (
        results["base_correct"]
        .rolling(
            window=lookback,
            min_periods=min_history,
        )
        .mean()
        .shift(1)
    )

    # Use the usual prediction until enough accuracy history exists.
    results["invert"] = (
        results["prior_base_accuracy"].notna()
        & (results["prior_base_accuracy"] < 0.5)
    )

    results["prediction"] = np.where(
        results["invert"],
        -results["base_prediction"],
        results["base_prediction"],
    )

    results["predicted_direction"] = np.sign(results["prediction"])

    results["correct"] = np.where(
        results["predicted_direction"] != 0,
        (
            results["predicted_direction"]
            == results["actual_direction"]
        ).astype(float),
        np.nan,
    )

    results["strategy_return"] = (
        results["predicted_direction"] * results["actual"]
    )
    results["strategy_growth"] = (
        1 + results["strategy_return"]
    ).cumprod()
    results["target_growth"] = (
        1 + results["actual"]
    ).cumprod()

    results["rolling_adaptive_accuracy"] = (
        results["correct"]
        .rolling(lookback, min_periods=5)
        .mean()
    )

    results["rolling_base_accuracy"] = (
        results["base_correct"]
        .rolling(lookback, min_periods=5)
        .mean()
    )

    # Tomorrow's base forecast comes from the final row of peer returns.
    tomorrow_base_forecast = peer_signals.iloc[-1]

    # All currently observed outcomes can be used for tomorrow's decision.
    tomorrow_base_accuracy = (
        results["base_correct"]
        .tail(lookback)
        .mean()
    )

    enough_history = (
        results["base_correct"]
        .tail(lookback)
        .count()
        >= min_history
    )

    invert_tomorrow = (
        enough_history
        and tomorrow_base_accuracy < 0.5
    )

    tomorrow_forecast = (
        -tomorrow_base_forecast
        if invert_tomorrow
        else tomorrow_base_forecast
    )

    diagnostics = {
        "tomorrow_forecast": tomorrow_forecast,
        "tomorrow_base_forecast": tomorrow_base_forecast,
        "tomorrow_base_accuracy": tomorrow_base_accuracy,
        "invert_tomorrow": invert_tomorrow,
        "base_accuracy": results["base_correct"].mean(),
        "adaptive_accuracy": results["correct"].mean(),
        "adaptive_mae": (
            results["prediction"] - results["actual"]
        ).abs().mean(),
        "number_inverted": int(results["invert"].sum()),
        "results": results,
    }

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        results[
            ["prediction", "actual"]
        ].plot(ax=axes[0, 0], alpha=0.7)
        axes[0, 0].axhline(0, color="grey", linewidth=0.8)
        axes[0, 0].set_title("Adaptive prediction vs actual")

        results[
            ["rolling_base_accuracy", "rolling_adaptive_accuracy"]
        ].plot(ax=axes[0, 1])
        axes[0, 1].axhline(
            0.5,
            color="black",
            linestyle="--",
            label="Inversion threshold",
        )
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title(f"{lookback}-day directional accuracy")
        axes[0, 1].legend()

        results[
            ["strategy_growth", "target_growth"]
        ].plot(ax=axes[1, 0])
        axes[1, 0].set_title("Adaptive strategy vs target")
        axes[1, 0].set_ylabel("Growth of $1")

        results["invert"].astype(int).plot(
            ax=axes[1, 1],
            color="firebrick",
        )
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_yticklabels(["Usual", "Opposite"])
        axes[1, 1].set_title("Prediction regime")

        fig.suptitle(
            f"{target} tomorrow forecast: {tomorrow_forecast:.3%} — "
            f"{'INVERTED' if invert_tomorrow else 'USUAL'}"
        )
        fig.tight_layout()
        plt.show()

        diagnostics["figure"] = fig

    return tomorrow_forecast, diagnostics

Pair = tuple[Hashable, Hashable]


@dataclass(frozen=True)
class OLSWindowSelection:
    """Immutable output from :func:`select_ols_window`.

    Attributes
    ----------
    selected_window:
        Longest eligible window within one standard error of the best mean
        fold score.
    window_summary:
        One row per candidate window, including the cross-pair/fold score.
    fold_results:
        Diagnostics for every pair, window and chronological test fold.
    daily_results:
        Causal residuals, z-scores, positions and net strategy returns.
    """

    selected_window: int
    window_summary: pd.DataFrame
    fold_results: pd.DataFrame
    daily_results: pd.DataFrame


def _validate_inputs(
    prices: pd.DataFrame,
    pairs: Sequence[Pair],
    windows: Sequence[int],
    *,
    zscore_window: int,
    fold_size: int,
    annualization: int,
    entry_z: float,
    exit_z: float,
    transaction_cost_bps: float,
    minimum_trades: int,
) -> tuple[pd.DataFrame, tuple[Pair, ...], tuple[int, ...]]:
    """Validate and normalize public inputs."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.empty or prices.shape[1] < 2:
        raise ValueError("prices must contain observations for at least two instruments")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("prices index must be unique and ordered chronologically")
    if not prices.columns.is_unique:
        raise ValueError("prices columns must be unique")

    try:
        clean_prices = prices.astype(float).copy()
    except (TypeError, ValueError) as exc:
        raise TypeError("prices must contain numeric values") from exc
    values = clean_prices.to_numpy()
    if np.isinf(values).any():
        raise ValueError("prices must not contain infinite values")
    if (clean_prices <= 0).any().any():
        raise ValueError("log-price OLS requires strictly positive prices")

    normalized_pairs = tuple(tuple(pair) for pair in pairs)
    if not normalized_pairs:
        raise ValueError("pairs must not be empty")
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in normalized_pairs):
        raise ValueError("each pair must contain two different instruments")
    missing = {
        instrument
        for pair in normalized_pairs
        for instrument in pair
        if instrument not in clean_prices.columns
    }
    if missing:
        raise KeyError(f"unknown instruments: {sorted(missing, key=str)}")
    if len(set(normalized_pairs)) != len(normalized_pairs):
        raise ValueError("pairs must not contain duplicates")

    normalized_windows: list[int] = []
    for window in windows:
        if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
            raise TypeError("windows must contain integers")
        if int(window) < 3:
            raise ValueError("each OLS window must be at least 3")
        normalized_windows.append(int(window))
    if not normalized_windows:
        raise ValueError("windows must not be empty")
    normalized_windows = sorted(set(normalized_windows))

    integer_parameters = {
        "zscore_window": zscore_window,
        "fold_size": fold_size,
        "annualization": annualization,
        "minimum_trades": minimum_trades,
    }
    for name, value in integer_parameters.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if zscore_window < 2:
        raise ValueError("zscore_window must be at least 2")
    if not np.isfinite(entry_z) or entry_z <= 0:
        raise ValueError("entry_z must be positive and finite")
    if not np.isfinite(exit_z) or exit_z < 0 or exit_z >= entry_z:
        raise ValueError("exit_z must satisfy 0 <= exit_z < entry_z")
    if not np.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative and finite")

    required = max(normalized_windows) + zscore_window + fold_size + 1
    if len(clean_prices) < required:
        raise ValueError(
            f"prices has {len(clean_prices)} rows; at least {required} are required "
            "for the largest window, z-score history and one test fold"
        )
    return clean_prices, normalized_pairs, tuple(normalized_windows)


def _causal_rolling_ols(
    y: pd.Series,
    x: pd.Series,
    window: int,
) -> pd.DataFrame:
    """Return one-step-ahead rolling OLS parameters and residuals."""
    # Shift first: the estimate at t uses [t-window, ..., t-1].
    y_lagged = y.shift(1)
    x_lagged = x.shift(1)
    x_roll = x_lagged.rolling(window, min_periods=window)
    y_roll = y_lagged.rolling(window, min_periods=window)
    variance = x_roll.var(ddof=1)
    beta = y_roll.cov(x_lagged) / variance
    beta = beta.where(variance > np.finfo(float).eps)
    alpha = y_roll.mean() - beta * x_roll.mean()
    residual = y - alpha - beta * x
    return pd.DataFrame({"alpha": alpha, "beta": beta, "residual": residual})


def _causal_zscore(residual: pd.Series, window: int) -> pd.Series:
    """Standardize residuals using only prior residual observations."""
    history = residual.shift(1).rolling(window, min_periods=window)
    mean = history.mean()
    standard_deviation = history.std(ddof=1)
    return ((residual - mean) / standard_deviation).where(
        standard_deviation > np.finfo(float).eps
    )


def _positions_from_zscore(
    zscore: pd.Series,
    *,
    entry_z: float,
    exit_z: float,
) -> pd.Series:
    """Create a deterministic, close-to-close mean-reversion position."""
    positions = np.zeros(len(zscore), dtype=float)
    current = 0.0
    for location, value in enumerate(zscore.to_numpy(dtype=float)):
        if not np.isfinite(value):
            current = 0.0
        elif current == 0.0:
            if value >= entry_z:
                current = -1.0
            elif value <= -entry_z:
                current = 1.0
        elif abs(value) <= exit_z or np.sign(value) == current:
            current = 0.0
        positions[location] = current
    return pd.Series(positions, index=zscore.index, name="position")


def _daily_pair_results(
    log_prices: pd.DataFrame,
    pair: Pair,
    window: int,
    *,
    zscore_window: int,
    entry_z: float,
    exit_z: float,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    """Backtest one pair/window with lagged signals and normalized exposure."""
    dependent, independent = pair
    y = log_prices[dependent]
    x = log_prices[independent]
    model = _causal_rolling_ols(y, x, window)
    zscore = _causal_zscore(model["residual"], zscore_window)
    desired_position = _positions_from_zscore(
        zscore, entry_z=entry_z, exit_z=exit_z
    )

    # A signal observed at close t earns the close-to-close return t -> t+1.
    held_position = desired_position.shift(1).fillna(0.0)
    held_beta = model["beta"].shift(1)
    y_return = y.diff()
    x_return = x.diff()
    gross_return = (
        held_position
        * (y_return - held_beta * x_return)
        / (1.0 + held_beta.abs())
    )

    turnover = desired_position.diff().abs().fillna(desired_position.abs())
    costs = turnover.shift(1).fillna(0.0) * transaction_cost_bps / 10_000.0
    net_return = gross_return - costs
    trade_entry = (
        desired_position.ne(0.0)
        & desired_position.shift(1, fill_value=0.0).eq(0.0)
    )

    return model.assign(
        zscore=zscore,
        desired_position=desired_position,
        held_position=held_position,
        gross_return=gross_return,
        turnover=turnover,
        cost=costs,
        net_return=net_return,
        trade_entry=trade_entry.astype(int),
    )


def _fold_statistics(
    daily: pd.DataFrame,
    *,
    fold_size: int,
    annualization: int,
    minimum_trades: int,
) -> list[dict[str, float | int | bool]]:
    """Calculate non-overlapping chronological fold statistics."""
    valid = daily.dropna(subset=["net_return", "residual", "beta"])
    if valid.empty:
        return []

    records: list[dict[str, float | int | bool]] = []
    for fold_number, start in enumerate(range(0, len(valid), fold_size)):
        fold = valid.iloc[start : start + fold_size]
        if len(fold) < fold_size:
            break

        returns = fold["net_return"]
        volatility = float(returns.std(ddof=1))
        sharpe = (
            float(np.sqrt(annualization) * returns.mean() / volatility)
            if volatility > np.finfo(float).eps
            else np.nan
        )
        wealth = (1.0 + returns).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        trades = int(fold["trade_entry"].sum())
        beta_mean = float(fold["beta"].abs().mean())
        beta_instability = float(fold["beta"].diff().abs().mean())
        beta_instability /= max(beta_mean, np.finfo(float).eps)
        crossings = int(
            np.sign(fold["residual"])
            .diff()
            .abs()
            .fillna(0.0)
            .gt(0.0)
            .sum()
        )
        eligible = trades >= minimum_trades and np.isfinite(sharpe)
        records.append(
            {
                "fold": fold_number,
                "start": fold.index[0],
                "end": fold.index[-1],
                "observations": len(fold),
                "trades": trades,
                "crossings": crossings,
                "net_return": float(wealth.iloc[-1] - 1.0),
                "sharpe": sharpe,
                "maximum_drawdown": float(drawdown.min()),
                "turnover": float(fold["turnover"].sum()),
                "beta_instability": beta_instability,
                "eligible": eligible,
            }
        )
    return records


def select_ols_window(
    prices: pd.DataFrame,
    pairs: Sequence[Pair],
    *,
    windows: Sequence[int] = (20, 40, 60, 80, 120, 250),
    zscore_window: int = 20,
    fold_size: int = 63,
    annualization: int = 252,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_cost_bps: float = 5.0,
    minimum_trades: int = 2,
) -> OLSWindowSelection:
    """Select a global rolling-OLS window with chronological validation.

    Prices are converted to logs internally.  Each candidate is evaluated with
    the same causal mean-reversion strategy.  Fold score is net Sharpe; only
    folds meeting ``minimum_trades`` are eligible.  Window-level results pool
    eligible pair/fold observations.

    Selection uses the one-standard-error rule: among candidates whose mean
    score is no more than one standard error below the highest mean score,
    choose the longest window.  This favors hedge-ratio stability when observed
    performance is statistically indistinguishable.
    """
    clean_prices, normalized_pairs, normalized_windows = _validate_inputs(
        prices,
        pairs,
        windows,
        zscore_window=zscore_window,
        fold_size=fold_size,
        annualization=annualization,
        entry_z=entry_z,
        exit_z=exit_z,
        transaction_cost_bps=transaction_cost_bps,
        minimum_trades=minimum_trades,
    )
    log_prices = np.log(clean_prices)
    # Compare every candidate over exactly the same calendar observations.
    # Short windows must not receive extra, earlier folds unavailable to long
    # windows, as that would confound window length with market regime.
    common_evaluation_start = max(normalized_windows) + zscore_window

    daily_frames: list[pd.DataFrame] = []
    fold_records: list[dict[str, object]] = []
    for dependent, independent in normalized_pairs:
        for window in normalized_windows:
            daily = _daily_pair_results(
                log_prices,
                (dependent, independent),
                window,
                zscore_window=zscore_window,
                entry_z=entry_z,
                exit_z=exit_z,
                transaction_cost_bps=transaction_cost_bps,
            )
            daily.insert(0, "window", window)
            daily.insert(0, "independent", independent)
            daily.insert(0, "dependent", dependent)
            daily_frames.append(daily)

            for record in _fold_statistics(
                daily.iloc[common_evaluation_start:],
                fold_size=fold_size,
                annualization=annualization,
                minimum_trades=minimum_trades,
            ):
                fold_records.append(
                    {
                        "dependent": dependent,
                        "independent": independent,
                        "window": window,
                        **record,
                    }
                )

    if not fold_records:
        raise ValueError("no complete validation folds could be constructed")

    fold_results = pd.DataFrame.from_records(fold_records)
    eligible = fold_results.loc[fold_results["eligible"]].copy()
    if eligible.empty:
        raise ValueError(
            "no fold met minimum_trades; reduce minimum_trades, entry_z, or fold_size"
        )

    grouped = eligible.groupby("window", sort=True)
    summary = grouped.agg(
        eligible_folds=("sharpe", "size"),
        mean_fold_sharpe=("sharpe", "mean"),
        median_fold_sharpe=("sharpe", "median"),
        fold_sharpe_std=("sharpe", "std"),
        worst_fold_sharpe=("sharpe", "min"),
        mean_maximum_drawdown=("maximum_drawdown", "mean"),
        total_trades=("trades", "sum"),
        mean_turnover=("turnover", "mean"),
        mean_beta_instability=("beta_instability", "mean"),
    )
    summary = summary.reindex(normalized_windows)
    summary["standard_error"] = (
        summary["fold_sharpe_std"]
        .fillna(0.0)
        / np.sqrt(summary["eligible_folds"])
    )
    summary["eligible"] = summary["eligible_folds"].fillna(0).gt(0)

    available = summary.loc[summary["eligible"]]
    best_window = int(available["mean_fold_sharpe"].idxmax())
    cutoff = float(
        available.loc[best_window, "mean_fold_sharpe"]
        - available.loc[best_window, "standard_error"]
    )
    summary["within_one_standard_error"] = (
        summary["eligible"] & summary["mean_fold_sharpe"].ge(cutoff)
    )
    selected_window = int(
        summary.index[summary["within_one_standard_error"]].max()
    )
    summary["selected"] = summary.index == selected_window
    summary.index.name = "window"

    daily_results = pd.concat(daily_frames, axis=0)
    daily_results = daily_results.set_index(
        ["dependent", "independent", "window"], append=True
    ).reorder_levels(["dependent", "independent", "window", daily_results.index.name])
    daily_results.index.names = [
        "dependent",
        "independent",
        "window",
        "observation",
    ]

    return OLSWindowSelection(
        selected_window=selected_window,
        window_summary=summary,
        fold_results=fold_results,
        daily_results=daily_results.sort_index(),
    )

