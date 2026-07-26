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

import statsmodels.api as sm

from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.decomposition import PCA

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
    instruments: Sequence[object] | None = None,
    ema_windows: Sequence[int] | None = (10, 30),
    volatility_window: int | None = 20,
    sharpe_window: int | None = 60,
    annualization: int = 250,
    normalize: bool = False,
    n_cols: int = 4,
    title: str | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot selected instruments and their indicators in a compact grid.

    Parameters
    ----------
    prices:
        Price observations with rows ordered through time and one instrument
        per column.
    instruments:
        Instruments to plot, specified by column name, zero-based column
        position, or a mixture of both. The supplied order determines the plot
        order. If ``None``, all instruments are plotted.
    ema_windows:
        EMA windows to display. Pass ``None`` to exclude EMAs.
    volatility_window:
        Rolling volatility window. Pass ``None`` to exclude volatility.
    sharpe_window:
        Rolling Sharpe window. Pass ``None`` to exclude Sharpe.
    annualization:
        Number of observations per year.
    normalize:
        Normalize each selected price series to 100 at its first valid value.
    n_cols:
        Maximum number of instruments per figure row.
    title:
        Optional figure title.

    Returns
    -------
    tuple[Figure, numpy.ndarray]
        The figure and an axes array shaped
        ``(instrument_rows, columns, panels_per_instrument)``.

    Raises
    ------
    TypeError
        If ``instruments`` is a string instead of a sequence or contains an
        invalid selector type.
    ValueError
        If no instruments are selected, selectors are duplicated, or a
        configuration value is invalid.
    KeyError
        If a requested instrument name does not exist.
    IndexError
        If a requested instrument position is outside the available columns.
    """
    frame = _as_frame(prices, name="price")

    if instruments is not None:
        if isinstance(instruments, (str, bytes)):
            raise TypeError(
                "instruments must be a sequence of names or indices, "
                "not a single string"
            )

        selectors = list(instruments)

        if not selectors:
            raise ValueError("instruments must not be empty")

        selected_columns: list[object] = []

        for selector in selectors:
            if isinstance(selector, (int, np.integer)) and not isinstance(
                selector,
                (bool, np.bool_),
            ):
                position = int(selector)

                if position < 0 or position >= frame.shape[1]:
                    raise IndexError(
                        f"instrument index {position} is outside "
                        f"[0, {frame.shape[1] - 1}]"
                    )

                selected_columns.append(frame.columns[position])
                continue

            try:
                exists = selector in frame.columns
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "each instrument selector must be a column name "
                    "or zero-based integer position"
                ) from exc

            if not exists:
                raise KeyError(
                    f"unknown instrument name: {selector!r}"
                )

            selected_columns.append(selector)

        if len(set(selected_columns)) != len(selected_columns):
            raise ValueError(
                "instruments must not resolve to duplicate columns"
            )

        frame = frame.loc[:, selected_columns].copy()

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

    if len(set(windows)) != len(windows):
        raise ValueError("ema_windows must not contain duplicates")

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

        if first.isna().any():
            missing = first.index[first.isna()].tolist()
            raise ValueError(
                "cannot normalize instruments without a valid observation: "
                f"{missing}"
            )

        if (first == 0).any():
            zero_columns = first.index[first == 0].tolist()
            raise ValueError(
                "cannot normalize instruments whose first valid value is "
                f"zero: {zero_columns}"
            )

        display = display.divide(first).multiply(100)

    # Calculate returns only when a return-based panel is enabled.
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

        # Price and optional EMAs.
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

        # Optional rolling volatility.
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

        # Optional rolling Sharpe.
        if sharpe is not None:
            sharpe_axis = instrument_axes[panel_index]

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
    "OLSSyntheticInstrumentResult",
    "fit_algo_synthetic_ols",
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

PriceMode = Literal["raw", "ema"]


@dataclass(frozen=True)
class OLSSyntheticInstrumentResult:
    """Output from fitting an OLS synthetic instrument for one target.

    Attributes
    ----------
    model:
        Fitted statsmodels OLS results object.
    diagnostics:
        Window-level observations containing the target, prediction, residual,
        residual z-score, and target's next-day return.
    coefficients:
        Fitted intercept and predictor coefficients.
    figure:
        Four-panel diagnostic figure.
    axes:
        Two-by-two array of diagnostic axes.
    target:
        Name of the modelled instrument.
    predictors:
        Predictor instrument names in model order.
    price_mode:
        Whether the regression used raw or EMA-smoothed prices.
    window:
        Number of most recent observations used to fit the model.
    ema_span:
        EMA span when ``price_mode="ema"``; otherwise ``None``.
    """

    model: object
    diagnostics: pd.DataFrame
    coefficients: pd.Series
    figure: Figure
    axes: np.ndarray
    target: object
    predictors: tuple[object, ...]
    price_mode: PriceMode
    window: int
    ema_span: int | None


def fit_algo_synthetic_ols(
    prices: pd.DataFrame,
    instruments: Sequence[object],
    *,
    target: object = "ALGO",
    window: int = 250,
    price_mode: PriceMode = "raw",
    ema_span: int = 20,
    include_intercept: bool = True,
    distribution_bins: int = 30,
    confidence_level: float = 0.95,
    figure_size: tuple[float, float] = (14.0, 10.0),
) -> OLSSyntheticInstrumentResult:
    """Fit a rolling-window OLS synthetic instrument and plot diagnostics.

    The target instrument is regressed on the supplied predictor instruments:

        target_price[t] = intercept + X[t] @ beta + residual[t]

    Only the latest ``window`` observations are used. ``price_mode="raw"``
    regresses raw price levels. ``price_mode="ema"`` first applies an
    exponentially weighted moving average to every selected price series.

    The next-day target return in the diagnostic table is always calculated
    from the original, unsmoothed target prices:

        next_day_return[t] = price[t + 1] / price[t] - 1

    Consequently, the final residual has no known next-day return and is
    excluded only from the residual-versus-forward-return scatter plot.

    Parameters
    ----------
    prices:
        Price DataFrame with days as rows and instruments as columns. Rows must
        be chronologically ordered.
    instruments:
        Predictor instruments. Do not include ``target``.
    target:
        Target instrument, normally ``"ALGO"``.
    window:
        Number of most recent complete observations used for OLS.
    price_mode:
        ``"raw"`` or ``"ema"``.
    ema_span:
        EMA span used when ``price_mode="ema"``.
    include_intercept:
        Include an OLS intercept when true.
    distribution_bins:
        Histogram bin count for the residual distribution.
    confidence_level:
        Confidence level used for the residual reference band in the
        residual-time plot.
    figure_size:
        Matplotlib figure size.

    Returns
    -------
    OLSSyntheticInstrumentResult
        Structured result containing the model and all diagnostics.

    Raises
    ------
    TypeError
        If ``prices`` is not a DataFrame or contains non-numeric data.
    ValueError
        If configuration or observations are invalid.
    KeyError
        If a requested instrument is missing.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    if prices.empty:
        raise ValueError("prices must not be empty")

    if not prices.columns.is_unique:
        raise ValueError("price columns must be unique")

    if not prices.index.is_monotonic_increasing:
        raise ValueError(
            "prices must be ordered chronologically by an increasing index"
        )

    if isinstance(instruments, (str, bytes)):
        raise TypeError(
            "instruments must be a sequence of instrument names, not a string"
        )

    predictors = tuple(instruments)

    if not predictors:
        raise ValueError("instruments must contain at least one predictor")

    if len(set(predictors)) != len(predictors):
        raise ValueError("instruments must not contain duplicates")

    if target in predictors:
        raise ValueError(
            f"target {target!r} must not also appear in instruments"
        )

    requested = (target, *predictors)
    missing = [
        instrument
        for instrument in requested
        if instrument not in prices.columns
    ]

    if missing:
        raise KeyError(f"unknown instrument(s): {missing}")

    if (
        isinstance(window, bool)
        or not isinstance(window, (int, np.integer))
        or window <= 0
    ):
        raise ValueError("window must be a positive integer")

    window = int(window)

    if price_mode not in {"raw", "ema"}:
        raise ValueError("price_mode must be 'raw' or 'ema'")

    if (
        isinstance(ema_span, bool)
        or not isinstance(ema_span, (int, np.integer))
        or ema_span <= 0
    ):
        raise ValueError("ema_span must be a positive integer")

    ema_span = int(ema_span)

    if (
        isinstance(distribution_bins, bool)
        or not isinstance(distribution_bins, (int, np.integer))
        or distribution_bins < 5
    ):
        raise ValueError("distribution_bins must be an integer of at least 5")

    distribution_bins = int(distribution_bins)

    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    if len(figure_size) != 2 or any(
        not np.isfinite(value) or value <= 0
        for value in figure_size
    ):
        raise ValueError(
            "figure_size must contain two positive finite values"
        )

    try:
        selected_prices = prices.loc[:, list(requested)].astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "selected price columns must contain only numeric values"
        ) from exc

    selected_values = selected_prices.to_numpy(dtype=float)

    if not np.isfinite(selected_values).all():
        raise ValueError(
            "selected price observations must all be finite"
        )

    if (selected_values <= 0).any():
        raise ValueError(
            "selected prices must be strictly positive"
        )

    number_of_parameters = len(predictors) + int(include_intercept)
    minimum_observations = number_of_parameters + 2

    if window < minimum_observations:
        raise ValueError(
            f"window={window} is too small for {number_of_parameters} "
            f"model parameters; use at least {minimum_observations} observations"
        )

    if len(selected_prices) < window:
        raise ValueError(
            f"window={window} requires at least {window} price rows; "
            f"received {len(selected_prices)}"
        )

    if price_mode == "ema":
        modelling_prices = selected_prices.ewm(
            span=ema_span,
            adjust=False,
            min_periods=1,
        ).mean()
    else:
        modelling_prices = selected_prices.copy()

    modelling_window = modelling_prices.iloc[-window:].copy()

    y = modelling_window[target].rename(str(target))
    X = modelling_window.loc[:, list(predictors)].copy()

    if include_intercept:
        X_design = sm.add_constant(
            X,
            has_constant="add",
        )
    else:
        X_design = X

    matrix = X_design.to_numpy(dtype=float)
    matrix_rank = int(np.linalg.matrix_rank(matrix))

    if matrix_rank < matrix.shape[1]:
        raise ValueError(
            "OLS design matrix is rank deficient. The predictor instruments "
            "are perfectly or nearly perfectly collinear; remove predictors "
            "or use ridge regression/PCA."
        )

    model = sm.OLS(
        endog=y,
        exog=X_design,
        missing="raise",
    ).fit()

    predicted = pd.Series(
        model.fittedvalues,
        index=modelling_window.index,
        name="predicted",
        dtype=float,
    )
    residual = pd.Series(
        model.resid,
        index=modelling_window.index,
        name="residual",
        dtype=float,
    )

    residual_mean = float(residual.mean())
    residual_std = float(residual.std(ddof=0))

    if not np.isfinite(residual_mean):
        raise ValueError("fitted residual mean is not finite")

    if (
        not np.isfinite(residual_std)
        or residual_std <= np.finfo(float).eps
    ):
        raise ValueError(
            "fitted residuals have zero or numerically negligible variation"
        )

    residual_zscore = (
        (residual - residual_mean) / residual_std
    ).rename("residual_zscore")

    # Calculate forward returns from original, unsmoothed ALGO prices.
    all_target_returns = (
        selected_prices[target]
        .pct_change(fill_method=None)
        .shift(-1)
        .rename("next_day_target_return")
    )

    diagnostics = pd.DataFrame(
        {
            "target": y,
            "predicted": predicted,
            "residual": residual,
            "residual_zscore": residual_zscore,
            "next_day_target_return": all_target_returns.reindex(
                modelling_window.index
            ),
        },
        index=modelling_window.index,
    )
    diagnostics.index.name = prices.index.name or "observation"

    if not np.isfinite(
        diagnostics[
            ["target", "predicted", "residual", "residual_zscore"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "model produced non-finite diagnostic values"
        )

    coefficients = pd.Series(
        model.params,
        index=X_design.columns,
        name="coefficient",
        dtype=float,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=figure_size,
        squeeze=False,
    )

    confidence_multiplier = float(
        norm.ppf((1.0 + float(confidence_level)) / 2.0)
    )

    # Panel 1: residual through time.
    time_axis: Axes = axes[0, 0]
    time_axis.plot(
        residual.index,
        residual,
        color="tab:blue",
        linewidth=1.3,
        label="OLS residual",
    )
    time_axis.axhline(
        residual_mean,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"Mean = {residual_mean:.4g}",
    )
    time_axis.axhline(
        residual_mean + confidence_multiplier * residual_std,
        color="tab:red",
        linestyle=":",
        linewidth=1.0,
        label=(
            f"{confidence_level:.0%} normal reference band"
        ),
    )
    time_axis.axhline(
        residual_mean - confidence_multiplier * residual_std,
        color="tab:red",
        linestyle=":",
        linewidth=1.0,
    )
    time_axis.scatter(
        [residual.index[-1]],
        [residual.iloc[-1]],
        color="tab:orange",
        s=45,
        zorder=3,
        label=f"Latest z = {residual_zscore.iloc[-1]:.2f}",
    )
    time_axis.set_title("OLS residual through time")
    time_axis.set_xlabel("Observation")
    time_axis.set_ylabel("Residual")
    time_axis.grid(alpha=0.25)
    time_axis.legend(fontsize="small")

    # Panel 2: residual versus predicted target price.
    fitted_axis: Axes = axes[0, 1]
    fitted_axis.scatter(
        predicted,
        residual,
        alpha=0.7,
        edgecolors="none",
    )
    fitted_axis.axhline(
        residual_mean,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )

    fitted_correlation = float(
        predicted.corr(residual)
    )

    fitted_axis.set_title(
        "Residual vs predicted target price\n"
        f"Correlation = {fitted_correlation:.3f}"
    )
    fitted_axis.set_xlabel(f"Predicted {target} price")
    fitted_axis.set_ylabel("Residual")
    fitted_axis.grid(alpha=0.25)

    # Panel 3: residual distribution and fitted normal density.
    distribution_axis: Axes = axes[1, 0]
    distribution_axis.hist(
        residual,
        bins=distribution_bins,
        density=True,
        alpha=0.65,
        color="tab:blue",
        edgecolor="white",
        label="Residuals",
    )

    x_min = float(residual.min())
    x_max = float(residual.max())

    if np.isclose(x_min, x_max):
        x_min = residual_mean - 4.0 * residual_std
        x_max = residual_mean + 4.0 * residual_std

    density_x = np.linspace(
        x_min,
        x_max,
        400,
    )
    density_y = norm.pdf(
        density_x,
        loc=residual_mean,
        scale=residual_std,
    )

    distribution_axis.plot(
        density_x,
        density_y,
        color="tab:red",
        linewidth=2.0,
        label="Fitted normal density",
    )
    distribution_axis.axvline(
        residual.iloc[-1],
        color="tab:orange",
        linestyle="--",
        linewidth=1.2,
        label="Latest residual",
    )
    distribution_axis.set_title(
        "Residual distribution and normal fit"
    )
    distribution_axis.set_xlabel("Residual")
    distribution_axis.set_ylabel("Density")
    distribution_axis.grid(alpha=0.25)
    distribution_axis.legend(fontsize="small")

    # Panel 4: residual versus the following day's ALGO return.
    forward_axis: Axes = axes[1, 1]
    forward_data = diagnostics.dropna(
        subset=["residual", "next_day_target_return"]
    )

    if len(forward_data) >= 2:
        forward_axis.scatter(
            forward_data["residual"],
            forward_data["next_day_target_return"],
            alpha=0.7,
            edgecolors="none",
        )

        regression_x = forward_data["residual"].to_numpy(
            dtype=float
        )
        regression_y = forward_data[
            "next_day_target_return"
        ].to_numpy(dtype=float)

        slope, intercept = np.polyfit(
            regression_x,
            regression_y,
            deg=1,
        )

        line_x = np.linspace(
            regression_x.min(),
            regression_x.max(),
            200,
        )
        forward_axis.plot(
            line_x,
            intercept + slope * line_x,
            color="tab:red",
            linewidth=1.5,
            label=f"Linear slope = {slope:.4g}",
        )

        forward_correlation = float(
            forward_data["residual"].corr(
                forward_data["next_day_target_return"]
            )
        )
        forward_axis.set_title(
            f"Residual vs next-day {target} return\n"
            f"Correlation = {forward_correlation:.3f}"
        )
        forward_axis.legend(fontsize="small")
    else:
        forward_axis.text(
            0.5,
            0.5,
            "Insufficient forward-return observations",
            ha="center",
            va="center",
            transform=forward_axis.transAxes,
        )
        forward_axis.set_title(
            f"Residual vs next-day {target} return"
        )

    forward_axis.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    forward_axis.axvline(
        residual_mean,
        color="black",
        linestyle=":",
        linewidth=0.8,
    )
    forward_axis.set_xlabel("Current residual")
    forward_axis.set_ylabel(f"Next-day {target} return")
    forward_axis.grid(alpha=0.25)

    mode_description = (
        f"EMA(span={ema_span})"
        if price_mode == "ema"
        else "raw prices"
    )

    figure.suptitle(
        f"{target} synthetic-instrument OLS diagnostics\n"
        f"{len(predictors)} predictors · {window}-day window · "
        f"{mode_description} · adjusted R² = {model.rsquared_adj:.4f}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    return OLSSyntheticInstrumentResult(
        model=model,
        diagnostics=diagnostics,
        coefficients=coefficients,
        figure=figure,
        axes=axes,
        target=target,
        predictors=predictors,
        price_mode=price_mode,
        window=window,
        ema_span=ema_span if price_mode == "ema" else None,
    )

def _prepare_multi_ols_data(
    prices: DataLike,
    instruments: Sequence[object],
) -> pd.DataFrame:
    """Validate and select data used by multi-instrument OLS functions."""
    frame = _as_frame(prices, name="price")

    if isinstance(instruments, (str, bytes)):
        raise TypeError("instruments must be a sequence, not a string")

    instruments = tuple(instruments)

    if len(instruments) < 2:
        raise ValueError("at least two instruments are required")

    if len(set(instruments)) != len(instruments):
        raise ValueError("instruments must not contain duplicates")

    selected = select_instruments(frame, instruments)

    values = selected.to_numpy(dtype=float)

    if np.isinf(values).any():
        raise ValueError("prices must not contain infinite values")

    return selected

def plot_multi_ols_fits(
    prices: DataLike,
    instruments: Sequence[object],
    *,
    window: int = 250,
    columns: int = 3,
    include_intercept: bool = True,
) -> tuple[
    dict[object, object],
    pd.DataFrame,
    Figure,
    np.ndarray,
]:
    """
    Fit each instrument against all other instruments and plot actual
    versus fitted target prices.

    Returns
    -------
    models:
        Mapping from target name to fitted statsmodels OLS result.
    fitted_data:
        DataFrame with MultiIndex columns:
            (target, "actual")
            (target, "fitted")
            (target, "residual")
    figure, axes:
        Matplotlib plotting objects.
    """
    frame = _prepare_multi_ols_data(
        prices,
        instruments,
    )

    window = _positive_int(window, name="window")
    columns = _positive_int(columns, name="columns")

    # Use the most recent complete observations.
    sample = frame.dropna(how="any").iloc[-window:]

    if len(sample) < window:
        raise ValueError(
            f"window={window} requires at least {window} "
            "complete observations"
        )

    number_of_predictors = sample.shape[1] - 1
    number_of_parameters = (
        number_of_predictors + int(include_intercept)
    )

    if len(sample) <= number_of_parameters:
        raise ValueError(
            "window is too small for the number of OLS parameters"
        )

    columns = min(columns, sample.shape[1])
    rows = int(np.ceil(sample.shape[1] / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.5 * rows),
        squeeze=False,
        sharex=True,
    )

    models: dict[object, object] = {}
    results: dict[object, pd.DataFrame] = {}

    for axis, target_name in zip(
        axes.flat,
        sample.columns,
    ):
        predictor_names = [
            name
            for name in sample.columns
            if name != target_name
        ]

        y = sample[target_name]
        X = sample.loc[:, predictor_names]

        if include_intercept:
            X_design = sm.add_constant(
                X,
                has_constant="add",
            )
        else:
            X_design = X

        matrix = X_design.to_numpy(dtype=float)

        if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
            axis.text(
                0.5,
                0.5,
                "Rank-deficient OLS",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_title(str(target_name))
            axis.grid(alpha=0.25)
            continue

        model = sm.OLS(
            y,
            X_design,
            missing="raise",
        ).fit()

        fitted = pd.Series(
            model.fittedvalues,
            index=sample.index,
            name="fitted",
        )
        residual = (
            y - fitted
        ).rename("residual")

        models[target_name] = model
        results[target_name] = pd.DataFrame(
            {
                "actual": y,
                "fitted": fitted,
                "residual": residual,
            }
        )

        axis.plot(
            y.index,
            y,
            linewidth=1.3,
            label="Actual",
        )
        axis.plot(
            fitted.index,
            fitted,
            linewidth=1.2,
            linestyle="--",
            label="OLS fitted",
        )

        axis.set_title(
            f"{target_name}\n"
            f"Adjusted R² = {model.rsquared_adj:.3f}"
        )
        axis.set_xlabel("Observation")
        axis.set_ylabel("Price")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")

    for axis in axes.flat[sample.shape[1]:]:
        axis.set_visible(False)

    if not results:
        raise ValueError(
            "all OLS models were rank deficient"
        )

    fitted_data = pd.concat(
        results,
        axis=1,
    )
    fitted_data.columns.names = [
        "target",
        "series",
    ]

    figure.suptitle(
        "Multi-Instrument OLS: Actual vs Fitted",
        y=1.01,
    )
    figure.tight_layout()

    return models, fitted_data, figure, axes


def plot_rolling_multi_ols_spreads(
    prices: DataLike,
    instruments: Sequence[object],
    *,
    rolling_window: int = 250,
    columns: int = 3,
    include_intercept: bool = True,
) -> tuple[pd.DataFrame, Figure, np.ndarray]:
    """
    Plot one-step-ahead rolling OLS residuals for every target instrument.

    For each target and observation t:

        model_t = OLS fitted on [t - rolling_window, t)
        residual_t = actual_target[t] - prediction_t

    Thus, the current observation is never included in its own OLS fit.
    """
    frame = _prepare_multi_ols_data(
        prices,
        instruments,
    )

    rolling_window = _positive_int(
        rolling_window,
        name="rolling_window",
    )
    columns = _positive_int(
        columns,
        name="columns",
    )

    number_of_predictors = frame.shape[1] - 1
    number_of_parameters = (
        number_of_predictors + int(include_intercept)
    )

    if rolling_window <= number_of_parameters:
        raise ValueError(
            f"rolling_window={rolling_window} is too small for "
            f"{number_of_parameters} OLS parameters"
        )

    if len(frame) <= rolling_window:
        raise ValueError(
            "prices must contain more observations than rolling_window"
        )

    residuals = pd.DataFrame(
        np.nan,
        index=frame.index,
        columns=frame.columns,
        dtype=float,
    )

    for target_name in frame.columns:
        predictor_names = [
            name
            for name in frame.columns
            if name != target_name
        ]

        y_all = frame[target_name].to_numpy(dtype=float)
        X_all = frame.loc[
            :,
            predictor_names,
        ].to_numpy(dtype=float)

        for current_location in range(
            rolling_window,
            len(frame),
        ):
            start = current_location - rolling_window

            y_train = y_all[
                start:current_location
            ]
            X_train = X_all[
                start:current_location
            ]
            y_current = y_all[current_location]
            X_current = X_all[current_location]

            if (
                not np.isfinite(y_train).all()
                or not np.isfinite(X_train).all()
                or not np.isfinite(y_current)
                or not np.isfinite(X_current).all()
            ):
                continue

            if include_intercept:
                design_train = np.column_stack(
                    [
                        np.ones(rolling_window),
                        X_train,
                    ]
                )
                design_current = np.concatenate(
                    ([1.0], X_current)
                )
            else:
                design_train = X_train
                design_current = X_current

            if (
                np.linalg.matrix_rank(design_train)
                < design_train.shape[1]
            ):
                continue

            coefficients, _, _, _ = np.linalg.lstsq(
                design_train,
                y_train,
                rcond=None,
            )

            prediction = (
                design_current @ coefficients
            )
            residuals.loc[
                frame.index[current_location],
                target_name,
            ] = y_current - prediction

    usable_columns = [
        name
        for name in residuals.columns
        if residuals[name].notna().sum() >= 2
        and residuals[name].std(ddof=0) > np.finfo(float).eps
    ]

    if not usable_columns:
        raise ValueError(
            "no valid rolling residual series could be estimated"
        )

    residuals = residuals.loc[:, usable_columns]

    columns = min(columns, residuals.shape[1])
    rows = int(np.ceil(residuals.shape[1] / columns))

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 3.5 * rows),
        squeeze=False,
        sharex=True,
    )

    for axis, target_name in zip(
        axes.flat,
        residuals.columns,
    ):
        spread = residuals[target_name]
        spread_mean = spread.mean()
        spread_std = spread.std(ddof=0)

        axis.plot(
            spread.index,
            spread,
            linewidth=1.0,
            label="Rolling residual",
        )
        axis.axhline(
            spread_mean,
            color="black",
            linestyle="--",
            linewidth=0.9,
            label="Mean",
        )
        axis.axhline(
            spread_mean + 2.0 * spread_std,
            color="tab:red",
            linestyle=":",
            linewidth=0.9,
            label="±2σ",
        )
        axis.axhline(
            spread_mean - 2.0 * spread_std,
            color="tab:red",
            linestyle=":",
            linewidth=0.9,
        )

        latest = spread.dropna()
        latest_zscore = (
            (latest.iloc[-1] - spread_mean) / spread_std
        )

        axis.set_title(
            f"{target_name} rolling spread\n"
            f"Latest z = {latest_zscore:.2f}"
        )
        axis.set_xlabel("Observation")
        axis.set_ylabel("OLS residual")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")

    for axis in axes.flat[residuals.shape[1]:]:
        axis.set_visible(False)

    figure.suptitle(
        "Rolling Multi-Instrument OLS Residual Spreads",
        y=1.01,
    )
    figure.tight_layout()

    return residuals, figure, axes

