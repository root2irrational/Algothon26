"""SIG Algothon strategy: pairs plus an EMA-correlation regime."""

import numpy as np
import pandas as pd


N_INSTRUMENTS = 51
ALGO_INDEX = 0
ALGO_LIMIT = 100_000.0
STANDARD_LIMIT = 10_000.0

EMA_WINDOW = 20
CORR_THRESHOLD = 0.75
MIN_CORR_GROUP_SIZE = 20

ENABLE_PAIRS = True
ENABLE_INDIVIDUAL = True

PAIRS_WINDOW = 250
PAIRS_Z_LOWER = 0.0
PAIRS_Z_UPPER = 10.0

INSTRUMENTS = (
    "ALGO", "AENO", "LSST", "SRNA", "ELLT", "AMRP", "OTCS", "HETT",
    "HUXZ", "DUCT", "SMAH", "NPCK", "MSDP", "EORC", "CUBO", "HRET",
    "ANSO", "DIHO", "RTTH", "SPLZ", "NWIG", "MMBT", "MDGI", "AGVF",
    "RRES", "CTGI", "ALUT", "ACAC", "SRTX", "GARI", "RCRI", "ACIX",
    "CCNS", "MTNS", "IHOZ", "NAYO", "FWWG", "EELT", "HRND", "AETS",
    "ULXY", "BLBT", "BENI", "ITPA", "HTRK", "NGTE", "ILVX", "FCSG",
    "FARS", "MHRM", "EAFC",
)

PAIRS = (
    ("MHRM", "EAFC"),
    ("ACIX", "ITPA"),
    ("EORC", "NGTE"),
    ("AENO", "NWIG"),
    ("SMAH", "ILVX"),
    ("HUXZ", "ACAC"),
    ("ALUT", "CCNS"),
    ("FWWG", "BLBT"),
    ("CTGI", "EELT"),
    ("HETT", "ULXY"),
    ("RTTH", "RRES"),
    ("RCRI", "NAYO"),
)

INDEX = {name: index for index, name in enumerate(INSTRUMENTS)}
PAIRED_INDICES = frozenset(INDEX[name] for pair in PAIRS for name in pair)
INDIVIDUAL_INDICES = tuple(
    index for index in range(N_INSTRUMENTS) if index not in PAIRED_INDICES
)

if len(INSTRUMENTS) != N_INSTRUMENTS:
    raise RuntimeError("Instrument configuration must contain 51 instruments.")

def _validate_prices(prices):
    """Return a validated floating-point price matrix."""
    matrix = np.asarray(prices, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != N_INSTRUMENTS:
        raise ValueError(
            f"prices must have shape ({N_INSTRUMENTS}, observations)"
        )
    return matrix


def _dollar_limit(index):
    return ALGO_LIMIT if index == ALGO_INDEX else STANDARD_LIMIT


def _quantity_for_limit(index, price):
    """Return the largest whole-share quantity inside the dollar limit."""
    if not np.isfinite(price) or price <= 0.0:
        return 0
    return int(np.floor(_dollar_limit(index) / price))


def Trade_pairs(prices):
    """Return rolling-OLS residual mean-reversion targets for the 12 pairs."""
    prices = _validate_prices(prices)
    target = np.zeros(N_INSTRUMENTS, dtype=np.int64)

    if prices.shape[1] < PAIRS_WINDOW + 1:
        return target

    epsilon = np.finfo(float).eps

    for y_name, x_name in PAIRS:
        y_index = INDEX[y_name]
        x_index = INDEX[x_name]
        y = prices[y_index, -(PAIRS_WINDOW + 1):]
        x = prices[x_index, -(PAIRS_WINDOW + 1):]

        if (
            not np.isfinite(y).all()
            or not np.isfinite(x).all()
            or np.any(y <= 0.0)
            or np.any(x <= 0.0)
        ):
            continue

        y_train, y_current = y[:-1], y[-1]
        x_train, x_current = x[:-1], x[-1]
        x_centered = x_train - x_train.mean()
        x_variance = x_centered @ x_centered
        if x_variance <= epsilon:
            continue

        beta = (x_centered @ (y_train - y_train.mean())) / x_variance
        intercept = y_train.mean() - beta * x_train.mean()
        residuals = y_train - intercept - beta * x_train
        residual_std = residuals.std(ddof=1)
        if not np.isfinite(residual_std) or residual_std <= epsilon:
            continue

        current_residual = y_current - intercept - beta * x_current
        z_score = (current_residual - residuals.mean()) / residual_std
        if (
            not np.isfinite(z_score)
            or not PAIRS_Z_LOWER <= abs(z_score) <= PAIRS_Z_UPPER
        ):
            continue

        direction = int(-np.sign(z_score))
        if direction == 0:
            continue

        scales = [_dollar_limit(y_index) / y_current]
        if abs(beta) > epsilon:
            scales.append(_dollar_limit(x_index) / (abs(beta) * x_current))
        scale = min(scales)
        if not np.isfinite(scale) or scale <= 0.0:
            continue

        target[y_index] += int(np.rint(direction * scale))
        target[x_index] += int(np.rint(-direction * beta * scale))

    return target


def _correlation_regime_directions(prices):
    """Return directions for the largest qualifying EMA-correlation group."""
    directions = np.zeros(N_INSTRUMENTS, dtype=np.int8)
    individual_indices = np.asarray(INDIVIDUAL_INDICES, dtype=int)
    history = prices[individual_indices]
    valid = np.isfinite(history).all(axis=1) & (history > 0.0).all(axis=1)
    valid_indices = individual_indices[valid]

    if valid_indices.size < 2:
        return directions

    smoothed_prices = (
        pd.DataFrame(prices[valid_indices].T)
        .ewm(span=EMA_WINDOW, adjust=False)
        .mean()
        .to_numpy()
        .T
    )
    recent_ema = smoothed_prices[:, -(EMA_WINDOW + 1):]
    ema_returns = np.diff(np.log(recent_ema), axis=1)
    varying = ema_returns.std(axis=1) > np.finfo(float).eps

    if np.count_nonzero(varying) < 2:
        return directions

    active_indices = valid_indices[varying]
    active_ema = recent_ema[varying]
    correlation = np.corrcoef(ema_returns[varying])
    np.fill_diagonal(correlation, -np.inf)

    related = correlation >= CORR_THRESHOLD
    anchor = int(np.argmax(related.sum(axis=1)))
    group_locations = np.flatnonzero(related[anchor])
    group_locations = np.unique(np.append(group_locations, anchor))

    if group_locations.size < MIN_CORR_GROUP_SIZE:
        return directions

    group_moves = np.log(
        active_ema[group_locations, -1]
        / active_ema[group_locations, 0]
    )
    common_direction = int(np.sign(np.median(group_moves)))
    if common_direction == 0:
        return directions

    aligned = np.sign(group_moves) == common_direction
    directions[active_indices[group_locations[aligned]]] = common_direction
    return directions


def Trade_individual(prices):
    """Trade an EMA-correlation regime; remain flat without one."""
    prices = _validate_prices(prices)
    target = np.zeros(N_INSTRUMENTS, dtype=np.int64)

    if prices.shape[1] < EMA_WINDOW + 1:
        return target

    directions = _correlation_regime_directions(prices)
    if not np.any(directions):
        return target

    for index in np.flatnonzero(directions):
        current_price = prices[index, -1]
        quantity = _quantity_for_limit(index, current_price)
        target[index] = int(directions[index]) * quantity

    return target


def _cap_positions(positions, prices):
    """Enforce the current-price dollar limit for every instrument."""
    capped = np.asarray(positions, dtype=np.int64).copy()
    if prices.shape[1] == 0:
        return np.zeros(N_INSTRUMENTS, dtype=np.int64)

    for index, price in enumerate(prices[:, -1]):
        maximum = _quantity_for_limit(index, price)
        capped[index] = np.clip(capped[index], -maximum, maximum)
    return capped


def getMyPosition(prcSoFar):
    """Return today's desired total positions for the Algothon evaluator."""
    prices = _validate_prices(prcSoFar)
    positions = np.zeros(N_INSTRUMENTS, dtype=np.int64)

    if ENABLE_PAIRS:
        positions += Trade_pairs(prices)

    if ENABLE_INDIVIDUAL:
        positions += Trade_individual(prices)

    return _cap_positions(positions, prices)
