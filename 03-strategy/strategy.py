import numpy as np
import pandas as pd

# =============================================================================
# Instruments and absolute position limits
# =============================================================================

N_INST = 51

MAX_POS_ALGO = 100_000
MAX_POS_ELSE = 10_000

INSTRUMENTS = [
    "ALGO", "AENO", "LSST", "SRNA", "ELLT", "AMRP", "OTCS", "HETT",
    "HUXZ", "DUCT", "SMAH", "NPCK", "MSDP", "EORC", "CUBO", "HRET",
    "ANSO", "DIHO", "RTTH", "SPLZ", "NWIG", "MMBT", "MDGI", "AGVF",
    "RRES", "CTGI", "ALUT", "ACAC", "SRTX", "GARI", "RCRI", "ACIX",
    "CCNS", "MTNS", "IHOZ", "NAYO", "FWWG", "EELT", "HRND", "AETS",
    "ULXY", "BLBT", "BENI", "ITPA", "HTRK", "NGTE", "ILVX", "FCSG",
    "FARS", "MHRM", "EAFC",
]

if len(INSTRUMENTS) != N_INST:
    raise ValueError(
        f"N_INST={N_INST}, but {len(INSTRUMENTS)} instruments were provided"
    )


# =============================================================================
# Strategy parameters
# =============================================================================

ENABLE_PAIRS = False
ENABLE_INDIVIDUAL = True

# Strategy-level dollar allocations.
PAIR_BET = 10_000

MULTI_PAIR_BET_ALGO = 1_0
MULTI_PAIR_BET = 10_000

PAIRS_WINDOW = 250
PAIRS_Z_LOWER = 0.0
PAIRS_Z_UPPER = 10.0

INDIVIDUAL_WINDOW = 250
INDIVIDUAL_Z_LOWER = 0.0
INDIVIDUAL_Z_UPPER = 10.0

EMA_WINDOW = 100
EMA_MOMENTUM_Z_THRESHOLD = 0.000001
LOOKBACK = 20

# =============================================================================
# Instrument groups
# =============================================================================

ALL_PAIRS = [
    ("MHRM", "EAFC"),
    ("ACIX", "ITPA"),
    ("EORC", "NGTE"),
    ("AENO", "NWIG"),
    ("SMAH", "ILVX"),
    ("HUXZ", "ACAC"),
    ("ALUT", "CCNS"),
    ("ALGO", "BENI"),
    ("NPCK", "SRTX"),
    ("FWWG", "BLBT"),
    ("SRNA", "IHOZ"),
    ("CTGI", "EELT"),
    ("HETT", "ULXY"),
    ("RTTH", "RRES"),
    ("MTNS", "HTRK"),
    ("DUCT", "GARI"),
    ("MSDP", "SPLZ"),
    ("ELLT", "DIHO"),
    ("RCRI", "NAYO"),
    ("HRND", "AETS"),
    ("MDGI", "AGVF"),
    ("AMRP", "FCSG"),
    ("LSST", "HRET"),
    ("CUBO", "ANSO"),
    ("OTCS", "MMBT"),
]

PAIRS_1 = [
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
]

PAIRS_2 = [
    ("MHRM", "EAFC"),
    ("ACIX", "ITPA"),
    ("AENO", "NWIG"),
    ("SMAH", "ILVX"),
    ("HUXZ", "ACAC"),
    ("FWWG", "BLBT"),
    ("CTGI", "EELT"),
    ("HETT", "ULXY"),
    ("RTTH", "RRES"),
    ("RCRI", "NAYO"),
]

PAIRS = PAIRS_1

INDEX = {
    name: index
    for index, name in enumerate(INSTRUMENTS)
}

PAIRED = {
    name
    for pair in PAIRS
    for name in pair
}

INDIVIDUAL = [name for name in INSTRUMENTS if name not in PAIRED]


# =============================================================================
# Position-sizing helpers
# =============================================================================

def absolute_dollar_limit(instrument_name):
    """Return the overall absolute dollar-position limit."""
    return (
        MAX_POS_ALGO
        if instrument_name == "ALGO"
        else MAX_POS_ELSE
    )


def pair_dollar_limit(instrument_name):
    """Return the allowed dollar exposure for the pairs strategy."""
    return min(
        PAIR_BET,
        absolute_dollar_limit(instrument_name),
    )


def individual_dollar_limit(instrument_name):
    """Return the allowed dollar exposure for the multi-OLS strategy."""
    strategy_bet = (
        MULTI_PAIR_BET_ALGO
        if instrument_name == "ALGO"
        else MULTI_PAIR_BET
    )

    return min(
        strategy_bet,
        absolute_dollar_limit(instrument_name),
    )


def validate_prices(prices):
    """Validate and return the supplied price matrix."""
    prices = np.asarray(prices, dtype=float)

    if prices.ndim != 2 or prices.shape[0] != N_INST:
        raise ValueError(
            f"prices must have shape ({N_INST}, observations)"
        )

    return prices


def cap_combined_positions(positions, prices):
    """
    Enforce absolute dollar-position limits after combining strategies.

    This protects against future configurations in which instruments appear
    in multiple strategies or pairs.
    """
    positions = np.asarray(positions, dtype=int).copy()

    if prices.shape[1] == 0:
        return np.zeros(N_INST, dtype=int)

    for index, name in enumerate(INSTRUMENTS):
        price = prices[index, -1]

        if not np.isfinite(price) or price <= 0.0:
            positions[index] = 0
            continue

        maximum_quantity = int(
            np.floor(
                absolute_dollar_limit(name) / price
            )
        )

        positions[index] = int(
            np.clip(
                positions[index],
                -maximum_quantity,
                maximum_quantity,
            )
        )

    return positions


# =============================================================================
# Pair strategy
# =============================================================================

def Trade_pairs(prices):
    """
    Trade each configured pair using a rolling OLS residual z-score.

    The previous PAIRS_WINDOW observations are used to fit:

        y = alpha + beta*x + residual

    Today's observation is excluded from the fit and used only to generate
    the current residual signal.
    """
    prices = validate_prices(prices)
    target = np.zeros(N_INST, dtype=int)

    if (
        PAIRS_WINDOW < 2
        or PAIRS_Z_LOWER < 0.0
        or PAIRS_Z_UPPER < PAIRS_Z_LOWER
        or prices.shape[1] < PAIRS_WINDOW + 1
    ):
        return target

    for y_name, x_name in PAIRS:
        yi = INDEX[y_name]
        xi = INDEX[x_name]

        y = prices[
            yi,
            -(PAIRS_WINDOW + 1):
        ]
        x = prices[
            xi,
            -(PAIRS_WINDOW + 1):
        ]

        if (
            not np.isfinite(y).all()
            or not np.isfinite(x).all()
            or np.any(y <= 0.0)
            or np.any(x <= 0.0)
        ):
            continue

        y_train = y[:-1]
        x_train = x[:-1]

        y_current = y[-1]
        x_current = x[-1]

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        x_centered = x_train - x_mean
        y_centered = y_train - y_mean

        x_variance = x_centered @ x_centered

        if x_variance <= np.finfo(float).eps:
            continue

        beta = (
            x_centered @ y_centered
        ) / x_variance
        intercept = y_mean - beta * x_mean

        training_residuals = (
            y_train
            - intercept
            - beta * x_train
        )

        residual_mean = np.mean(training_residuals)
        residual_std = np.std(
            training_residuals,
            ddof=1,
        )

        if (
            not np.isfinite(beta)
            or not np.isfinite(intercept)
            or not np.isfinite(residual_mean)
            or not np.isfinite(residual_std)
            or residual_std <= np.finfo(float).eps
        ):
            continue

        current_residual = (
            y_current
            - intercept
            - beta * x_current
        )

        zscore = (
            current_residual - residual_mean
        ) / residual_std

        if not np.isfinite(zscore):
            continue

        if not (
            PAIRS_Z_LOWER
            <= abs(zscore)
            <= PAIRS_Z_UPPER
        ):
            continue

        direction = int(-np.sign(zscore))

        if direction == 0:
            continue

        y_dollar_limit = pair_dollar_limit(y_name)
        x_dollar_limit = pair_dollar_limit(x_name)

        # Residual position:
        #
        #     q_y = direction * scale
        #     q_x = -direction * beta * scale
        #
        # Select the largest scale that respects both instruments'
        # pair-strategy dollar limits.
        possible_scales = [
            y_dollar_limit / y_current,
        ]

        if abs(beta) > np.finfo(float).eps:
            possible_scales.append(
                x_dollar_limit
                / (abs(beta) * x_current)
            )

        scale = min(possible_scales)

        if not np.isfinite(scale) or scale <= 0.0:
            continue

        y_quantity = int(
            np.rint(direction * scale)
        )
        x_quantity = int(
            np.rint(-direction * beta * scale)
        )

        # Accumulate rather than overwrite, allowing for future overlapping
        # pair definitions.
        target[yi] += y_quantity
        target[xi] += x_quantity

    return target


# =============================================================================
# Multi-instrument individual strategy
# =============================================================================

def Trade_individual(prices, previous_positions):
    """
    Add each individual mean-reversion signal to the existing position.

    Positions are not clipped here. Apply position limits only after all
    instruments and strategies have finished contributing their bets.
    """
    prices = validate_prices(prices)
    position = np.asarray(previous_positions, dtype=int).copy()

    if position.shape != (N_INST,):
        raise ValueError(
            f"previous_positions must have shape ({N_INST},), "
            f"but received {position.shape}"
        )

    if (
        INDIVIDUAL_WINDOW < 2
        or INDIVIDUAL_Z_LOWER < 0.0
        or INDIVIDUAL_Z_UPPER < INDIVIDUAL_Z_LOWER
        or prices.shape[1] < INDIVIDUAL_WINDOW + 1
    ):
        return position

    individual_indices = np.asarray(
        [INDEX[name] for name in INDIVIDUAL],
        dtype=int,
    )

    if len(individual_indices) < 2:
        return position

    history = prices[
        individual_indices,
        -(INDIVIDUAL_WINDOW + 1):,
    ]

    valid = (
        np.isfinite(history).all(axis=1)
        & (history > 0.0).all(axis=1)
    )
    valid_locations = np.flatnonzero(valid)

    if len(valid_locations) < 2:
        return position

    valid_indices = individual_indices[valid_locations]
    valid_history = history[valid_locations]

    number_of_instruments = len(valid_indices)

    for target_location in range(number_of_instruments):
        target_index = valid_indices[target_location]

        predictor_locations = (
            np.arange(number_of_instruments) != target_location
        )

        y = valid_history[target_location]
        X = valid_history[predictor_locations].T

        y_train = y[:-1]
        X_train = X[:-1]

        y_current = y[-1]
        X_current = X[-1]

        number_of_parameters = X_train.shape[1] + 1

        if len(y_train) <= number_of_parameters:
            continue

        design_train = np.column_stack(
            [
                np.ones(len(X_train)),
                X_train,
            ]
        )

        coefficients, _, rank, _ = np.linalg.lstsq(
            design_train,
            y_train,
            rcond=None,
        )

        if (
            rank < number_of_parameters
            or not np.isfinite(coefficients).all()
        ):
            continue

        fitted_train = design_train @ coefficients
        training_residuals = y_train - fitted_train

        residual_mean = np.mean(training_residuals)
        residual_std = np.std(
            training_residuals,
            ddof=1,
        )

        if (
            not np.isfinite(residual_mean)
            or not np.isfinite(residual_std)
            or residual_std <= np.finfo(float).eps
        ):
            continue

        current_design = np.concatenate(
            ([1.0], X_current)
        )
        predicted_current = current_design @ coefficients
        current_residual = y_current - predicted_current

        zscore = (
            current_residual - residual_mean
        ) / residual_std

        if not np.isfinite(zscore):
            continue

        if not (
            INDIVIDUAL_Z_LOWER
            <= abs(zscore)
            <= INDIVIDUAL_Z_UPPER
        ):
            continue

        direction = int(-np.sign(zscore))

        if direction == 0:
            continue

        name = INSTRUMENTS[target_index]
        dollar_limit = individual_dollar_limit(name)

        quantity = int(
            np.floor(dollar_limit / y_current)
        )

        current_signal_bet = direction * quantity
        position[target_index] += current_signal_bet

    return position

def Trade_individual_2(prices, previous_positions):
    """
    Trade the current EMA trend only when EMA trend-following was profitable
    over the historical LOOKBACK period.
    """
    prices = validate_prices(prices)
    position = np.asarray(previous_positions, dtype=int).copy()

    if position.shape != (N_INST,):
        raise ValueError(
            f"previous_positions must have shape ({N_INST},), "
            f"but received {position.shape}"
        )

    if (
        isinstance(EMA_WINDOW, bool)
        or not isinstance(EMA_WINDOW, (int, np.integer))
        or EMA_WINDOW < 2
        or isinstance(LOOKBACK, bool)
        or not isinstance(LOOKBACK, (int, np.integer))
        or LOOKBACK < 2
    ):
        return position

    required_observations = EMA_WINDOW + LOOKBACK + 1

    if prices.shape[1] < required_observations:
        return position

    for name in INDIVIDUAL:
        instrument_index = INDEX[name]

        history = prices[
            instrument_index,
            -required_observations:
        ]

        if (
            not np.isfinite(history).all()
            or np.any(history <= 0.0)
        ):
            continue

        ema = (
            pd.Series(history)
            .ewm(
                span=EMA_WINDOW,
                adjust=False,
            )
            .mean()
            .to_numpy()
        )

        # Historical trend signals:
        # price above EMA -> +1
        # price below EMA -> -1
        historical_signals = np.sign(
            history[EMA_WINDOW:-1]
            - ema[EMA_WINDOW:-1]
        )

        next_returns = np.diff(
            np.log(history)
        )[EMA_WINDOW:]

        active = historical_signals != 0

        if not np.any(active):
            continue

        trend_score = np.mean(
            historical_signals[active]
            * next_returns[active]
        )

        # Do not trade unless trend following worked over LOOKBACK.
        if not np.isfinite(trend_score) or trend_score <= 0.0:
            continue

        direction = int(
            np.sign(history[-1] - ema[-1])
        )

        if direction == 0:
            continue

        current_price = history[-1]

        quantity = int(
            np.floor(
                individual_dollar_limit(name)
                / current_price
            )
        )

        if quantity <= 0:
            continue

        position[instrument_index] += direction * quantity

    return position
# =============================================================================
# Main entry point
# =============================================================================

currentPos = np.zeros(N_INST, dtype=int)


def getMyPosition(prcSoFar):
    global currentPos

    prices = validate_prices(prcSoFar)

    # Start from the previously held positions.
    position = np.asarray(currentPos, dtype=int).copy()

    # Add pair-strategy bets without clipping.
    if ENABLE_PAIRS:
        position += Trade_pairs(prices)

    # Trade_individual adds its bets to the supplied positions.
    if ENABLE_INDIVIDUAL:
        position = Trade_individual_2(
            prices,
            position,
        )

    # Clip only once, after all bets have been added.
    currentPos = cap_combined_positions(
        position,
        prices,
    )

    return currentPos.copy()