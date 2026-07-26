import numpy as np


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

ENABLE_PAIRS = True
ENABLE_INDIVIDUAL = True

# Strategy-level dollar allocations.
PAIR_BET = 10_000

MULTI_PAIR_BET_ALGO = 1_000
MULTI_PAIR_BET = 10_000

PAIRS_WINDOW = 250
PAIRS_Z_LOWER = 0.0
PAIRS_Z_UPPER = 10.0

INDIVIDUAL_WINDOW = 250
INDIVIDUAL_Z_LOWER = 0.0
INDIVIDUAL_Z_UPPER = 10.0


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

# ALGO is deliberately excluded from the individual strategy.
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

def Trade_individual(prices):
    """
    For every instrument in INDIVIDUAL:

    1. Use that instrument as the OLS target.
    2. Use all other valid INDIVIDUAL instruments as predictors.
    3. Fit on the previous INDIVIDUAL_WINDOW raw-price observations.
    4. Calculate today's out-of-sample residual z-score.
    5. Trade only the target instrument in the mean-reversion direction.

    ALGO is excluded from INDIVIDUAL and therefore is neither a target nor
    a predictor in this strategy.
    """
    prices = validate_prices(prices)
    target = np.zeros(N_INST, dtype=int)

    if (
        INDIVIDUAL_WINDOW < 2
        or INDIVIDUAL_Z_LOWER < 0.0
        or INDIVIDUAL_Z_UPPER < INDIVIDUAL_Z_LOWER
        or prices.shape[1] < INDIVIDUAL_WINDOW + 1
    ):
        return target

    individual_indices = np.asarray(
        [
            INDEX[name]
            for name in INDIVIDUAL
        ],
        dtype=int,
    )

    if len(individual_indices) < 2:
        return target

    history = prices[
        individual_indices,
        -(INDIVIDUAL_WINDOW + 1):
    ]

    valid = (
        np.isfinite(history).all(axis=1)
        & (history > 0.0).all(axis=1)
    )
    valid_locations = np.flatnonzero(valid)

    if len(valid_locations) < 2:
        return target

    valid_indices = individual_indices[
        valid_locations
    ]
    valid_history = history[
        valid_locations
    ]

    number_of_instruments = len(valid_indices)

    for target_location in range(number_of_instruments):
        target_index = valid_indices[target_location]

        predictor_locations = (
            np.arange(number_of_instruments)
            != target_location
        )

        y = valid_history[target_location]
        X = valid_history[predictor_locations].T

        y_train = y[:-1]
        X_train = X[:-1]

        y_current = y[-1]
        X_current = X[-1]

        number_of_parameters = (
            X_train.shape[1] + 1
        )

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

        fitted_train = (
            design_train @ coefficients
        )
        training_residuals = (
            y_train - fitted_train
        )

        residual_mean = np.mean(
            training_residuals
        )
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
        predicted_current = (
            current_design @ coefficients
        )
        current_residual = (
            y_current - predicted_current
        )

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

        target[target_index] = (
            direction * quantity
        )

    return target


# =============================================================================
# Main entry point
# =============================================================================

currentPos = np.zeros(N_INST, dtype=int)


def getMyPosition(prcSoFar):
    global currentPos

    prices = validate_prices(prcSoFar)

    pair_positions = (
        Trade_pairs(prices)
        if ENABLE_PAIRS
        else np.zeros(N_INST, dtype=int)
    )

    individual_positions = (
        Trade_individual(prices)
        if ENABLE_INDIVIDUAL
        else np.zeros(N_INST, dtype=int)
    )

    combined_positions = (
        pair_positions
        + individual_positions
    )

    currentPos = cap_combined_positions(
        combined_positions,
        prices,
    )

    return currentPos.copy()
