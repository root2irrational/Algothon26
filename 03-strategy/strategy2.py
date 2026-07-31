"""SIG Algothon pairs and multi-instrument residual strategy."""

import numpy as np
from sklearn.linear_model import Ridge


N_INST = 51
ALGO_INDEX = 0
MAX_POS_ALGO = 100_000.0
MAX_POS_ELSE = 10_000.0

ENABLE_PAIRS = True
ENABLE_INDIVIDUAL = True

PAIR_BET = 10_000.0
MULTI_PAIR_BET_ALGO = 1_000.0
MULTI_PAIR_BET = 10_000.0

PAIRS_WINDOW = 250
PAIRS_Z_LOWER = 0.0
PAIRS_Z_UPPER = 10.0

INDIVIDUAL_WINDOW = 250
INDIVIDUAL_RIDGE_ALPHA = 0.05
INDIVIDUAL_Z_LOWER = 0.0
INDIVIDUAL_Z_UPPER = 10.0

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

if len(INSTRUMENTS) != N_INST:
    raise RuntimeError(f"Expected {N_INST} instruments.")

INDEX = {name: index for index, name in enumerate(INSTRUMENTS)}
PAIR_INDICES = tuple((INDEX[y], INDEX[x]) for y, x in PAIRS)
PAIRED = frozenset(index for pair in PAIR_INDICES for index in pair)
INDIVIDUAL_INDICES = np.array(
    [index for index in range(N_INST) if index not in PAIRED],
    dtype=int,
)

ABSOLUTE_LIMITS = np.full(N_INST, MAX_POS_ELSE)
ABSOLUTE_LIMITS[ALGO_INDEX] = MAX_POS_ALGO
PAIR_LIMITS = np.minimum(ABSOLUTE_LIMITS, PAIR_BET)
INDIVIDUAL_LIMITS = np.minimum(ABSOLUTE_LIMITS, MULTI_PAIR_BET)
INDIVIDUAL_LIMITS[ALGO_INDEX] = min(
    MAX_POS_ALGO,
    MULTI_PAIR_BET_ALGO,
)


def _prices(prices):
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 2 or prices.shape[0] != N_INST:
        raise ValueError(f"prices must have shape ({N_INST}, observations)")
    return prices


def Trade_pairs(prices):
    """Trade configured pairs using out-of-sample OLS residual z-scores."""
    prices = _prices(prices)
    positions = np.zeros(N_INST, dtype=np.int64)
    if (
        PAIRS_WINDOW < 2
        or PAIRS_Z_LOWER < 0
        or PAIRS_Z_UPPER < PAIRS_Z_LOWER
        or prices.shape[1] <= PAIRS_WINDOW
    ):
        return positions

    for y_index, x_index in PAIR_INDICES:
        pair = prices[[y_index, x_index], -(PAIRS_WINDOW + 1):]
        if not np.isfinite(pair).all() or np.any(pair <= 0):
            continue

        y, x = pair[:, :-1]
        if np.ptp(x) <= np.finfo(float).eps:
            continue

        beta, intercept = np.polyfit(x, y, 1)
        residuals = y - (intercept + beta * x)
        residual_std = residuals.std(ddof=1)
        if (
            not np.isfinite(residual_std)
            or residual_std <= np.finfo(float).eps
        ):
            continue

        current = pair[:, -1]
        current_residual = current[0] - (intercept + beta * current[1])
        z_score = (current_residual - residuals.mean()) / residual_std
        if not np.isfinite(z_score) or not (
            PAIRS_Z_LOWER <= abs(z_score) <= PAIRS_Z_UPPER
        ):
            continue

        direction = -np.sign(z_score)
        if direction == 0:
            continue

        scale = PAIR_LIMITS[y_index] / current[0]
        if abs(beta) > np.finfo(float).eps:
            scale = min(
                scale,
                PAIR_LIMITS[x_index] / (abs(beta) * current[1]),
            )
        positions[y_index] += round(direction * scale)
        positions[x_index] += round(-direction * beta * scale)

    return positions


def Trade_individual(prices):
    """Trade each target's Ridge residual against all other valid instruments."""
    prices = _prices(prices)
    positions = np.zeros(N_INST, dtype=np.int64)
    if (
        INDIVIDUAL_WINDOW < 2
        or INDIVIDUAL_RIDGE_ALPHA < 0
        or INDIVIDUAL_Z_LOWER < 0
        or INDIVIDUAL_Z_UPPER < INDIVIDUAL_Z_LOWER
        or prices.shape[1] <= INDIVIDUAL_WINDOW
    ):
        return positions

    history = prices[
        INDIVIDUAL_INDICES,
        -(INDIVIDUAL_WINDOW + 1):,
    ]
    valid = np.isfinite(history).all(axis=1) & (history > 0).all(axis=1)
    indices, history = INDIVIDUAL_INDICES[valid], history[valid]
    if len(indices) < 2:
        return positions

    for target_location, target_index in enumerate(indices):
        predictors = np.arange(len(indices)) != target_location
        y, X = history[target_location], history[predictors].T
        model = Ridge(
            alpha=INDIVIDUAL_RIDGE_ALPHA,
            fit_intercept=True,
            solver="svd",
        ).fit(X[:-1], y[:-1])

        residuals = y[:-1] - model.predict(X[:-1])
        residual_std = residuals.std(ddof=1)
        if (
            not np.isfinite(residual_std)
            or residual_std <= np.finfo(float).eps
        ):
            continue

        current_residual = y[-1] - model.predict(X[-1:])[0]
        z_score = (current_residual - residuals.mean()) / residual_std
        if not np.isfinite(z_score) or not (
            INDIVIDUAL_Z_LOWER <= abs(z_score) <= INDIVIDUAL_Z_UPPER
        ):
            continue

        direction = -np.sign(z_score)
        if direction:
            quantity = np.floor(
                INDIVIDUAL_LIMITS[target_index] / y[-1]
            )
            positions[target_index] = round(direction * quantity)

    return positions


def _cap_positions(positions, prices):
    if prices.shape[1] == 0:
        return np.zeros(N_INST, dtype=np.int64)

    current = prices[:, -1]
    valid = np.isfinite(current) & (current > 0)
    maximum = np.zeros(N_INST, dtype=np.int64)
    maximum[valid] = np.floor(
        ABSOLUTE_LIMITS[valid] / current[valid]
    ).astype(np.int64)
    return np.clip(
        np.asarray(positions, dtype=np.int64),
        -maximum,
        maximum,
    )


currentPos = np.zeros(N_INST, dtype=np.int64)


def getMyPosition(prcSoFar):
    """Return combined target positions for the evaluator."""
    global currentPos

    prices = _prices(prcSoFar)
    positions = np.zeros(N_INST, dtype=np.int64)
    if ENABLE_PAIRS:
        positions += Trade_pairs(prices)
    if ENABLE_INDIVIDUAL:
        positions += Trade_individual(prices)

    currentPos = _cap_positions(positions, prices)
    return currentPos.copy()
