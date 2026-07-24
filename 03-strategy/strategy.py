
# =============================================================================
# Imports
# =============================================================================

import numpy as np


# =============================================================================
# Constants
# =============================================================================

N_INST = 51
MAX_POS_ALGO, MAX_POS_ELSE = 100_000, 10_000

INSTRUMENTS = [
    "ALGO", "AENO", "LSST", "SRNA", "ELLT", "AMRP", "OTCS", "HETT",
    "HUXZ", "DUCT", "SMAH", "NPCK", "MSDP", "EORC", "CUBO", "HRET",
    "ANSO", "DIHO", "RTTH", "SPLZ", "NWIG", "MMBT", "MDGI", "AGVF",
    "RRES", "CTGI", "ALUT", "ACAC", "SRTX", "GARI", "RCRI", "ACIX",
    "CCNS", "MTNS", "IHOZ", "NAYO", "FWWG", "EELT", "HRND", "AETS",
    "ULXY", "BLBT", "BENI", "ITPA", "HTRK", "NGTE", "ILVX", "FCSG",
    "FARS", "MHRM", "EAFC",
]


# =============================================================================
# Parameters
# =============================================================================

BET_POS_ALGO, BET_POS_ELSE = 10_000, 1_000
PAIRS_WINDOW, PAIRS_THRESHOLD = 40, 0.0

PAIRS = [
    ("AENO", "NWIG"), ("SMAH", "ILVX"), ("ACIX", "ITPA"), ("MHRM", "EAFC"),
    ("EORC", "NGTE"), ("NPCK", "SRTX"), ("HUXZ", "ACAC"), ("HETT", "ULXY"),
    ("FWWG", "BLBT"), ("NAYO", "EELT"), ("ALUT", "CCNS"), ("RRES", "CTGI"),
]

INDEX = {name: i for i, name in enumerate(INSTRUMENTS)}
PAIRED = {name for pair in PAIRS for name in pair}
INDIVIDUAL = [name for name in INSTRUMENTS if name not in PAIRED]
currentPos = np.zeros(N_INST, dtype=int)


# =============================================================================
# Helpers
# =============================================================================

def position_limits(instrument_name):
    """Return bet size and maximum position for an instrument."""
    return ((BET_POS_ALGO, MAX_POS_ALGO) if instrument_name == "ALGO"
            else (BET_POS_ELSE, MAX_POS_ELSE))


# =============================================================================
# Strategies
# =============================================================================

def Trade_pairs(prices):
    """Return target positions from rolling OLS residual mean reversion."""
    target = np.zeros(N_INST, dtype=int)
    if prices.shape[1] < PAIRS_WINDOW:
        return target

    for y_name, x_name in PAIRS:
        yi, xi = INDEX[y_name], INDEX[x_name]
        y, x = prices[yi, -PAIRS_WINDOW:], prices[xi, -PAIRS_WINDOW:]

        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue

        xc, yc = x - x.mean(), y - y.mean()
        variance = xc @ xc
        if variance <= np.finfo(float).eps:
            continue

        beta = (xc @ yc) / variance
        residual = yc - beta * xc
        std = residual.std()

        if not np.isfinite(beta) or std <= np.finfo(float).eps:
            continue

        zscore = residual[-1] / std
        if abs(zscore) <= PAIRS_THRESHOLD:
            continue

        direction = -np.sign(zscore)
        bet = min(position_limits(y_name)[0], position_limits(x_name)[0])
        scale = bet / max(1.0, abs(beta))

        target[yi] = int(np.rint(direction * scale))
        target[xi] = int(np.rint(-direction * beta * scale))

        target[yi] = np.clip(
            target[yi], -position_limits(y_name)[1], position_limits(y_name)[1]
        )
        target[xi] = np.clip(
            target[xi], -position_limits(x_name)[1], position_limits(x_name)[1]
        )

    return target


def Trade_individual():
    """Keep all unpaired instruments flat."""
    return np.zeros(N_INST, dtype=int)


# =============================================================================
# Main
# =============================================================================

def getMyPosition(prcSoFar):
    global currentPos

    prices = np.asarray(prcSoFar, dtype=float)
    if prices.ndim != 2 or prices.shape[0] != N_INST:
        raise ValueError(f"prcSoFar must have shape ({N_INST}, observations)")

    currentPos = Trade_pairs(prices) + Trade_individual()
    return currentPos.copy()
