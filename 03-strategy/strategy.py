import numpy as np


# =============================================================================
# Constants
# =============================================================================

N_INST = 51
MAX_POS_ALGO, MAX_POS_ELSE = 100_000, 10_000

ENABLE_PAIRS = True
ENABLE_INDIVIDUAL = False

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

BET_POS_ALGO, BET_POS_ELSE = 10_000, 10_000
PAIRS_WINDOW, PAIRS_THRESHOLD = 40, 0.0

# Rank instruments by their return over this lookback.
INDIVIDUAL_WINDOW = 20

# Long this many winners and short the same number of losers.
INDIVIDUAL_COUNT = 5

# Equal dollar exposure per selected instrument. Keeping this at or below
# BET_POS_ELSE means ALGO and non-ALGO names can all receive equal notionals.
INDIVIDUAL_DOLLARS_PER_ASSET = 1_000

ALL_PAIRS = [
    ('MHRM', 'EAFC'), ('ACIX', 'ITPA'), ('EORC', 'NGTE'), ('AENO', 'NWIG'), 
    ('SMAH', 'ILVX'), ('HUXZ', 'ACAC'), ('ALUT', 'CCNS'), ('ALGO', 'BENI'), 
    ('NPCK', 'SRTX'), ('FWWG', 'BLBT'), ('SRNA', 'IHOZ'), ('CTGI', 'EELT'), 
    ('HETT', 'ULXY'), ('RTTH', 'RRES'), ('MTNS', 'HTRK'), ('DUCT', 'GARI'), 
    ('MSDP', 'SPLZ'), ('ELLT', 'DIHO'), ('RCRI', 'NAYO'), ('HRND', 'AETS'), 
    ('MDGI', 'AGVF'), ('AMRP', 'FCSG'), ('LSST', 'HRET'), ('CUBO', 'ANSO'), ('OTCS', 'MMBT')]

ALL_PAIRS_2 = [
    ('MHRM', 'EAFC'), ('ACIX', 'ITPA'), ('EORC', 'NGTE'), ('AENO', 'NWIG'), 
    ('SMAH', 'ILVX'), ('HUXZ', 'ACAC'), ('FWWG', 'BLBT'), ('CTGI', 'EELT')]

PAIRS = [
    ("AENO", "NWIG"), ("SMAH", "ILVX"), ("ACIX", "ITPA"), ("MHRM", "EAFC"),
    ("EORC", "NGTE"), ("NPCK", "SRTX"), ("HUXZ", "ACAC"), ("HETT", "ULXY"),
    ("FWWG", "BLBT"), ("NAYO", "EELT"), ("ALUT", "CCNS"), ("RRES", "CTGI"),
]

PAIRS = ALL_PAIRS_2

INDEX = {name: i for i, name in enumerate(INSTRUMENTS)}
PAIRED = {name for pair in PAIRS for name in pair}

# The cross-sectional strategy trades instruments not already used by pairs.
INDIVIDUAL = [name for name in INSTRUMENTS if name not in PAIRED]

currentPos = np.zeros(N_INST, dtype=int)


# =============================================================================
# Helpers
# =============================================================================

def position_limits(instrument_name):
    """Return bet size and maximum dollar position for an instrument."""
    return (
        (BET_POS_ALGO, MAX_POS_ALGO)
        if instrument_name == "ALGO"
        else (BET_POS_ELSE, MAX_POS_ELSE)
    )


def _within_limit(name, shares, price):
    """Check whether an integer position respects its dollar limit."""
    _, dollar_limit = position_limits(name)
    return abs(shares * price) <= dollar_limit


def _reduce_rounding_imbalance(target, prices, selected_names):
    """
    Reduce net dollar exposure created by integer-share rounding.

    Exact dollar neutrality is generally impossible with integer shares at
    different prices. This greedily makes one-share adjustments whenever they
    reduce the absolute net exposure without breaching a position limit.
    """
    selected_indices = [INDEX[name] for name in selected_names]
    current_prices = prices[selected_indices, -1]

    for _ in range(100):
        shares = target[selected_indices]
        net_dollars = float(shares @ current_prices)

        best_improvement = 0.0
        best_index = None
        best_change = 0

        for name, i, price in zip(
            selected_names,
            selected_indices,
            current_prices,
        ):
            if net_dollars > 0:
                # Reduce long exposure or increase short exposure.
                change = -1
            elif net_dollars < 0:
                # Increase long exposure or reduce short exposure.
                change = 1
            else:
                return

            proposed_shares = target[i] + change

            # Do not cross through zero and create an unintended position.
            if target[i] > 0 and proposed_shares < 0:
                continue
            if target[i] < 0 and proposed_shares > 0:
                continue

            if not _within_limit(name, proposed_shares, price):
                continue

            new_net = net_dollars + change * price
            improvement = abs(net_dollars) - abs(new_net)

            if improvement > best_improvement:
                best_improvement = improvement
                best_index = i
                best_change = change

        if best_index is None:
            return

        target[best_index] += best_change


# =============================================================================
# Strategies
# =============================================================================

def Trade_pairs(prices):
    """Return target pair positions from rolling OLS residual mean reversion."""
    target = np.zeros(N_INST, dtype=int)

    if prices.shape[1] < PAIRS_WINDOW:
        return target

    for y_name, x_name in PAIRS:
        yi, xi = INDEX[y_name], INDEX[x_name]
        y = prices[yi, -PAIRS_WINDOW:]
        x = prices[xi, -PAIRS_WINDOW:]

        if (
            not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.any(x <= 0)
            or np.any(y <= 0)
        ):
            continue

        xc = x - x.mean()
        yc = y - y.mean()
        variance = xc @ xc

        if variance <= np.finfo(float).eps:
            continue

        beta = (xc @ yc) / variance
        residual = yc - beta * xc
        residual_std = residual.std()

        if (
            not np.isfinite(beta)
            or residual_std <= np.finfo(float).eps
        ):
            continue

        zscore = residual[-1] / residual_std

        if abs(zscore) <= PAIRS_THRESHOLD:
            continue

        direction = -np.sign(zscore)
        bet = min(
            position_limits(y_name)[0],
            position_limits(x_name)[0],
        )

        # qx = -beta*qy, while keeping both dollar legs within `bet`.
        scale = bet / max(y[-1], abs(beta) * x[-1])

        target[yi] = int(np.rint(direction * scale))
        target[xi] = int(np.rint(-direction * beta * scale))

    return target


def Trade_individual(prices):
    """
    Trade cross-sectional momentum with a dollar-neutral long/short portfolio.

    Instruments are ranked by their INDIVIDUAL_WINDOW log return. The strategy
    buys the top INDIVIDUAL_COUNT winners and shorts the same number of losers.
    Every selected instrument receives the same target dollar exposure.
    """
    target = np.zeros(N_INST, dtype=int)

    # A W-period return needs W + 1 prices.
    if prices.shape[1] < INDIVIDUAL_WINDOW + 1:
        return target

    ranked = []

    for name in INDIVIDUAL:
        i = INDEX[name]
        start_price = prices[i, -INDIVIDUAL_WINDOW - 1]
        current_price = prices[i, -1]

        if (
            not np.isfinite(start_price)
            or not np.isfinite(current_price)
            or start_price <= 0
            or current_price <= 0
        ):
            continue

        momentum = np.log(current_price / start_price)

        if np.isfinite(momentum):
            ranked.append((momentum, name))

    if len(ranked) < 2:
        return target

    ranked.sort(key=lambda item: item[0])
    count = min(
        INDIVIDUAL_COUNT,
        len(ranked) // 2,
    )

    if count == 0:
        return target

    losers = [name for _, name in ranked[:count]]
    winners = [name for _, name in ranked[-count:]]

    for direction, names in ((-1, losers), (1, winners)):
        for name in names:
            i = INDEX[name]
            current_price = prices[i, -1]
            bet, dollar_limit = position_limits(name)

            target_dollars = min(
                INDIVIDUAL_DOLLARS_PER_ASSET,
                bet,
                dollar_limit,
            )
            shares = int(target_dollars / current_price)
            target[i] = direction * shares

    selected = losers + winners
    _reduce_rounding_imbalance(
        target,
        prices,
        selected,
    )
    # target = -1*target
    return target


# =============================================================================
# Main
# =============================================================================

def getMyPosition(prcSoFar):
    global currentPos

    prices = np.asarray(prcSoFar, dtype=float)

    if prices.ndim != 2 or prices.shape[0] != N_INST:
        raise ValueError(
            f"prcSoFar must have shape ({N_INST}, observations)"
        )

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

    currentPos = pair_positions + individual_positions
    return currentPos.copy()
