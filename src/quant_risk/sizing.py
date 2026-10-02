"""Turn signals into numbers of contracts, so that the portfolio runs at a target level of risk.

One crude oil contract moves around $1,500 a day; one 10-year note contract around $400. Holding
"one of each" would leave the book dominated by oil. Volatility targeting sizes every position by
its risk instead of its count:

1. **Risk budget.** Capital x annual volatility target, converted to a daily dollar amount, is
   shared between instruments: equally between sectors, then equally inside each sector, so ten
   correlated grain contracts do not outweigh three equity indices.
2. **Contracts.** An instrument's share of the budget divided by the dollar volatility of one
   contract gives the number of contracts at full signal strength. The signal scales that down.
3. **Diversification multiplier.** Instruments that are not perfectly correlated partly cancel
   each other out, so a portfolio of 35 budgets each sized to the target runs *below* the target.
   The multiplier (1 / sqrt(w' C w), from correlations known at the time) scales positions back up.
   It is deliberately conservative: negative correlations count as zero and it is capped at 2.5,
   so a well-diversified book realises somewhat less volatility than the target, never more.
4. **Risk overlay.** The portfolio's ex-ante volatility, sqrt(x' S x) with S the covariance matrix
   of one-contract P&L, is checked every day. If it exceeds the target by more than a set ratio,
   for example when correlations jump in a crisis, every position is scaled down. Never up.
5. **Buffer.** A position is only traded when it drifts more than a fraction of a full position
   away from its ideal size, which avoids paying costs for tiny adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_risk.bars import pnl_per_contract
from quant_risk.fx import to_usd, usd_per_unit
from quant_risk.instruments import get_instrument
from quant_risk.strategies import daily_price_vol

TRADING_DAYS = 252


def usd_vol_per_contract(closes: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    """Daily volatility, in USD, of holding one contract of each instrument."""
    multipliers = pd.Series({s: get_instrument(s).multiplier for s in closes.columns})
    fx = pd.DataFrame({s: usd_per_unit(get_instrument(s).currency, closes.index) for s in closes.columns})
    return daily_price_vol(closes, span) * multipliers * fx


def sector_weights(symbols: list[str]) -> pd.Series:
    """Equal weight per sector, split equally between the instruments inside it. Sums to 1."""
    sectors = pd.Series({s: get_instrument(s).sector for s in symbols})
    per_sector = 1.0 / sectors.nunique()
    return sectors.map(lambda sector: per_sector / (sectors == sector).sum()).astype(float)


def diversification_multiplier(
    pnl_usd: pd.DataFrame, weights: pd.Series, *, cap: float = 2.5, min_days: int = 250
) -> pd.Series:
    """1 / sqrt(w' C w), re-estimated at each year end from correlations known up to then.

    Negative correlations are floored at zero: they are unstable and would inflate the multiplier.
    Before `min_days` of history exist, the multiplier is 1 (no adjustment).
    """
    normalised = pnl_usd / pnl_usd.ewm(span=60, min_periods=60).std().shift(1)
    estimates = {}
    for year_end in pnl_usd.resample("YE").last().index:
        history = normalised.loc[:year_end].iloc[-min_days * 5 :]  # last ~5 years
        live = history.columns[history.count() >= min_days]
        if len(live) == 0:
            continue
        corr = history[live].corr().fillna(0.0).clip(lower=0.0).to_numpy()
        w = weights[live].to_numpy() / weights[live].sum()
        estimates[year_end] = min(1.0 / np.sqrt(w @ corr @ w), cap)
    # An estimate made at a year end is used from the next day on.
    timeline = pd.Series(estimates, dtype=float).reindex(pnl_usd.index.union(estimates)).ffill().shift(1)
    return timeline.reindex(pnl_usd.index).fillna(1.0)


def ewma_covariance(pnl_usd: pd.DataFrame, span: int = 125) -> np.ndarray:
    """Exponentially weighted covariance matrix of daily P&L for every date, shape (days, n, n).

    The RiskMetrics recursion S_t = lam * S_{t-1} + (1 - lam) * r_t r_t', with lam = 1 - 2 / (span + 1)
    and daily P&L treated as zero-mean. A day an instrument did not trade counts as zero P&L. The
    matrix on day t uses P&L up to and including day t, which is known at that day's close.
    Dates before `span` days of history are NaN.
    """
    lam = 1.0 - 2.0 / (span + 1)
    r = np.nan_to_num(pnl_usd.to_numpy(dtype=float))
    n_days, n = r.shape
    out = np.empty((n_days, n, n))
    current = np.zeros((n, n))
    for t in range(n_days):
        current = lam * current + (1.0 - lam) * np.outer(r[t], r[t])
        out[t] = current
    out[:span] = np.nan
    return out


def ex_ante_vol(positions: pd.DataFrame, pnl_usd: pd.DataFrame, span: int = 125) -> pd.Series:
    """Predicted daily USD volatility of the portfolio: sqrt(x' S x) with an EWMA covariance S."""
    pnl_usd = pnl_usd.reindex(index=positions.index, columns=positions.columns)
    cov = ewma_covariance(pnl_usd, span)
    x = np.nan_to_num(positions.to_numpy(dtype=float))
    variance = np.einsum("ti,tij,tj->t", x, cov, x)
    return pd.Series(np.sqrt(np.clip(variance, 0.0, None)), index=positions.index)


def apply_buffer(ideal: pd.DataFrame, full_position: pd.DataFrame, buffer: float = 0.1) -> pd.DataFrame:
    """Whole contracts, traded only when the ideal position leaves a band around the current one.

    The band is +/- `buffer` x the full-signal position. When the ideal moves outside it, trade to
    the nearest edge of the band, not all the way to the ideal.
    """
    out = pd.DataFrame(0.0, index=ideal.index, columns=ideal.columns)
    for symbol in ideal.columns:
        target = ideal[symbol].to_numpy()
        width = buffer * np.abs(full_position[symbol].to_numpy())
        held = np.zeros(len(target))
        current = 0.0
        for t in range(len(target)):
            if np.isnan(target[t]) or np.isnan(width[t]):
                current = 0.0
            else:
                low, high = target[t] - width[t], target[t] + width[t]
                if current < low:
                    current = float(np.ceil(low - 1e-9))
                elif current > high:
                    current = float(np.floor(high + 1e-9))
            held[t] = current
        out[symbol] = held
    return out


@dataclass
class SizingResult:
    positions: pd.DataFrame  # whole contracts held at each close
    ideal: pd.DataFrame  # unrounded, unbuffered contracts after the overlay
    full_position: pd.DataFrame  # contracts at full signal strength
    multiplier: pd.Series  # diversification multiplier
    ex_ante_vol: pd.Series  # predicted daily USD vol of the ideal book before the overlay
    overlay_scale: pd.Series  # 1.0 unless the overlay cut risk


def size_positions(
    signals: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    capital: float,
    target_vol: float = 0.15,
    max_risk_ratio: float = 1.5,
    buffer: float = 0.1,
) -> SizingResult:
    """Contracts to hold for each instrument and day, at `target_vol` annual volatility on `capital`."""
    daily_budget = capital * target_vol / np.sqrt(TRADING_DAYS)
    weights = sector_weights(list(closes.columns))
    pnl_usd = to_usd(pnl_per_contract(closes.ffill()))
    multiplier = diversification_multiplier(pnl_usd, weights).reindex(closes.index).ffill().fillna(1.0)

    full = (daily_budget * weights / usd_vol_per_contract(closes)).mul(multiplier, axis=0)
    ideal = signals * full

    predicted = ex_ante_vol(ideal, pnl_usd)
    scale = (max_risk_ratio * daily_budget / predicted).clip(upper=1.0).fillna(1.0)
    ideal = ideal.mul(scale, axis=0)

    positions = apply_buffer(ideal, full, buffer)
    return SizingResult(positions, ideal, full, multiplier, predicted, scale)
