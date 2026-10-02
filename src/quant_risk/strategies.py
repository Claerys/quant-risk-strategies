"""Trading signals for the futures universe.

Every strategy takes a frame of closing prices (one column per instrument) and returns a frame of
the same shape with a signal between -1 (fully short) and +1 (fully long). A signal only says which
way and how strongly; how many contracts that means is decided by position sizing.

Two rules hold for every strategy here:

* **No look-ahead.** The signal on day t uses closes up to and including day t. The backtest then
  trades it from day t+1, never on the price that produced it.
* **Price differences, not ratios.** The futures prices are back-adjusted and can be negative, so a
  return like close_t / close_{t-h} - 1 is meaningless. Trends are measured as price changes,
  standardised by each instrument's own typical daily price change. That makes a signal comparable
  across crude oil, bonds and currencies, and unaffected by the back-adjustment.

``time_series_momentum`` and ``multi_horizon_trend`` are my long/cash strategies from the BSQF
project, rebuilt for futures: long/short, and on price differences.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def on_own_trading_days(fn: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    """Run `fn` on each instrument's own trading days, then align the results again.

    Closes for many exchanges share one calendar, so an instrument has blank days whenever only
    other markets were open. Rolling windows and lags must count that instrument's trading days,
    not calendar slots: a 20-day average should never come out blank because of a holiday abroad.
    On those blank days the signal keeps its last value.
    """

    @functools.wraps(fn)
    def wrapper(closes: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        columns = {}
        for symbol in closes.columns:
            own = closes[[symbol]].dropna()
            columns[symbol] = fn(own, *args, **kwargs)[symbol].reindex(closes.index)
        out = pd.DataFrame(columns, index=closes.index)
        return out.ffill().where(closes.ffill().notna())

    return wrapper


@on_own_trading_days
def daily_price_vol(closes: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    """Exponentially weighted standard deviation of daily price changes, in price units."""
    return closes.diff().ewm(span=span, min_periods=span).std()


def standardised_trend(closes: pd.DataFrame, horizon: int, vol_span: int = 60) -> pd.DataFrame:
    """Price change over `horizon` days, in units of the change one would expect from noise.

    Under a random walk with daily volatility sigma, an h-day change has volatility sigma*sqrt(h).
    A value of +2 means prices rose twice as much as noise alone would typically move them.
    """
    expected = daily_price_vol(closes, vol_span) * np.sqrt(horizon)
    return (closes - closes.shift(horizon)) / expected.replace(0, np.nan)


@on_own_trading_days
def buy_and_hold(closes: pd.DataFrame) -> pd.DataFrame:
    """Always long. The baseline every other strategy has to beat."""
    return closes.notna().astype(float).where(closes.notna())


@on_own_trading_days
def time_series_momentum(
    closes: pd.DataFrame, lookback: int = 252, skip: int = 21, long_only: bool = False
) -> pd.DataFrame:
    """12-1 momentum: long if the price rose from 12 months ago to 1 month ago, short if it fell.

    The most recent month is skipped because very short-term moves tend to partly reverse.
    """
    move = closes.shift(skip) - closes.shift(lookback)
    signal = np.sign(move)
    return signal.clip(lower=0) if long_only else signal


@on_own_trading_days
def multi_horizon_trend(
    closes: pd.DataFrame,
    horizons: tuple[int, ...] = (21, 63, 126, 252),
    vol_span: int = 60,
    cap: float = 2.0,
    long_only: bool = False,
) -> pd.DataFrame:
    """Average of standardised trends over 1, 3, 6 and 12 months, scaled to [-1, 1].

    Combining horizons reacts faster than 12-month momentum alone while staying less noisy than a
    1-month signal. A combined score of `cap` (2 by default) or more is a full-strength position.
    """
    scores = [standardised_trend(closes, h, vol_span) for h in horizons]
    combined = sum(scores) / len(scores)
    signal = combined.clip(-cap, cap) / cap
    return signal.clip(lower=0) if long_only else signal


@on_own_trading_days
def ma_crossover(closes: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.DataFrame:
    """Long while the fast moving average is above the slow one, short while it is below."""
    if fast >= slow:
        raise ValueError(f"fast window must be shorter than slow, got {fast} >= {slow}")
    return np.sign(closes.rolling(fast).mean() - closes.rolling(slow).mean())


@on_own_trading_days
def mean_reversion(closes: pd.DataFrame, window: int = 20, full_at: float = 2.0) -> pd.DataFrame:
    """Lean against stretched prices: short when far above the recent average, long when far below.

    The signal grows with the z-score and is full strength at `full_at` standard deviations.
    """
    mean = closes.rolling(window).mean()
    std = closes.rolling(window).std().replace(0, np.nan)
    z = (closes - mean) / std
    return (-z / full_at).clip(-1, 1)


Strategy = Callable[[pd.DataFrame], pd.DataFrame]

STRATEGIES: dict[str, Strategy] = {
    "buy_and_hold": buy_and_hold,
    "time_series_momentum": time_series_momentum,
    "multi_horizon_trend": multi_horizon_trend,
    "ma_crossover": ma_crossover,
    "mean_reversion": mean_reversion,
}
