"""FX rates for turning non-USD P&L into US dollars.

The portfolio is measured in USD, but DAX, Euro Stoxx 50 and Bund futures pay their P&L in euros.
Each euro amount is converted at that day's EUR/USD rate.

Spot EUR/USD (``data/fx/EURUSD.parquet``) covers only part of the history. Outside it, the rate is
extended with the daily *changes* of the Euro FX futures (EC), anchored to the spot rate at the
edge of its coverage. A futures price moves almost one-for-one with spot (the difference is a small
interest-rate carry), so this is a close approximation, and it never uses EC's back-adjusted
*level*, which drifts away from the real exchange rate over the years.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_risk.bars import data_dir, read_bars
from quant_risk.instruments import get_instrument

# currency -> (spot file in data/fx, futures symbol used to extend it)
FX_SOURCES = {"EUR": ("EURUSD", "EC")}


def fx_dir() -> Path:
    return data_dir().parent / "fx"


def usd_per_unit(currency: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Daily USD value of one unit of `currency` on each date (1.0 for USD)."""
    if currency == "USD":
        return pd.Series(1.0, index=dates)
    if currency not in FX_SOURCES:
        raise KeyError(f"no FX source for {currency}; add one to FX_SOURCES")
    spot_symbol, futures_symbol = FX_SOURCES[currency]
    spot = read_bars(spot_symbol, fx_dir())["close"]
    futures = read_bars(futures_symbol)["close"]
    rate = extend_with_futures(spot, futures)
    return rate.reindex(rate.index.union(dates)).ffill().bfill().reindex(dates)


def extend_with_futures(spot: pd.Series, futures: pd.Series) -> pd.Series:
    """Spot where available; before and after, spot at the edge plus the futures' price change."""
    first, last = spot.index[0], spot.index[-1]
    before = futures.loc[futures.index < first]
    after = futures.loc[futures.index > last]
    pieces = []
    if not before.empty:
        anchor = futures.asof(first)
        pieces.append(spot.iloc[0] + (before - anchor))
    pieces.append(spot)
    if not after.empty:
        anchor = futures.asof(last)
        pieces.append(spot.iloc[-1] + (after - anchor))
    return pd.concat(pieces).sort_index()


def to_usd(pnl: pd.DataFrame) -> pd.DataFrame:
    """Convert a frame of daily P&L (one column per instrument, own currency) into USD."""
    out = pnl.copy()
    for symbol in pnl.columns:
        currency = get_instrument(symbol).currency
        if currency != "USD":
            out[symbol] = pnl[symbol] * usd_per_unit(currency, pnl.index)
    return out
