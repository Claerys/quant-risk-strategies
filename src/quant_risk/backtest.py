"""Daily backtest of a futures portfolio, in US dollars.

Inputs are the number of contracts held at each day's close and the closing prices. The engine:

1. **lags positions by one day**: contracts decided at the close of day t earn the price move of
   day t+1. This is the single place where the no-look-ahead rule is enforced.
2. **values P&L in money**: price change x contract multiplier x contracts, then converted to USD.
3. **charges trading costs** on every contract bought or sold. Roll trades are not charged (the
   back-adjusted series hides them), so costs are slightly understated.

Returns are P&L divided by a fixed capital amount. Futures need only margin, not the full contract
value, so "capital" is the account size the strategy is run on, and returns do not compound: a
$10,000 loss is the same 1% of a $1m account whenever it happens. Because a futures position earns
no interest on the notional, these returns are already excess returns over cash.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_risk.fx import to_usd
from quant_risk.instruments import get_instrument


@dataclass
class BacktestResult:
    positions: pd.DataFrame  # contracts held at each close
    pnl: pd.DataFrame  # USD P&L per instrument per day, before costs
    costs: pd.DataFrame  # USD trading costs per instrument per day
    capital: float

    @property
    def net_pnl(self) -> pd.Series:
        return (self.pnl - self.costs).sum(axis=1).rename("net_pnl")

    @property
    def returns(self) -> pd.Series:
        return (self.net_pnl / self.capital).rename("return")

    @property
    def equity(self) -> pd.Series:
        return (self.capital + self.net_pnl.cumsum()).rename("equity")


def run_backtest(
    positions: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    capital: float,
    cost_per_contract: float = 3.0,
) -> BacktestResult:
    """Simulate holding `positions` (contracts, decided at each close) through `closes`.

    `cost_per_contract` is a flat USD amount per contract traded, covering commission and the
    half-spread paid to cross the market.
    """
    positions = positions.reindex(index=closes.index, columns=closes.columns).fillna(0.0)
    multipliers = pd.Series({s: get_instrument(s).multiplier for s in closes.columns})

    # A day an instrument does not trade carries yesterday's price, so its move lands on the next
    # day it does trade instead of disappearing.
    price_change = closes.ffill().diff()
    held = positions.shift(1).fillna(0.0)
    pnl_local = (held * price_change * multipliers).fillna(0.0)
    pnl = to_usd(pnl_local)

    traded = positions.diff().abs()
    traded.iloc[0] = positions.iloc[0].abs()
    costs = traded * cost_per_contract
    return BacktestResult(positions=positions, pnl=pnl, costs=costs, capital=capital)
