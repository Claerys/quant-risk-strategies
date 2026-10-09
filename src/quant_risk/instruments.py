"""Contract specifications for the futures universe.

Risk is measured in money, not in percent. A futures price only becomes money through the
contract multiplier: one point on ES is worth $50, one cent on corn is worth $50, one dollar on
crude oil is worth $1,000. Every P&L, exposure and VaR number in this project goes through
``Instrument.multiplier``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
from importlib import resources


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    exchange: str
    currency: str
    multiplier: float  # money (in `currency`) per 1.0 move in the quoted price
    price_unit: str
    asset_class: str
    sector: str
    primary_exchange: str = ""  # stocks only: the listing SMART should resolve to

    def pnl(self, price_change: float, contracts: float = 1.0) -> float:
        """Profit or loss, in the contract currency, of a price move on `contracts` contracts."""
        return price_change * self.multiplier * contracts

    def notional(self, price: float, contracts: float = 1.0) -> float:
        """Market value of the position: price x multiplier x contracts."""
        return price * self.multiplier * contracts


def _read_universe(filename: str) -> dict[str, Instrument]:
    text = resources.files("quant_risk").joinpath(filename).read_text(encoding="utf-8")
    instruments = {}
    for row in csv.DictReader(text.splitlines()):
        multiplier = float(row["multiplier"])
        if multiplier <= 0:
            raise ValueError(f"{row['symbol']}: multiplier must be positive, got {multiplier}")
        instruments[row["symbol"]] = Instrument(
            symbol=row["symbol"],
            name=row["name"],
            exchange=row["exchange"],
            currency=row["currency"],
            multiplier=multiplier,
            price_unit=row["price_unit"],
            asset_class=row["asset_class"],
            sector=row["sector"],
            primary_exchange=row.get("primary_exchange") or "",
        )
    return instruments


@cache
def load_instruments() -> dict[str, Instrument]:
    """The futures universe (instruments.csv), keyed by symbol."""
    return _read_universe("instruments.csv")


@cache
def load_equities() -> dict[str, Instrument]:
    """The stock universe (equities.csv): one share is one contract with a multiplier of 1.

    Kept apart from the futures universe so that the futures results never change when stocks
    are added. Point the engine at stock data with QRS_DATA_DIR=data/stocks.
    """
    return _read_universe("equities.csv")


def get_instrument(symbol: str) -> Instrument:
    for universe in (load_instruments(), load_equities()):
        if symbol in universe:
            return universe[symbol]
    raise KeyError(f"unknown instrument {symbol!r}; add it to instruments.csv (futures) or equities.csv (stocks)")
