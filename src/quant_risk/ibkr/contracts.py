"""Turn an instrument in the universe into its IBKR futures months, failing closed.

The archive uses TradeStation-style symbols (TY, EC, JY, C...); IBKR uses its own roots (ZN, 6E,
6J, ZC...). ``futures_map.csv`` maps one to the other and records ``price_scale``: JY is quoted in
USD per 100 JPY in the archive but USD per JPY at IBKR, so prices are multiplied by 100 to match.

Nothing is guessed. A month is rejected unless its multiplier, divided by the price scale, equals
the instrument's multiplier in ``instruments.csv``, and its currency matches.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
from importlib import resources
from itertools import pairwise

import pandas as pd

from quant_risk.ibkr.contract import ContractKey
from quant_risk.instruments import Instrument


@dataclass(frozen=True)
class FuturesMap:
    ibkr_symbol: str
    exchange: str
    trading_class: str
    price_scale: float


@dataclass(frozen=True)
class ContractMonth:
    key: ContractKey  # qualified: carries conId, local symbol, expiry
    expiry: pd.Timestamp

    @property
    def local_symbol(self) -> str:
        return self.key.local_symbol


@cache
def load_futures_map() -> dict[str, FuturesMap]:
    text = resources.files("quant_risk.ibkr").joinpath("futures_map.csv").read_text(encoding="utf-8")
    return {
        row["symbol"]: FuturesMap(row["ibkr_symbol"], row["ibkr_exchange"], row["trading_class"],
                                  float(row["price_scale"]))
        for row in csv.DictReader(text.splitlines())
    }


def futures_map_for(instrument: Instrument) -> FuturesMap:
    try:
        return load_futures_map()[instrument.symbol]
    except KeyError:
        raise ValueError(f"{instrument.symbol}: no IBKR mapping in futures_map.csv") from None


def root_key(instrument: Instrument, *, include_expired: bool = False) -> ContractKey:
    mapping = futures_map_for(instrument)
    return ContractKey(symbol=mapping.ibkr_symbol, sec_type="FUT", exchange=mapping.exchange,
                       currency=instrument.currency, trading_class=mapping.trading_class,
                       include_expired=include_expired)


def futures_chain(client, instrument: Instrument, *, include_expired: bool = True) -> list[ContractMonth]:
    """Every listed (and by default expired) month of the instrument, oldest first. Raises on anything odd."""
    mapping = futures_map_for(instrument)
    months: list[ContractMonth] = []
    for found in client.candidates(root_key(instrument, include_expired=include_expired)):
        key = found.key
        label = f"{instrument.symbol} {key.local_symbol or key.expiry}"
        if key.currency != instrument.currency:
            raise ValueError(f"{label}: currency {key.currency}, expected {instrument.currency}")
        try:
            multiplier = float(key.multiplier) / mapping.price_scale
        except ValueError:
            raise ValueError(f"{label}: IBKR reported no usable multiplier ({key.multiplier!r})") from None
        if abs(multiplier - instrument.multiplier) > 1e-9:
            raise ValueError(
                f"{label}: IBKR multiplier {key.multiplier} / price scale {mapping.price_scale:g} = "
                f"{multiplier:g}, but instruments.csv says {instrument.multiplier:g}; check futures_map.csv"
            )
        if len(key.expiry) != 8:
            raise ValueError(f"{label}: expiry {key.expiry!r} is not a full date")
        months.append(ContractMonth(key, pd.Timestamp(key.expiry)))
    months.sort(key=lambda m: m.expiry)
    for earlier, later in pairwise(months):
        if earlier.expiry == later.expiry:
            raise ValueError(f"{instrument.symbol}: two contracts expire on {later.expiry:%Y-%m-%d}; "
                             "the mapping matches more than one trading class")
    return months


def front_contract(months: list[ContractMonth], today: pd.Timestamp, *, min_days: int = 7) -> ContractMonth:
    """Nearest month expiring at least ``min_days`` calendar days from today."""
    for month in months:
        if month.expiry >= today + pd.Timedelta(days=min_days):
            return month
    raise ValueError("no listed contract expires far enough in the future")
