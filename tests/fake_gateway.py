"""Scripted stand-ins for IB Gateway. Nothing here opens a socket or can send a real order.

Test doubles adapted from the BSQF ibkr_base project, with its authors' permission.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pandas as pd

from quant_risk.ibkr.contract import ContractKey, OrderResult, QualifiedContract
from quant_risk.ibkr.qualification import QualificationError


def make_contract(symbol="ES", con_id=1, sec_type="FUT", exchange="CME", currency="USD", expiry="20250620",
                  multiplier="50", local_symbol="ESM5", trading_class="ES") -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, conId=con_id, secType=sec_type, exchange=exchange, currency=currency,
                           lastTradeDateOrContractMonth=expiry, multiplier=multiplier,
                           tradingClass=trading_class, localSymbol=local_symbol)


def make_details(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(contract=make_contract(**kwargs), minTick=0.25, timeZoneId="US/Central",
                           longName="E-mini S&P 500")


def make_bars(start: str, days: int, base: float = 100.0, volume: float = 1000.0) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=days, name="date")
    close = pd.Series([base + i for i in range(days)], index=index, dtype=float)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close,
                         "volume": volume}, index=index)


Script = Callable[["FakeTransport", int, Any], None]


@dataclass
class FakeTransport:
    """Drives a ContractDetailsCollector with a scripted callback sequence."""

    collector: Any
    script: Script | None = None
    connected: bool = True
    next_id: int = 1000
    sent: list = field(default_factory=list)
    cancelled: list = field(default_factory=list)
    send_error: Exception | None = None

    def allocate_request_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def request_contract_details(self, req_id: int, contract: Any) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append((req_id, contract))
        if self.script is not None:
            self.script(self, req_id, contract)

    def cancel_contract_details(self, req_id: int) -> None:
        self.cancelled.append(req_id)

    def is_connected(self) -> bool:
        return self.connected


@dataclass
class FakeClient:
    """Same public surface as IbkrClient, answering from dictionaries.

    ``months``: root symbol -> list of (local_symbol, expiry YYYYMMDD, con_id);
    ``bars``: local_symbol -> DataFrame (served as-is, regardless of window);
    ``held``: local_symbol -> contracts; ``fills``: local_symbol -> contracts filled.
    """

    months: dict[str, list[tuple[str, str, int]]] = field(default_factory=dict)
    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    held: dict[str, float] = field(default_factory=dict)
    fills: dict[str, float] = field(default_factory=dict)
    fill_price: float = 100.0
    multiplier: str = "50"
    multipliers: dict[str, str] = field(default_factory=dict)  # per IBKR root, overrides `multiplier`
    accounts: list[str] = field(default_factory=lambda: ["DU1234567"])
    calls: list = field(default_factory=list)
    orders: list = field(default_factory=list)
    bars_fn: Callable[..., pd.DataFrame] | None = None  # (key, end, duration) -> bars; overrides `bars`
    connect_error: Exception | None = None
    connected_with: Any = None
    disconnected: bool = False

    def connect_and_start(self, settings) -> None:
        if self.connect_error is not None:
            raise self.connect_error
        self.connected_with = settings

    def disconnect_and_stop(self) -> None:
        self.disconnected = True

    def candidates(self, key: ContractKey) -> tuple[QualifiedContract, ...]:
        self.calls.append(("candidates", key.symbol, key.include_expired))
        found = self.months.get(key.symbol, [])
        if not found:
            raise QualificationError(f"{key.label}: no IBKR contract matches this description")
        return tuple(
            QualifiedContract.from_contract_details(make_details(
                symbol=key.symbol, con_id=con_id, sec_type=key.sec_type, exchange=key.exchange,
                currency=key.currency, expiry=expiry, local_symbol=local, multiplier=self.multipliers.get(key.symbol, self.multiplier)))
            for local, expiry, con_id in found)

    def qualify(self, key: ContractKey) -> QualifiedContract:
        found = self.candidates(key)
        if len(found) != 1:
            raise QualificationError(f"{key.label}: {len(found)} contracts match; set con_id to choose one")
        return found[0]

    def historical_bars(self, key: ContractKey, end: str, duration: str, what: str = "TRADES",
                        bar_size: str = "1 day", use_rth: bool = True) -> pd.DataFrame:
        self.calls.append(("bars", key.local_symbol or key.symbol, end, what, duration))
        if self.bars_fn is not None:
            return self.bars_fn(key, end, duration)
        return self.bars.get(key.local_symbol or key.symbol, pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]))

    def positions(self) -> dict[ContractKey, float]:
        def root(local: str) -> str:
            return next((r for r, found in self.months.items() if any(m[0] == local for m in found)), local)
        def is_stock(local: str) -> bool:
            return any(m[0] == local and m[1] == "" for found in self.months.values() for m in found)
        return {ContractKey(symbol=root(local), sec_type="STK" if is_stock(local) else "FUT",
                            trading_class="" if is_stock(local) else root(local), local_symbol=local,
                            con_id=i + 1): qty for i, (local, qty) in enumerate(self.held.items()) if qty}

    def place_market_order(self, key: ContractKey, quantity: float, account: str) -> OrderResult:
        self.orders.append((key.local_symbol, quantity, account))
        filled = self.fills.get(key.local_symbol, abs(quantity))
        self.held[key.local_symbol] = self.held.get(key.local_symbol, 0.0) + (filled if quantity > 0 else -filled)
        return OrderResult(filled=filled, avg_price=self.fill_price, status="Filled")
