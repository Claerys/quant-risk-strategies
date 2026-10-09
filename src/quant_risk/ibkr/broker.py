"""Route the paper-trading cycle's orders to an Interactive Brokers *paper* account.

``IbkrBroker`` implements the same two-method ``Broker`` interface as the simulated ``PaperBroker``,
so ``cycle.run_cycle`` and its pre-trade limits are unchanged: the cycle still decides what the book
may hold, the broker only carries out the approved orders.

Safety, in layers:
  * the port must be a paper port and the account id must start with ``DU`` (IBKR's paper prefix);
  * every order's front-month contract is qualified before the first order is sent, so a lookup
    failure sends nothing at all;
  * only the quantity IBKR reports as filled is returned, never the quantity that was asked for.

This is verified against a scripted fake gateway. It has not been run against a live paper session.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_risk.ibkr.config import IbkrSettings, require_paper_port
from quant_risk.ibkr.contract import ContractKey
from quant_risk.ibkr.contracts import (
    ContractMonth,
    front_contract,
    futures_chain,
    futures_map_for,
    load_futures_map,
)
from quant_risk.instruments import Instrument, load_instruments
from quant_risk.paper import VAR_DIR, Fill

PAPER_ACCOUNT_PREFIX = "DU"


class IbkrBroker:
    def __init__(self, client, settings: IbkrSettings, *, instruments: dict[str, Instrument] | None = None,
                 state_path: Path = VAR_DIR / "ibkr_state.json", cost_per_contract: float = 0.0) -> None:
        require_paper_port(settings.port)
        account = settings.account or ""
        if not account.startswith(PAPER_ACCOUNT_PREFIX):
            raise ValueError(f"account {account!r} is not a paper account (paper ids start with "
                             f"{PAPER_ACCOUNT_PREFIX}); set IBKR_ACCOUNT")
        known = list(getattr(client, "accounts", []) or [])
        if known and account not in known:
            raise ValueError(f"account {account} is not among the connected accounts {known}")
        self.client, self.account = client, account
        self.instruments = instruments or load_instruments()
        self.state_path, self.cost_per_contract = state_path, cost_per_contract
        self.ignored: list[str] = []  # held contracts outside the universe, left alone

    @property
    def last_date(self) -> pd.Timestamp | None:
        if not self.state_path.exists():
            return None
        return pd.Timestamp(json.loads(self.state_path.read_text())["last_date"])

    def _symbol_of(self, key: ContractKey) -> str | None:
        for symbol, mapping in load_futures_map().items():
            if (key.sec_type == "FUT" and key.symbol == mapping.ibkr_symbol
                    and key.trading_class in ("", mapping.trading_class)):
                return symbol
        return None

    def positions(self) -> pd.Series:
        held: dict[str, float] = {}
        self.ignored = []
        for key, quantity in self.client.positions().items():
            symbol = self._symbol_of(key)
            if symbol is None:
                self.ignored.append(key.label)
                continue
            held[symbol] = held.get(symbol, 0.0) + quantity
        return pd.Series(held, dtype=float)

    def _front(self, symbol: str, today: pd.Timestamp) -> ContractMonth:
        months = futures_chain(self.client, self.instruments[symbol], include_expired=False)
        return front_contract(months, today)

    def execute(self, orders: pd.Series, prices: pd.Series, date: pd.Timestamp) -> list[Fill]:
        wanted = {s: float(q) for s, q in orders.items() if q != 0}
        for symbol, quantity in wanted.items():
            if abs(quantity - round(quantity)) > 1e-9:
                raise ValueError(f"{symbol}: order of {quantity} contracts is not a whole number")
        # Qualify everything first: one failed lookup must mean no orders at all.
        fronts = {symbol: self._front(symbol, pd.Timestamp(date)) for symbol in wanted}

        fills: list[Fill] = []
        for symbol, quantity in wanted.items():
            result = self.client.place_market_order(fronts[symbol].key, round(quantity), self.account)
            if result.filled <= 0:
                continue
            signed = result.filled if quantity > 0 else -result.filled
            price = result.avg_price * futures_map_for(self.instruments[symbol]).price_scale
            fills.append(Fill(symbol, signed, price, result.filled * self.cost_per_contract))
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({"last_date": pd.Timestamp(date).strftime("%Y-%m-%d")}))
        return fills


def connect_ibkr_broker(client_factory, settings: IbkrSettings | None = None) -> tuple[object, IbkrBroker]:
    """Connect to a paper Gateway/TWS and return (client, broker); the caller disconnects the client."""
    settings = settings or IbkrSettings.from_env()
    client = client_factory()
    client.connect_and_start(settings)
    try:
        return client, IbkrBroker(client, settings)
    except Exception:
        client.disconnect_and_stop()
        raise
