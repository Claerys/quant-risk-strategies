"""Paper trading: a simulated broker and an audit journal.

**PaperBroker** fills orders at the given price and keeps positions in var/paper_state.json, so
a cycle can run day after day and pick up where it left off. It implements the same small
interface (`positions()`, `execute()`) a real broker adapter would, so an Interactive Brokers
connection can replace it without touching the trading cycle.

**Journal** is an append-only SQLite database (var/journal.sqlite) recording, for every cycle,
what was proposed, what the risk engine changed and why, every limit's utilisation, the VaR,
and the fills. It is the audit trail: after the fact anyone can answer "why did we hold this?".
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from quant_risk.bars import REPO_ROOT

VAR_DIR = REPO_ROOT / "var"


@dataclass(frozen=True)
class Fill:
    symbol: str
    contracts: float  # signed: + bought, - sold
    price: float
    cost: float  # USD


class Broker(Protocol):
    def positions(self) -> pd.Series: ...

    def execute(self, orders: pd.Series, prices: pd.Series, date: pd.Timestamp) -> list[Fill]: ...


class PaperBroker:
    """Fills every order in full at the supplied price; charges a flat cost per contract."""

    def __init__(self, path: Path = VAR_DIR / "paper_state.json", cost_per_contract: float = 3.0) -> None:
        self.path = path
        self.cost_per_contract = cost_per_contract
        self._state = json.loads(path.read_text()) if path.exists() else {"positions": {}, "last_date": None}

    @property
    def last_date(self) -> pd.Timestamp | None:
        return pd.Timestamp(self._state["last_date"]) if self._state["last_date"] else None

    def positions(self) -> pd.Series:
        return pd.Series(self._state["positions"], dtype=float)

    def execute(self, orders: pd.Series, prices: pd.Series, date: pd.Timestamp) -> list[Fill]:
        fills = []
        held = self._state["positions"]
        for symbol, contracts in orders.items():
            if contracts == 0:
                continue
            fills.append(Fill(symbol, float(contracts), float(prices[symbol]), abs(contracts) * self.cost_per_contract))
            held[symbol] = held.get(symbol, 0.0) + float(contracts)
            if held[symbol] == 0:
                del held[symbol]
        self._state["last_date"] = date.strftime("%Y-%m-%d")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2, sort_keys=True))
        return fills


SCHEMA = """
CREATE TABLE IF NOT EXISTS cycles (
    date TEXT, strategy TEXT, status TEXT, daily_pnl REAL, equity REAL, drawdown REAL,
    gross_exposure REAL, var_99 REAL, es_99 REAL, orders INTEGER, halted INTEGER, note TEXT
);
CREATE TABLE IF NOT EXISTS actions (date TEXT, strategy TEXT, action TEXT);
CREATE TABLE IF NOT EXISTS limit_checks (
    date TEXT, strategy TEXT, name TEXT, value REAL, limit_value REAL, utilisation REAL
);
CREATE TABLE IF NOT EXISTS fills (date TEXT, strategy TEXT, symbol TEXT, contracts REAL, price REAL, cost REAL);
CREATE TABLE IF NOT EXISTS positions (date TEXT, strategy TEXT, symbol TEXT, contracts REAL);
"""


class Journal:
    def __init__(self, path: Path = VAR_DIR / "journal.sqlite") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def record(self, table: str, rows: list[dict]) -> None:
        if not rows:
            return
        columns = list(rows[0])
        placeholders = ", ".join("?" for _ in columns)
        self.db.executemany(
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            [tuple(r[c] for c in columns) for r in rows],
        )
        self.db.commit()

    def read(self, table: str, strategy: str | None = None) -> pd.DataFrame:
        query = f"SELECT * FROM {table}" + (" WHERE strategy = ?" if strategy else "")
        return pd.read_sql_query(query, self.db, params=(strategy,) if strategy else None)

    def equity_history(self, strategy: str) -> pd.Series:
        cycles = self.read("cycles", strategy)
        return pd.Series(cycles["equity"].to_numpy(), index=pd.to_datetime(cycles["date"]))

    def close(self) -> None:
        self.db.close()
