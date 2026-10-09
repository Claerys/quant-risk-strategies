"""Load ticker definitions from a CSV file, as the BSQF ibkr_base project did.

A row is a *discovery input*, not a qualified contract: it says what to ask IBKR about, it does not
assert what the contract is. Nothing is downloaded against a row until IBKR has resolved it to
exactly one contract. When a symbol is ambiguous (two listings), put IBKR's ``con_id`` in the file
and it is used as the identity.

Columns: ``symbol`` (required); ``sec_type`` (STK), ``exchange`` (SMART), ``currency`` (USD),
``primary_exchange``, ``con_id`` (all optional). Futures are not listed here: they come from the
instrument universe (see ``contracts.py``).

Adapted from the BSQF ibkr_base project, with its authors' permission.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from quant_risk.ibkr.contract import ContractKey


@dataclass(frozen=True)
class TickerDefinition:
    symbol: str
    sec_type: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    primary_exchange: str = ""
    con_id: int = 0  # 0 until IBKR confirms it

    def to_key(self) -> ContractKey:
        return ContractKey(symbol=self.symbol, sec_type=self.sec_type, exchange=self.exchange,
                           currency=self.currency, primary_exchange=self.primary_exchange, con_id=self.con_id)


def _clean(value: object, default: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def _row_to_ticker(row: dict[str, str], row_number: int) -> TickerDefinition:
    symbol = _clean(row.get("symbol")).upper()
    if not symbol:
        raise ValueError(f"Ticker CSV row {row_number} has an empty symbol.")
    sec_type = _clean(row.get("sec_type"), "STK").upper()
    if sec_type != "STK":
        raise ValueError(f"Ticker CSV row {row_number}: {sec_type} {symbol} is not supported here; "
                         "only STK rows are read from a ticker file (futures come from instruments.csv).")
    text = _clean(row.get("con_id"), "0")
    try:
        con_id = int(float(text))
    except ValueError:
        raise ValueError(f"Ticker CSV row {row_number}: con_id must be a whole number, got {text!r}.") from None
    if con_id < 0:
        raise ValueError(f"Ticker CSV row {row_number}: con_id cannot be negative ({con_id}).")
    return TickerDefinition(symbol=symbol, sec_type=sec_type, exchange=_clean(row.get("exchange"), "SMART").upper(),
                            currency=_clean(row.get("currency"), "USD").upper(),
                            primary_exchange=_clean(row.get("primary_exchange")).upper(), con_id=con_id)


def load_tickers(path: Path | str) -> list[TickerDefinition]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ticker file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "symbol" not in reader.fieldnames:
            raise ValueError(f"{path} needs a 'symbol' column")
        tickers = [_row_to_ticker(row, number) for number, row in enumerate(reader, start=2)]
    seen: set[tuple] = set()
    for ticker in tickers:
        if ticker.to_key().identity in seen:
            raise ValueError(f"{path}: {ticker.symbol} is listed twice")
        seen.add(ticker.to_key().identity)
    if not tickers:
        raise ValueError(f"{path} contains no tickers")
    return tickers
