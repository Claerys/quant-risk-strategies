"""Command line for downloading futures and EUR/USD from IB Gateway.

    python scripts/download_ibkr.py --dry-run                  # qualify every contract, write nothing
    python scripts/download_ibkr.py --symbols ES CL --years 10
    python scripts/download_ibkr.py --fx
    python scripts/download_ibkr.py --tickers-file config/tickers_nasdaq100.csv --years 10
    python scripts/download_ibkr.py --tickers-file config/tickers_sp500_seed.csv --symbols AAPL MSFT

Futures come from the instrument universe; stocks come from a ticker file (one row per ticker).
A ticker row that IBKR will not resolve to exactly one contract is skipped and reported.

IB Gateway (paper, port 4002) or TWS (paper, port 7497) must be running with the API socket
enabled. No username or password is needed here: the login lives in Gateway.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from quant_risk.bars import data_dir
from quant_risk.ibkr.config import IbkrSettings
from quant_risk.ibkr.contracts import front_contract, futures_chain
from quant_risk.ibkr.download import download_fx, download_symbol
from quant_risk.ibkr.qualification import QualificationError
from quant_risk.ibkr.stocks import download_stock, stocks_dir
from quant_risk.ibkr.tickers import load_tickers
from quant_risk.instruments import load_instruments
from quant_risk.quality import summarize

# what a single symbol can fail with; one symbol failing must not stop the others
SYMBOL_ERRORS = (RuntimeError, ValueError, QualificationError, TimeoutError)


def default_client_factory():
    from quant_risk.ibkr.client import IbkrClient  # imported late: needs the optional ibapi package

    return IbkrClient()


def main(argv: list[str] | None = None, client_factory: Callable[[], object] = default_client_factory) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="*", default=None, help="symbols to do (default: all in the universe or file)")
    parser.add_argument("--tickers-file", type=Path, default=None,
                        help="download the STK rows of this CSV (see config/) instead of the futures universe")
    parser.add_argument("--years", type=int, default=20, help="years of history for a first download")
    parser.add_argument("--fx", action="store_true", help="also download daily EUR/USD")
    parser.add_argument("--dry-run", action="store_true", help="qualify contracts and print them; write nothing")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"destination (default {data_dir()} for futures, {stocks_dir()} for stocks)")
    args = parser.parse_args(argv)

    universe = load_instruments()
    tickers = None
    if args.tickers_file is not None:
        try:
            tickers = load_tickers(args.tickers_file)
        except (OSError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 1
        known = {t.symbol: t for t in tickers}
        symbols = args.symbols if args.symbols else list(known)
        unknown = [s for s in symbols if s not in known]
        source = str(args.tickers_file)
    else:
        symbols = args.symbols if args.symbols else list(universe)
        unknown = [s for s in symbols if s not in universe]
        source = "src/quant_risk/instruments.csv"
    if unknown:
        print(f"unknown symbols {unknown}; see {source}", file=sys.stderr)
        return 1
    try:
        settings = IbkrSettings.from_env()
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    client = client_factory()
    try:
        client.connect_and_start(settings)
    except (TimeoutError, OSError, ConnectionError) as exc:
        print(f"IBKR connection failed: {exc}", file=sys.stderr)
        return 1

    failures, skipped, issues, rows = [], [], [], []
    try:
        for symbol in symbols:
            try:
                if tickers is not None:
                    ticker = known[symbol]
                    if args.dry_run:
                        found = client.qualify(ticker.to_key())
                        rows.append((symbol, f"conId {found.con_id}", found.long_name, ticker.primary_exchange, ""))
                        continue
                    result = download_stock(client, ticker, years=args.years, directory=args.out)
                    issues += result.issues
                    rows.append((symbol, result.action, str(result.bars), f"+{result.added}", result.reason))
                    continue
                instrument = universe[symbol]
                if args.dry_run:
                    months = futures_chain(client, instrument)
                    front = front_contract(months, pd.Timestamp.today().normalize())
                    rows.append((symbol, f"{len(months)} months", f"{months[0].expiry:%Y-%m-%d}",
                                 f"{months[-1].expiry:%Y-%m-%d}", front.local_symbol))
                    continue
                result = download_symbol(client, instrument, years=args.years, directory=args.out)
                issues += result.issues
                rows.append((symbol, result.mode, str(result.bars), f"+{result.added}", f"{result.months} contracts"))
            except SYMBOL_ERRORS as exc:
                if tickers is not None and isinstance(exc, QualificationError):
                    skipped.append((symbol, str(exc)))  # a ticker row IBKR will not resolve: skip and report
                    print(f"SKIPPED {symbol}: {exc}", file=sys.stderr)
                else:
                    failures.append((symbol, str(exc)))
                    print(f"FAILED {symbol}: {exc}", file=sys.stderr)
        if args.fx and not args.dry_run:
            try:
                rows.append(("EURUSD", "fx", str(download_fx(client, years=args.years)), "", ""))
            except SYMBOL_ERRORS as exc:
                failures.append(("EURUSD", str(exc)))
                print(f"FAILED EURUSD: {exc}", file=sys.stderr)
    finally:
        client.disconnect_and_stop()

    if tickers is not None:
        header = ["symbol", "qualified", "name", "listing", ""] if args.dry_run else \
            ["symbol", "action", "bars", "added", "note"]
    else:
        header = ["symbol", "months", "first expiry", "last expiry", "front"] if args.dry_run else \
            ["symbol", "mode", "bars", "added", "contracts"]
    print(pd.DataFrame(rows, columns=header).to_string(index=False) if rows else "nothing done")
    if issues:
        print("\ndata-quality summary")
        print(summarize(issues).to_string(index=False))
    if skipped:
        print(f"\n{len(skipped)} skipped (IBKR would not resolve them to exactly one contract): "
              f"{[s for s, _ in skipped]}", file=sys.stderr)
    if failures:
        print(f"\n{len(failures)} failed: {[s for s, _ in failures]}", file=sys.stderr)
        return 1
    return 0
