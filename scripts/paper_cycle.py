"""Run the daily paper-trading cycle, for one day or replayed over a period.

    python scripts/paper_cycle.py                                  # one cycle on the latest data
    python scripts/paper_cycle.py --as-of 2025-04-02               # one cycle on a given day
    python scripts/paper_cycle.py --replay 2025-03-03 2025-05-30   # every trading day in a period
    python scripts/paper_cycle.py --replay 2025-03-03 2025-05-30 --reset --quiet

State (positions) is kept in var/paper_state.json and every cycle is recorded in
var/journal.sqlite, both git-ignored. --reset starts from a flat book and an empty journal.
Risk limits are read from config/limits.toml and the run refuses to start if any is missing.
"""

from __future__ import annotations

import argparse

import pandas as pd

from quant_risk.bars import available_symbols, load_closes
from quant_risk.cycle import run_cycle
from quant_risk.limits import LimitsNotConfigured, RiskLimits
from quant_risk.paper import VAR_DIR, Journal, PaperBroker
from quant_risk.strategies import STRATEGIES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="time_series_momentum", choices=list(STRATEGIES))
    parser.add_argument("--capital", type=float, default=10_000_000)
    when = parser.add_mutually_exclusive_group()
    when.add_argument("--as-of", help="run one cycle after this day's close (default: latest data)")
    when.add_argument("--replay", nargs=2, metavar=("START", "END"), help="run a cycle for every day in range")
    parser.add_argument("--reset", action="store_true", help="start from a flat book and empty journal")
    parser.add_argument("--quiet", action="store_true", help="print one line per day instead of full alerts")
    args = parser.parse_args(argv)

    try:
        limits = RiskLimits.from_file()
    except LimitsNotConfigured as exc:
        print(exc)
        return 2

    if args.reset:
        for name in ("paper_state.json", "journal.sqlite"):
            (VAR_DIR / name).unlink(missing_ok=True)

    closes = load_closes(available_symbols())
    if args.replay:
        days = closes.loc[args.replay[0] : args.replay[1]].index
    else:
        days = [pd.Timestamp(args.as_of) if args.as_of else closes.index[-1]]

    broker, journal = PaperBroker(), Journal()
    try:
        for day in days:
            if broker.last_date is not None and day <= broker.last_date:
                print(f"skip {day:%Y-%m-%d}: book already processed up to {broker.last_date:%Y-%m-%d}")
                continue
            result = run_cycle(day, closes, strategy=args.strategy, capital=args.capital, limits=limits,
                               broker=broker, journal=journal, send_alert=not args.quiet)
            if args.quiet:
                print(f"{result.date:%Y-%m-%d} {result.status:8} P&L {result.daily_pnl:+12,.0f}  "
                      f"DD {result.drawdown:6.1%}  VaR99 {result.var_99:10,.0f}  orders {int((result.orders != 0).sum()):3}"
                      + (f"  | {result.actions[0]}" if result.actions else ""))
    finally:
        journal.close()
    print(f"\njournal: {journal.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
