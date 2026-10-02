"""Backtest every strategy on the stored data and compare them.

    python scripts/run_backtest.py
    python scripts/run_backtest.py --capital 10000000 --target-vol 0.15 --start 2003

Prints one row per strategy and writes daily returns and positions to reports/ (git-ignored).
"Predicted vol" is the average ex-ante volatility the sizing expected; "realised vol" is what the
backtest actually produced. A risk function watches that ratio: well below 1 means the model is
conservative, above 1 means it under-estimates risk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quant_risk.backtest import run_backtest
from quant_risk.bars import REPO_ROOT, available_symbols, load_closes
from quant_risk.metrics import TRADING_DAYS, performance_summary
from quant_risk.sizing import size_positions
from quant_risk.strategies import STRATEGIES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategies", nargs="*", default=list(STRATEGIES), choices=list(STRATEGIES))
    parser.add_argument("--capital", type=float, default=10_000_000)
    parser.add_argument("--target-vol", type=float, default=0.15)
    parser.add_argument("--cost", type=float, default=3.0, help="USD per contract traded")
    parser.add_argument("--start", default="2003", help="first date used (history before it warms up signals)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports")
    args = parser.parse_args(argv)

    closes = load_closes(available_symbols())
    args.out.mkdir(parents=True, exist_ok=True)
    rows, returns = {}, {}
    for name in args.strategies:
        signals = STRATEGIES[name](closes)
        sized = size_positions(signals, closes, capital=args.capital, target_vol=args.target_vol)
        result = run_backtest(sized.positions, closes, capital=args.capital, cost_per_contract=args.cost)

        window = result.returns.loc[args.start:]
        stats = performance_summary(window)
        predicted = (sized.ex_ante_vol.loc[args.start:] * np.sqrt(TRADING_DAYS) / args.capital).mean()
        stats["predicted_vol"] = predicted
        stats["realised_vs_predicted"] = stats["annual_volatility"] / predicted
        stats["costs_per_year"] = result.costs.sum(axis=1).loc[args.start:].sum() / len(window) * TRADING_DAYS
        rows[name] = stats
        returns[name] = window
        sized.positions.loc[args.start:].to_csv(args.out / f"positions_{name}.csv")

    pd.DataFrame(returns).to_csv(args.out / "daily_returns.csv")
    table = pd.DataFrame(rows).T
    percent = ["annual_return", "annual_volatility", "predicted_vol", "max_drawdown", "worst_day", "best_day"]
    shown = table.copy()
    shown[percent] = shown[percent].map(lambda v: f"{v:.1%}")
    shown["costs_per_year"] = table["costs_per_year"].map(lambda v: f"${v:,.0f}")
    for col in ["sharpe", "sortino", "calmar", "skew", "hit_rate", "realised_vs_predicted"]:
        shown[col] = table[col].map(lambda v: f"{v:.2f}")

    print(f"\n{args.start} onwards, capital ${args.capital:,.0f}, target volatility {args.target_vol:.0%}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(shown[["annual_return", "annual_volatility", "predicted_vol", "realised_vs_predicted", "sharpe",
                     "sortino", "max_drawdown", "calmar", "skew", "worst_day", "costs_per_year"]])
    print(f"\ndaily returns and positions written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
