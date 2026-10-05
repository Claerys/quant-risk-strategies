"""Print the daily risk report for one strategy's current book.

    python scripts/risk_report.py
    python scripts/risk_report.py --strategy multi_horizon_trend --as-of 2025-04-08

The same numbers as the dashboard (streamlit run app/dashboard.py), as plain text.
"""

from __future__ import annotations

import argparse

import pandas as pd

from quant_risk.report import build_report
from quant_risk.strategies import STRATEGIES


def money(v: float) -> str:
    return "n/a" if pd.isna(v) else f"{v:,.0f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="time_series_momentum", choices=list(STRATEGIES))
    parser.add_argument("--capital", type=float, default=10_000_000)
    parser.add_argument("--as-of", default=None, help="report as of this date (default: latest data)")
    args = parser.parse_args(argv)

    r = build_report(args.strategy, args.capital, args.as_of)
    h = r.headline
    rule = "-" * 78
    with pd.option_context("display.width", 140, "display.max_columns", 12, "display.max_colwidth", 50):
        print(f"\nDAILY RISK REPORT  {r.as_of:%d %b %Y}  {args.strategy}  capital {money(args.capital)} USD\n{rule}")
        print(f"gross {h['gross exposure'] / args.capital:.2f}x  net {h['net exposure'] / args.capital:+.2f}x  "
              f"VaR99 {money(h['VaR 99% (1d)'])}  ES97.5 {money(h['ES 97.5% (1d)'])}")
        print(f"worst stress: {h['worst stress']} {money(h['worst stress P&L'])}  "
              f"highest limit use: {h['highest limit use']} {h['highest limit use %']:.0%}")

        print(f"\nLIMITS\n{rule}")
        limits = r.limits.copy().astype(object)
        for name, row in r.limits.iterrows():
            fmt = (lambda v: f"{v:.1%}") if name == "drawdown" else money  # drawdown is a fraction of capital
            limits.loc[name] = [fmt(row["value"]), fmt(row["limit value"]), f"{row['utilisation']:.0%}"]
        print(limits.to_string())

        print(f"\nVaR AND ES (1-day, USD)\n{rule}")
        print(r.var.map(money).to_string())

        print(f"\nTOP RISK CONTRIBUTORS (99% parametric, Euler)\n{rule}")
        top = r.contributions[r.contributions["contracts"] != 0].head(8)
        print(top[["sector", "contracts", "standalone VaR", "component VaR", "share of VaR"]].to_string(
            formatters={"standalone VaR": money, "component VaR": money, "share of VaR": "{:.0%}".format}))

        print(f"\nEXPOSURE BY SECTOR (USD notional)\n{rule}")
        print(r.exposure_by_sector.map(money).to_string())

        print(f"\nSTRESS SCENARIOS\n{rule}")
        stress = r.stress[["category", "P&L", "% of capital", "biggest losses"]].sort_values("P&L")
        print(stress.to_string(formatters={"P&L": money, "% of capital": "{:.1%}".format}))

        print(f"\nREVERSE STRESS: worst historical windows for this book\n{rule}")
        print(r.reverse_stress.to_string(formatters={"worst P&L": money}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
