"""Everything a daily risk report needs, computed in one place.

The dashboard (app/dashboard.py) and the text report (scripts/risk_report.py) only display what
this module returns, so every number on screen comes from tested code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_risk.backtest import BacktestResult, run_backtest
from quant_risk.bars import available_symbols, load_closes, pnl_per_contract
from quant_risk.fx import to_usd
from quant_risk.instruments import get_instrument
from quant_risk.limits import RiskLimits, contract_value_frame, review
from quant_risk.metrics import drawdown, performance_summary
from quant_risk.sizing import SizingResult, size_positions
from quant_risk.strategies import STRATEGIES
from quant_risk.stress import reverse_stress, sector_shock_grid, stress_table
from quant_risk.var import filtered_scenarios, historical_var_es, risk_contributions, rolling_var, var_table
from quant_risk.var_backtest import backtest_var, rolling_basel_zone


@dataclass
class RiskReport:
    as_of: pd.Timestamp
    strategy: str
    capital: float
    positions: pd.DataFrame  # one row per held instrument
    exposure_by_sector: pd.DataFrame
    var: pd.DataFrame
    contributions: pd.DataFrame
    limits: pd.DataFrame
    stress: pd.DataFrame
    reverse_stress: pd.DataFrame
    sector_grid: pd.DataFrame
    performance: dict
    equity: pd.Series
    drawdown: pd.Series

    @property
    def headline(self) -> dict:
        var99 = self.var.loc["filtered historical", "VaR 99.0%"]
        es975 = self.var.loc["filtered historical", "ES 97.5%"]
        worst = self.stress["P&L"].idxmin()
        return {
            "gross exposure": self.positions["notional"].abs().sum(),
            "net exposure": self.positions["notional"].sum(),
            "VaR 99% (1d)": var99,
            "ES 97.5% (1d)": es975,
            "worst stress": worst,
            "worst stress P&L": self.stress.loc[worst, "P&L"],
            "highest limit use": self.limits["utilisation"].idxmax(),
            "highest limit use %": self.limits["utilisation"].max(),
        }


def load_market() -> tuple[pd.DataFrame, pd.DataFrame]:
    closes = load_closes(available_symbols())
    pnl = to_usd(pnl_per_contract(closes.ffill())).reindex(closes.index).fillna(0.0)
    return closes, pnl


def run_strategy(strategy: str, closes: pd.DataFrame, capital: float) -> tuple[SizingResult, BacktestResult]:
    sized = size_positions(STRATEGIES[strategy](closes), closes, capital=capital)
    return sized, run_backtest(sized.positions, closes, capital=capital)


def build_report(
    strategy: str = "time_series_momentum",
    capital: float = 10_000_000,
    as_of: str | pd.Timestamp | None = None,
    *,
    limits: RiskLimits | None = None,
    start: str = "2003",
) -> RiskReport:
    closes, pnl = load_market()
    if as_of is not None:
        closes, pnl = closes.loc[:as_of], pnl.loc[:as_of]
    limits = limits or RiskLimits.from_file()
    sized, result = run_strategy(strategy, closes, capital)
    book = sized.positions.iloc[-1]
    held = book[book != 0]

    values = contract_value_frame(closes).iloc[-1]
    positions = pd.DataFrame({
        "name": [get_instrument(s).name for s in held.index],
        "sector": [get_instrument(s).sector for s in held.index],
        "contracts": held,
        "price": closes.ffill().iloc[-1][held.index],
        "notional": held * values[held.index],
    }, index=held.index)
    positions["% of capital"] = positions["notional"] / capital

    exposure = positions.groupby("sector").agg(
        long=("notional", lambda v: v[v > 0].sum()),
        short=("notional", lambda v: v[v < 0].sum()),
        net=("notional", "sum"),
        gross=("notional", lambda v: v.abs().sum()),
    ).sort_values("gross", ascending=False)

    contributions = risk_contributions(held, pnl)
    var99 = historical_var_es(filtered_scenarios(held, pnl), 0.99)[0]

    daily = result.net_pnl
    dd = drawdown(result.returns)
    checks = review(held, held, values.replace(0, np.nan).fillna(1.0), limits=limits, capital=capital,
                    daily_pnl=float(daily.iloc[-1]), drawdown=float(dd.iloc[-1])).checks
    limit_rows = [{"limit": c.name, "value": c.value, "limit value": c.limit, "utilisation": c.utilisation}
                  for c in checks if c.name != "largest order"]
    limit_rows.append({"limit": "VaR 99%", "value": var99, "limit value": limits.max_var_99,
                       "utilisation": var99 / limits.max_var_99})

    window = result.returns.loc[start:]
    return RiskReport(
        as_of=closes.index[-1],
        strategy=strategy,
        capital=capital,
        positions=positions.sort_values("notional", key=abs, ascending=False),
        exposure_by_sector=exposure,
        var=var_table(held, pnl),
        contributions=contributions,
        limits=pd.DataFrame(limit_rows).set_index("limit"),
        stress=stress_table(held, closes, capital=capital),
        reverse_stress=reverse_stress(held, pnl),
        sector_grid=sector_shock_grid(held, closes),
        performance=performance_summary(window),
        equity=result.equity.loc[start:],
        drawdown=dd.loc[start:],
    )


def strategy_comparison(capital: float = 10_000_000, start: str = "2003") -> pd.DataFrame:
    closes, _ = load_market()
    rows = {}
    for name in STRATEGIES:
        _, result = run_strategy(name, closes, capital)
        rows[name] = performance_summary(result.returns.loc[start:])
    return pd.DataFrame(rows).T


def var_backtest_report(
    strategy: str = "time_series_momentum",
    capital: float = 10_000_000,
    start: str = "2004",
    methods: tuple[str, ...] = ("historical", "filtered", "parametric"),
    level: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test summary per VaR method, and a daily frame of P&L, VaR and Basel zone for charting."""
    closes, pnl = load_market()
    sized, _ = run_strategy(strategy, closes, capital)
    realised = run_backtest(sized.positions, closes, capital=capital, cost_per_contract=0.0).pnl.sum(axis=1)
    summary, series = {}, {"P&L": realised.loc[start:]}
    for method in methods:
        forecast = rolling_var(sized.positions, pnl, level=level, method=method).loc[start:]
        test = backtest_var(forecast, realised.loc[start:], level)
        summary[method] = test.summary()
        series[f"{method} VaR"] = forecast.shift(1)
        series[f"{method} zone"] = rolling_basel_zone(test.hits.dropna().astype(bool))
    return pd.DataFrame(summary).T, pd.DataFrame(series)
