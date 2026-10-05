"""One daily trading cycle: data -> signal -> size -> risk checks -> orders -> journal -> alert.

Runs after the close of `as_of`, using only data up to that close:

1. **Data checks.** Any instrument whose latest bar is older than `max_bar_age_days` is frozen:
   its position is held, no order is sent, and the alert says so. Trading on a stale price is how
   a feed outage turns into a loss.
2. **Mark to market.** Yesterday's positions are valued at today's close to get the day's P&L,
   equity and drawdown, which the hard loss limits need.
3. **Target.** The strategy's signal is sized to the volatility target.
4. **Pre-trade review.** The limits engine approves, shrinks or blocks the move (see limits.py).
5. **VaR limit.** The 99% filtered historical VaR of the approved book is computed; if it is over
   the limit the book is scaled down and reviewed again.
6. **Execution, journal, alert.** Orders go to the broker, everything is journalled, and a
   summary is sent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_risk import alerts
from quant_risk.bars import pnl_per_contract
from quant_risk.fx import to_usd
from quant_risk.limits import RiskLimits, contract_value_frame, review
from quant_risk.paper import Broker, Journal
from quant_risk.sizing import size_positions
from quant_risk.strategies import STRATEGIES
from quant_risk.var import filtered_scenarios, historical_var_es


@dataclass
class CycleResult:
    date: pd.Timestamp
    status: str
    approved: pd.Series
    orders: pd.Series
    var_99: float
    es_99: float
    daily_pnl: float
    drawdown: float
    actions: list[str]
    message: str


def _var_es(book: pd.Series, pnl: pd.DataFrame) -> tuple[float, float]:
    if not book.abs().sum():
        return 0.0, 0.0
    return historical_var_es(filtered_scenarios(book, pnl), 0.99)


def run_cycle(
    as_of: pd.Timestamp,
    closes: pd.DataFrame,
    *,
    strategy: str,
    capital: float,
    limits: RiskLimits,
    broker: Broker,
    journal: Journal,
    target_vol: float = 0.15,
    max_bar_age_days: int = 5,
    send_alert: bool = True,
) -> CycleResult:
    as_of = pd.Timestamp(as_of)
    history = closes.loc[:as_of]
    today = history.index[-1]
    symbols = list(history.columns)
    pnl_contract = to_usd(pnl_per_contract(history.ffill())).reindex(history.index).fillna(0.0)

    # 1. data checks
    last_bar = history.apply(lambda col: col.last_valid_index())
    stale = [s for s in symbols if last_bar[s] is None or (today - last_bar[s]).days > max_bar_age_days]

    # 2. mark to market
    held = broker.positions().reindex(symbols).fillna(0.0)
    daily_pnl = float(held @ pnl_contract.iloc[-1])
    equity_history = journal.equity_history(strategy)
    equity = (equity_history.iloc[-1] if len(equity_history) else capital) + daily_pnl
    peak = max([capital, equity, *equity_history.tolist()])
    drawdown = (equity - peak) / capital

    # 3. target
    signals = STRATEGIES[strategy](history)
    target = size_positions(signals, history, capital=capital, target_vol=target_vol).positions.iloc[-1]
    target[stale] = held[stale]

    # 4. pre-trade review
    values = contract_value_frame(history).iloc[-1].replace(0.0, np.nan).fillna(1.0)
    result = review(held, target, values, limits=limits, capital=capital, daily_pnl=daily_pnl, drawdown=drawdown)
    actions = list(result.actions) + [f"{s}: data is stale (last bar {last_bar[s]:%Y-%m-%d}), position frozen"
                                      for s in stale if last_bar[s] is not None]

    # 5. VaR limit: scale the approved book down until it fits, then review the smaller book again
    var_99, es_99 = _var_es(result.approved, pnl_contract)
    if var_99 > limits.max_var_99:
        scale = limits.max_var_99 / var_99
        smaller = np.trunc(result.approved * scale)
        result = review(held, smaller, values, limits=limits, capital=capital, daily_pnl=daily_pnl, drawdown=drawdown)
        actions.append(f"VaR {var_99:,.0f} over limit {limits.max_var_99:,.0f}: book scaled by {scale:.2f}")
        var_99, es_99 = _var_es(result.approved, pnl_contract)

    # 6. execute, journal, alert
    fills = broker.execute(result.orders, history.ffill().iloc[-1], today)
    status = "halted" if result.halted else ("adjusted" if actions else "ok")
    gross = float((result.approved.abs() * values).sum())
    key = {"date": today.strftime("%Y-%m-%d"), "strategy": strategy}
    journal.record("cycles", [{**key, "status": status, "daily_pnl": daily_pnl, "equity": equity,
                               "drawdown": drawdown, "gross_exposure": gross, "var_99": var_99, "es_99": es_99,
                               "orders": len(fills), "halted": int(result.halted), "note": "; ".join(actions)}])
    journal.record("actions", [{**key, "action": a} for a in actions])
    journal.record("limit_checks", [{**key, "name": c.name, "value": c.value, "limit_value": c.limit,
                                     "utilisation": c.utilisation} for c in result.checks]
                   + [{**key, "name": "VaR 99%", "value": var_99, "limit_value": limits.max_var_99,
                       "utilisation": var_99 / limits.max_var_99}])
    journal.record("fills", [{**key, "symbol": f.symbol, "contracts": f.contracts, "price": f.price, "cost": f.cost}
                             for f in fills])
    journal.record("positions", [{**key, "symbol": s, "contracts": float(c)}
                                 for s, c in result.approved.items() if c])

    hottest = max(result.checks, key=lambda c: c.utilisation)
    lines = [
        f"[{status.upper()}] {strategy} {today:%Y-%m-%d}",
        f"P&L {daily_pnl:+,.0f}  equity {equity:,.0f}  drawdown {drawdown:.1%}",
        f"VaR99 {var_99:,.0f} ({var_99 / limits.max_var_99:.0%} of limit)  ES99 {es_99:,.0f}",
        f"gross {gross / capital:.2f}x capital, {len(fills)} orders, highest use: {hottest.name} {hottest.utilisation:.0%}",
        *[f"- {a}" for a in actions[:8]],
    ]
    message = "\n".join(lines)
    if send_alert:
        alerts.send(message)
    return CycleResult(today, status, result.approved, result.orders, var_99, es_99, daily_pnl, drawdown, actions,
                       message)
