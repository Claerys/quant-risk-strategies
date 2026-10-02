"""Performance statistics of a daily return series.

All statistics are annualised with 252 trading days. Returns are taken as excess returns (see
backtest.py), so the Sharpe ratio needs no risk-free rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def drawdown(returns: pd.Series) -> pd.Series:
    """Fall from the running peak of cumulative P&L, as a fraction of capital (0 or negative).

    Returns are on fixed capital and do not compound, so the equity curve is 1 + cumulative sum.
    """
    equity = 1.0 + returns.cumsum()
    return equity - equity.cummax()


def performance_summary(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    mean, std = returns.mean(), returns.std()
    downside = returns[returns < 0].std()
    dd = drawdown(returns)
    annual_return = mean * TRADING_DAYS
    max_dd = dd.min()
    return {
        "annual_return": annual_return,
        "annual_volatility": std * np.sqrt(TRADING_DAYS),
        "sharpe": mean / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan,
        "sortino": mean / downside * np.sqrt(TRADING_DAYS) if downside > 0 else np.nan,
        "max_drawdown": max_dd,
        "calmar": annual_return / -max_dd if max_dd < 0 else np.nan,
        "hit_rate": (returns > 0).sum() / (returns != 0).sum() if (returns != 0).any() else np.nan,
        "skew": returns.skew(),
        "worst_day": returns.min(),
        "best_day": returns.max(),
    }
