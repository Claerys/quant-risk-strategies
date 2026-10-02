import numpy as np
import pandas as pd
import pytest

from quant_risk.backtest import run_backtest
from quant_risk.metrics import drawdown, performance_summary

DATES = pd.bdate_range("2024-01-01", periods=4)


def closes(**cols: list[float]) -> pd.DataFrame:
    return pd.DataFrame(cols, index=DATES)


def test_position_earns_only_the_moves_after_it_is_taken() -> None:
    prices = closes(ES=[4000.0, 4010.0, 4030.0, 4020.0])
    # Buy 2 contracts at the close of day 1; day 1's own +10 move must not count.
    positions = closes(ES=[0.0, 2.0, 2.0, 2.0])
    result = run_backtest(positions, prices, capital=1_000_000, cost_per_contract=0.0)
    assert result.pnl["ES"].tolist() == [0.0, 0.0, 2 * 20 * 50, 2 * -10 * 50]


def test_short_position_profits_from_a_fall() -> None:
    prices = closes(CL=[70.0, 70.0, 69.0, 69.0])
    result = run_backtest(closes(CL=[-1.0] * 4), prices, capital=100_000, cost_per_contract=0.0)
    assert result.pnl["CL"].sum() == 1_000.0


def test_costs_are_charged_on_every_contract_traded() -> None:
    prices = closes(ES=[4000.0] * 4)
    positions = closes(ES=[3.0, 3.0, -1.0, -1.0])  # buy 3, then sell 4
    result = run_backtest(positions, prices, capital=1_000_000, cost_per_contract=2.5)
    assert result.costs["ES"].tolist() == [7.5, 0.0, 10.0, 0.0]
    assert result.net_pnl.sum() == -17.5


def test_missing_price_day_carries_the_move_to_the_next_trading_day() -> None:
    prices = closes(ES=[4000.0, np.nan, 4020.0, 4020.0])
    result = run_backtest(closes(ES=[1.0] * 4), prices, capital=1_000_000, cost_per_contract=0.0)
    assert result.pnl["ES"].tolist() == [0.0, 0.0, 1_000.0, 0.0]


def test_returns_and_equity_use_fixed_capital() -> None:
    prices = closes(ES=[4000.0, 4020.0, 4000.0, 4040.0])
    result = run_backtest(closes(ES=[1.0] * 4), prices, capital=100_000, cost_per_contract=0.0)
    assert result.returns.tolist() == pytest.approx([0.0, 0.01, -0.01, 0.02])
    assert result.equity.iloc[-1] == 102_000.0


def test_drawdown_measures_fall_from_peak() -> None:
    returns = pd.Series([0.10, -0.05, -0.05, 0.02])
    assert drawdown(returns).tolist() == pytest.approx([0.0, -0.05, -0.10, -0.08])


def test_summary_of_a_known_series() -> None:
    returns = pd.Series([0.01, -0.01] * 126)
    summary = performance_summary(returns)
    assert summary["annual_return"] == pytest.approx(0.0)
    assert summary["annual_volatility"] == pytest.approx(0.01 * np.sqrt(252), rel=1e-2)
    assert summary["hit_rate"] == 0.5
    assert summary["max_drawdown"] == pytest.approx(-0.01)
