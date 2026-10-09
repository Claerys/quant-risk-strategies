"""The risk engine on a stock book: one share is one contract with a multiplier of 1.

Synthetic prices only. This proves the plumbing (universe, data folder, limits file, report), not that
any strategy makes money on stocks.
"""

import numpy as np
import pandas as pd
import pytest

from quant_risk.bars import write_bars
from quant_risk.instruments import get_instrument, load_equities
from quant_risk.report import build_report, strategy_comparison
from quant_risk.strategies import STRATEGIES

CAPITAL = 10_000_000
SYMBOLS = ["SPY", "AAPL", "MSFT", "XOM", "JPM", "JNJ", "NEE", "CAT"]


@pytest.fixture
def stock_folder(tmp_path, monkeypatch):
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2019-01-01", "2025-05-30", name="date")
    folder = tmp_path / "stocks"
    for n, symbol in enumerate(SYMBOLS):
        close = pd.Series(100 + n * 20 + np.cumsum(rng.normal(0.05, 1.5, len(index))), index=index).clip(lower=5)
        write_bars(symbol, pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close,
                                         "volume": 1e6}), folder)
    monkeypatch.setenv("QRS_DATA_DIR", str(folder))
    monkeypatch.setenv("QRS_LIMITS_FILE", "config/limits_equities.toml")
    return folder


def test_a_share_is_a_contract_with_multiplier_one():
    aapl = get_instrument("AAPL")
    assert (aapl.multiplier, aapl.currency, aapl.asset_class) == (1.0, "USD", "equity")
    assert aapl.pnl(2.5, contracts=-100) == -250.0  # short 100 shares, price up $2.50
    assert aapl.notional(190.0, contracts=100) == 19_000.0
    assert all(i.multiplier == 1.0 and i.currency == "USD" for i in load_equities().values())


def test_the_daily_risk_report_runs_on_stock_data(stock_folder):
    report = build_report("time_series_momentum", CAPITAL)
    assert len(report.positions) > 0 and report.positions.index.isin(SYMBOLS).all()
    assert report.headline["VaR 99% (1d)"] > 0
    assert set(report.exposure_by_sector.index) <= {i.sector for i in load_equities().values()}
    # the stock limits file is the one in force: 1.5x gross, not the futures book's 4.0x
    assert report.limits.loc["gross exposure", "limit value"] == 15_000_000


def test_without_the_override_the_futures_limits_apply(stock_folder, monkeypatch):
    monkeypatch.delenv("QRS_LIMITS_FILE")
    assert build_report("time_series_momentum", CAPITAL).limits.loc["gross exposure", "limit value"] == 40_000_000


def test_the_strategy_comparison_runs_on_stock_data(stock_folder):
    table = strategy_comparison(CAPITAL)
    assert len(table) == len(STRATEGIES) and table.notna().all().all()
