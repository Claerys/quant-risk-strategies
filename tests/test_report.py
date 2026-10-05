import numpy as np
import pandas as pd
import pytest

from quant_risk.bars import write_bars
from quant_risk.limits import RiskLimits
from quant_risk.report import build_report, var_backtest_report

DATES = pd.bdate_range("2019-01-01", periods=900)
LIMITS = RiskLimits(
    max_gross_exposure=1e9, max_net_exposure=1e9, max_position_notional=1e9, max_sector_gross=1e9,
    max_order_contracts=10_000, max_daily_loss=1e9, max_drawdown=0.9, max_var_99=1e9,
)


@pytest.fixture(autouse=True)
def synthetic_market(tmp_path, monkeypatch):
    monkeypatch.setenv("QRS_DATA_DIR", str(tmp_path / "daily"))
    rng = np.random.default_rng(11)
    for symbol, start, vol, drift in [("ES", 4000, 30, 1.5), ("TY", 110, 0.4, -0.02), ("CL", 70, 1.5, 0.03)]:
        close = start + np.cumsum(rng.normal(drift, vol, len(DATES)))
        bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000}, index=DATES)
        write_bars(symbol, bars, tmp_path / "daily")


def test_report_has_every_section() -> None:
    report = build_report(limits=LIMITS, start="2019")
    assert not report.positions.empty
    assert set(report.var.index) >= {"historical", "filtered historical", "parametric (normal)"}
    assert "VaR 99%" in report.limits.index
    assert len(report.stress) == 15
    assert report.headline["worst stress"] in report.stress.index
    assert report.equity.index[0] >= pd.Timestamp("2019-01-01")


def test_var_backtest_report_compares_methods() -> None:
    summary, series = var_backtest_report(start="2020")
    assert list(summary.index) == ["historical", "filtered", "parametric"]
    assert {"P&L", "filtered VaR", "filtered zone"} <= set(series.columns)
