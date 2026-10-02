import pandas as pd
import pytest

from quant_risk.bars import write_bars
from quant_risk.fx import extend_with_futures, to_usd, usd_per_unit


def series(dates: str, values: list[float], periods: int | None = None) -> pd.Series:
    index = pd.bdate_range(dates, periods=periods or len(values))
    return pd.Series(values, index=index)


def as_bars(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 0})


def test_extension_follows_futures_changes_not_level() -> None:
    spot = series("2024-01-03", [1.10, 1.11])  # Wed, Thu
    futures = series("2024-01-01", [5.00, 5.01, 5.02, 5.03, 5.05])  # back-adjusted: wrong level
    rate = extend_with_futures(spot, futures)
    assert rate.tolist() == pytest.approx([1.08, 1.09, 1.10, 1.11, 1.13])


@pytest.fixture
def eur_data(tmp_path, monkeypatch):
    monkeypatch.setenv("QRS_DATA_DIR", str(tmp_path / "daily"))
    dates = pd.bdate_range("2024-01-01", periods=5)
    write_bars("EURUSD", as_bars(pd.Series(1.10, index=dates)), tmp_path / "fx")
    write_bars("EC", as_bars(pd.Series(1.30, index=dates)), tmp_path / "daily")
    return dates


def test_usd_needs_no_conversion() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    assert usd_per_unit("USD", dates).tolist() == [1.0, 1.0, 1.0]


def test_euro_pnl_is_converted_at_the_daily_rate(eur_data) -> None:
    pnl = pd.DataFrame({"FDAX": [1_000.0] * 5, "ES": [500.0] * 5}, index=eur_data)
    usd = to_usd(pnl)
    assert usd["FDAX"].tolist() == pytest.approx([1_100.0] * 5)
    assert usd["ES"].tolist() == [500.0] * 5
