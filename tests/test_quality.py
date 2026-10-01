import numpy as np
import pandas as pd

from quant_risk.quality import ERROR, INFO, WARNING, check_bars, summarize


def clean_bars(n: int = 100, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = start + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1_000},
        index=pd.bdate_range("2024-01-01", periods=n, name="date"),
    )


def checks(issues, severity=None) -> set[str]:
    return {i.check for i in issues if severity is None or i.severity == severity}


def test_clean_data_has_no_issues() -> None:
    assert check_bars("ES", clean_bars()) == []


def test_impossible_bars_are_errors() -> None:
    bars = clean_bars()
    bars.iloc[10, bars.columns.get_loc("high")] = bars.iloc[10]["low"] - 5
    bars.iloc[20, bars.columns.get_loc("close")] = np.nan
    bars.iloc[30, bars.columns.get_loc("close")] = bars.iloc[30]["high"] + 5
    assert checks(check_bars("ES", bars), ERROR) == {"high_below_low", "missing_price", "close_outside_range"}


def test_gap_is_flagged() -> None:
    bars = clean_bars().drop(pd.bdate_range("2024-02-01", "2024-02-20"))
    assert "gap" in checks(check_bars("ES", bars), WARNING)


def test_stale_price_is_flagged_once_per_run() -> None:
    bars = clean_bars()
    bars.iloc[40:47, bars.columns.get_loc("close")] = bars.iloc[40]["close"]
    bars["high"] = bars[["high", "close"]].max(axis=1)
    bars["low"] = bars[["low", "close"]].min(axis=1)
    stale = [i for i in check_bars("ES", bars) if i.check == "stale_price"]
    assert len(stale) == 1


def test_bad_tick_is_an_extreme_move() -> None:
    bars = clean_bars()
    bars.iloc[50, bars.columns.get_loc("close")] += 100
    bars["high"] = bars[["high", "close"]].max(axis=1)
    assert "extreme_move" in checks(check_bars("ES", bars), WARNING)


def test_series_that_stopped_updating_is_flagged() -> None:
    bars = clean_bars()
    as_of = bars.index[-1] + pd.Timedelta(days=30)
    assert "stale_series" in checks(check_bars("ES", bars, as_of=as_of), WARNING)


def test_negative_back_adjusted_prices_are_info_not_error() -> None:
    bars = clean_bars(start=-50.0)
    found = check_bars("CL", bars)
    assert checks(found) == {"non_positive_price"}
    assert found[0].severity == INFO


def test_summary_puts_errors_first() -> None:
    bars = clean_bars(start=-50.0)
    bars.iloc[10, bars.columns.get_loc("high")] = bars.iloc[10]["low"] - 5
    summary = summarize(check_bars("CL", bars))
    assert summary.iloc[0]["severity"] == ERROR
