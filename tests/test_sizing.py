import numpy as np
import pandas as pd
import pytest

from quant_risk.backtest import run_backtest
from quant_risk.metrics import performance_summary
from quant_risk.sizing import (
    apply_buffer,
    diversification_multiplier,
    ex_ante_vol,
    sector_weights,
    size_positions,
    usd_vol_per_contract,
)

DATES = pd.bdate_range("2015-01-01", periods=900)


def random_walk(daily_vol: float, seed: int, start: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(start + np.cumsum(rng.normal(0, daily_vol, len(DATES))), index=DATES)


def test_dollar_vol_includes_the_multiplier() -> None:
    closes = pd.DataFrame({"ES": random_walk(20.0, 1, 4000.0)})
    vol = usd_vol_per_contract(closes)["ES"].iloc[-1]
    assert vol == pytest.approx(20.0 * 50, rel=0.25)  # 20 points a day x $50 a point


def test_sector_weights_share_risk_by_sector_first() -> None:
    w = sector_weights(["ES", "NQ", "TY", "C", "W", "S"])
    assert w.sum() == pytest.approx(1.0)
    assert w["TY"] == pytest.approx(1 / 3)  # alone in rates
    assert w["ES"] == pytest.approx(1 / 6)  # shares equities with NQ
    assert w["C"] == pytest.approx(1 / 9)  # shares grains with W and S


def test_diversification_multiplier() -> None:
    a = pd.Series(np.random.default_rng(1).normal(0, 1, len(DATES)), index=DATES)
    b = pd.Series(np.random.default_rng(2).normal(0, 1, len(DATES)), index=DATES)
    weights = pd.Series({"X": 0.5, "Y": 0.5})
    same = diversification_multiplier(pd.DataFrame({"X": a, "Y": a}), weights)
    independent = diversification_multiplier(pd.DataFrame({"X": a, "Y": b}), weights)
    assert same.iloc[-1] == pytest.approx(1.0)
    assert independent.iloc[-1] == pytest.approx(np.sqrt(2), rel=0.1)
    assert same.iloc[0] == 1.0  # no estimate before enough history


def test_ex_ante_vol_of_offsetting_positions_is_lower() -> None:
    a = pd.Series(np.random.default_rng(1).normal(0, 100, len(DATES)), index=DATES)
    pnl = pd.DataFrame({"X": a, "Y": a})  # perfectly correlated
    hedged = ex_ante_vol(pd.DataFrame({"X": 1.0, "Y": -1.0}, index=DATES), pnl)
    doubled = ex_ante_vol(pd.DataFrame({"X": 1.0, "Y": 1.0}, index=DATES), pnl)
    assert hedged.iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert doubled.iloc[-1] == pytest.approx(2 * a.std(), rel=0.2)


def test_buffer_ignores_small_changes_and_trades_to_the_band_edge() -> None:
    idx = pd.RangeIndex(5)
    ideal = pd.DataFrame({"ES": [10.0, 10.5, 9.6, 14.0, 13.9]}, index=idx)
    full = pd.DataFrame({"ES": [10.0] * 5}, index=idx)  # band = +/- 1 contract
    held = apply_buffer(ideal, full, buffer=0.1)["ES"].tolist()
    assert held == [9.0, 10.0, 10.0, 13.0, 13.0]


def test_portfolio_runs_near_its_volatility_target() -> None:
    closes = pd.DataFrame({
        "ES": random_walk(30.0, 1, 4000.0),
        "TY": random_walk(0.4, 2, 110.0),
        "CL": random_walk(1.5, 3, 70.0),
        "GC": random_walk(20.0, 4, 2000.0),
    })
    signals = pd.DataFrame(1.0, index=DATES, columns=closes.columns)
    capital, target = 50_000_000, 0.15
    sized = size_positions(signals, closes, capital=capital, target_vol=target, buffer=0.0)
    realised = performance_summary(run_backtest(sized.positions, closes, capital=capital).returns.iloc[300:])
    assert realised["annual_volatility"] == pytest.approx(target, rel=0.2)


def test_overlay_only_ever_cuts_risk() -> None:
    closes = pd.DataFrame({"ES": random_walk(30.0, 1, 4000.0), "NQ": random_walk(100.0, 2, 15000.0)})
    signals = pd.DataFrame(1.0, index=DATES, columns=closes.columns)
    sized = size_positions(signals, closes, capital=10_000_000, max_risk_ratio=0.5)
    assert (sized.overlay_scale <= 1.0).all()
    assert (sized.overlay_scale.iloc[200:] < 1.0).any()
