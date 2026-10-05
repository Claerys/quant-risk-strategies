import numpy as np
import pandas as pd
import pytest

from quant_risk.var_backtest import (
    backtest_var,
    basel_zone,
    christoffersen_independence,
    exceptions,
    kupiec_pof,
    rolling_basel_zone,
)

DATES = pd.bdate_range("2015-01-01", periods=2000)


def test_exception_compares_today_with_yesterdays_var() -> None:
    var = pd.Series([100.0, 100.0, 50.0], index=DATES[:3])
    pnl = pd.Series([0.0, -120.0, -80.0], index=DATES[:3])
    # Day 2: -120 vs VaR 100 from day 1 -> exception. Day 3: -80 vs VaR 100 from day 2 -> none.
    assert exceptions(var, pnl).tolist()[1:] == [True, False]


def test_kupiec_accepts_the_expected_rate_and_rejects_far_too_many() -> None:
    assert kupiec_pof(10, 1000, 0.99)[1] > 0.5
    assert kupiec_pof(30, 1000, 0.99)[1] < 0.001
    assert kupiec_pof(0, 1000, 0.99)[1] < 0.001  # never exceeded: VaR is far too high


def test_kupiec_known_value() -> None:
    # 5 exceptions in 250 days at 99%: textbook LR ~ 1.96, p ~ 0.16.
    lr, p = kupiec_pof(5, 250, 0.99)
    assert lr == pytest.approx(1.956, abs=0.01)
    assert p == pytest.approx(0.162, abs=0.01)


def test_independence_test_detects_clustering() -> None:
    clustered = pd.Series([0] * 980 + [1] * 10 + [0] * 10)
    spread = pd.Series(([0] * 99 + [1]) * 10)
    assert christoffersen_independence(clustered)[1] < 0.01
    assert christoffersen_independence(spread)[1] > 0.05


@pytest.mark.parametrize(("count", "zone"), [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")])
def test_basel_zones(count: int, zone: str) -> None:
    assert basel_zone(count) == zone


def test_a_correct_model_passes_and_an_undersized_one_fails() -> None:
    rng = np.random.default_rng(3)
    pnl = pd.Series(rng.normal(0, 1_000, len(DATES)), index=DATES)
    correct = pd.Series(2.3263 * 1_000, index=DATES)
    too_small = pd.Series(1.5 * 1_000, index=DATES)
    good = backtest_var(correct, pnl)
    bad = backtest_var(too_small, pnl)
    assert good.verdict == "not rejected"
    assert bad.verdict == "rejected: the model understates risk"
    assert bad.exceptions > good.exceptions


def test_rolling_zone_turns_red_after_a_cluster() -> None:
    hits = pd.Series([False] * 300 + [True] * 12, index=DATES[:312])
    zones = rolling_basel_zone(hits)
    assert zones.iloc[300] == "green"
    assert zones.iloc[-1] == "red"
