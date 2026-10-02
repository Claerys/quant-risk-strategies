import numpy as np
import pandas as pd
import pytest

from quant_risk.strategies import (
    STRATEGIES,
    ma_crossover,
    mean_reversion,
    multi_horizon_trend,
    standardised_trend,
    time_series_momentum,
)

DATES = pd.bdate_range("2020-01-01", periods=400)


def trending(slope: float, start: float = 100.0, noise: float = 0.5, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(start + slope * np.arange(len(DATES)) + rng.normal(0, noise, len(DATES)), index=DATES)


def frame(**columns: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(columns)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_every_strategy_stays_within_minus_one_and_one(name: str) -> None:
    closes = frame(UP=trending(0.3), DOWN=trending(-0.3, seed=2))
    signal = STRATEGIES[name](closes)
    assert signal.shape == closes.shape
    values = signal.stack().dropna()
    assert ((values >= -1) & (values <= 1)).all()


def test_momentum_is_long_an_uptrend_and_short_a_downtrend() -> None:
    signal = time_series_momentum(frame(UP=trending(0.3), DOWN=trending(-0.3, seed=2)))
    assert signal["UP"].iloc[-1] == 1.0
    assert signal["DOWN"].iloc[-1] == -1.0


def test_long_only_momentum_goes_to_cash_instead_of_short() -> None:
    signal = time_series_momentum(frame(DOWN=trending(-0.3, seed=2)), long_only=True)
    assert signal["DOWN"].iloc[-1] == 0.0


def test_momentum_needs_a_full_lookback() -> None:
    signal = time_series_momentum(frame(UP=trending(0.3)))
    assert signal["UP"].iloc[:252].isna().all()


def test_signals_ignore_the_back_adjustment_level() -> None:
    # Shifting every price by a constant (what back-adjustment does) must not change any signal,
    # even when the shift pushes prices below zero.
    closes = frame(CL=trending(0.2))
    shifted = closes - 150.0
    for strategy in (time_series_momentum, multi_horizon_trend, ma_crossover, mean_reversion):
        pd.testing.assert_frame_equal(strategy(closes), strategy(shifted))


def test_signal_does_not_use_future_prices() -> None:
    closes = frame(ES=trending(0.1))
    full = multi_horizon_trend(closes)
    truncated = multi_horizon_trend(closes.iloc[:300])
    pd.testing.assert_frame_equal(full.iloc[:300], truncated)


def test_standardised_trend_is_scale_free() -> None:
    # The same path quoted in cents instead of dollars gives the same standardised trend.
    closes = frame(C=trending(0.2))
    pd.testing.assert_frame_equal(standardised_trend(closes, 63), standardised_trend(closes * 100, 63))


def test_mean_reversion_leans_against_a_spike() -> None:
    closes = frame(GC=pd.Series(100.0 + np.sin(np.arange(len(DATES))), index=DATES))
    closes.iloc[-1] = 120.0
    assert mean_reversion(closes)["GC"].iloc[-1] == -1.0


def test_crossover_rejects_inverted_windows() -> None:
    with pytest.raises(ValueError):
        ma_crossover(frame(ES=trending(0.1)), fast=100, slow=20)
