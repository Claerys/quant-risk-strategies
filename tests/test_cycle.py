from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_risk.cycle import run_cycle
from quant_risk.limits import RiskLimits
from quant_risk.paper import Journal, PaperBroker

DATES = pd.bdate_range("2021-01-01", periods=700)
LIMITS = RiskLimits(
    max_gross_exposure=1e9, max_net_exposure=1e9, max_position_notional=1e9, max_sector_gross=1e9,
    max_order_contracts=10_000, max_daily_loss=1e9, max_drawdown=0.9, max_var_99=1e9,
)


@pytest.fixture
def closes() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "ES": 4000 + np.cumsum(rng.normal(2, 30, len(DATES))),
        "TY": 110 + np.cumsum(rng.normal(-0.01, 0.4, len(DATES))),
        "CL": 70 + np.cumsum(rng.normal(0.02, 1.5, len(DATES))),
    }, index=DATES)


@pytest.fixture
def setup(tmp_path):
    return PaperBroker(tmp_path / "state.json"), Journal(tmp_path / "journal.sqlite")


def cycle(closes, broker, journal, as_of, limits=LIMITS):
    return run_cycle(as_of, closes, strategy="time_series_momentum", capital=10_000_000, limits=limits,
                     broker=broker, journal=journal, send_alert=False)


def test_first_cycle_trades_to_target_and_journals_everything(closes, setup) -> None:
    broker, journal = setup
    result = cycle(closes, broker, journal, DATES[-1])
    assert result.status == "ok"
    assert broker.positions().reindex(result.approved.index).fillna(0).equals(result.approved)
    assert len(journal.read("cycles")) == 1
    assert {"gross exposure", "VaR 99%"} <= set(journal.read("limit_checks")["name"])
    assert len(journal.read("fills")) == int((result.orders != 0).sum())


def test_state_carries_over_between_cycles(closes, setup) -> None:
    broker, journal = setup
    cycle(closes, broker, journal, DATES[-2])
    second = cycle(closes, broker, journal, DATES[-1])
    assert second.daily_pnl != 0.0  # yesterday's book was marked to market
    assert len(journal.equity_history("time_series_momentum")) == 2


def test_cycle_never_uses_prices_after_as_of(closes, setup) -> None:
    broker, journal = setup
    future_crash = closes.copy()
    future_crash.iloc[-50:] *= 0.5
    a = cycle(closes, broker, journal, DATES[-60])
    broker2, journal2 = PaperBroker(broker.path.with_name("b.json")), Journal(journal.path.with_name("b.sqlite"))
    b = cycle(future_crash, broker2, journal2, DATES[-60])
    assert a.approved.equals(b.approved)


def test_var_limit_scales_the_book_down(closes, setup) -> None:
    broker, journal = setup
    result = cycle(closes, broker, journal, DATES[-1], limits=replace(LIMITS, max_var_99=50_000))
    assert result.var_99 <= 50_000 * 1.05
    assert any("VaR" in a for a in result.actions)


def test_stale_instrument_is_frozen(closes, setup) -> None:
    broker, journal = setup
    stale = closes.copy()
    stale.loc[DATES[-10]:, "CL"] = np.nan  # CL feed stopped two weeks ago
    result = cycle(stale, broker, journal, DATES[-1])
    assert result.orders["CL"] == 0
    assert any("CL: data is stale" in a for a in result.actions)


def test_kill_switch_halts_and_says_so(closes, setup) -> None:
    broker, journal = setup
    result = cycle(closes, broker, journal, DATES[-1], limits=replace(LIMITS, kill_switch=True))
    assert result.status == "halted"
    assert result.orders.abs().sum() == 0
    assert result.message.startswith("[HALTED]")
