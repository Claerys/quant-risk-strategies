from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_risk.limits import (
    DEFAULT_LIMITS_FILE,
    LimitsNotConfigured,
    RiskLimits,
    replay_loss_limits,
    review,
)

LIMITS = RiskLimits(
    max_gross_exposure=1_000_000,
    max_net_exposure=800_000,
    max_position_notional=600_000,
    max_sector_gross=700_000,
    max_order_contracts=50,
    max_daily_loss=10_000,
    max_drawdown=0.10,
)
SYMBOLS = ["ES", "NQ", "TY", "CL"]
VALUE = pd.Series({"ES": 250_000.0, "NQ": 400_000.0, "TY": 110_000.0, "CL": 70_000.0})


def run(current, target, limits=LIMITS, daily_pnl=0.0, drawdown=0.0):
    return review(
        pd.Series(current, index=SYMBOLS, dtype=float),
        pd.Series(target, index=SYMBOLS, dtype=float),
        VALUE,
        limits=limits,
        capital=1_000_000,
        daily_pnl=daily_pnl,
        drawdown=drawdown,
    )


def test_missing_limit_refuses_to_start() -> None:
    with pytest.raises(LimitsNotConfigured, match="max_daily_loss is not set"):
        RiskLimits.from_mapping({"max_gross_exposure": 1, "max_net_exposure": 1, "max_position_notional": 1,
                                 "max_sector_gross": 1, "max_order_contracts": 1, "max_drawdown": 0.1})


def test_zero_or_text_limit_is_rejected() -> None:
    with pytest.raises(LimitsNotConfigured, match="must be positive"):
        RiskLimits.from_mapping({**vars(LIMITS), "max_gross_exposure": 0})
    with pytest.raises(LimitsNotConfigured, match="not a number"):
        RiskLimits.from_mapping({**vars(LIMITS), "max_net_exposure": "lots"})


def test_shipped_limits_file_is_complete() -> None:
    assert RiskLimits.from_file(DEFAULT_LIMITS_FILE).max_drawdown == 0.25


def test_book_within_limits_is_approved_unchanged() -> None:
    result = run([0, 0, 0, 0], [1, 0, 2, -1])
    assert result.approved.tolist() == [1, 0, 2, -1]
    assert not result.actions and not result.halted


def test_kill_switch_blocks_every_order() -> None:
    result = run([1, 0, 0, 0], [0, 1, 2, 0], limits=replace(LIMITS, kill_switch=True))
    assert result.orders.abs().sum() == 0
    assert result.halted


def test_daily_loss_stop_allows_only_reducing_orders() -> None:
    # Cut ES from 2 to 1 (allowed), add TY (blocked), flip CL from short to long (blocked).
    result = run([2, 0, 0, -1], [1, 0, 3, 1], daily_pnl=-12_000)
    assert result.approved.tolist() == [1, 0, 0, -1]
    assert result.halted


def test_drawdown_limit_closes_the_book() -> None:
    assert run([2, 0, 1, -1], [3, 1, 1, -1], drawdown=-0.12).approved.tolist() == [0, 0, 0, 0]


def test_unknown_pnl_fails_closed() -> None:
    result = run([0, 0, 0, 0], [1, 0, 0, 0], daily_pnl=None)
    assert result.approved.tolist() == [0, 0, 0, 0]
    assert "cannot measure" in result.actions[0]


def test_position_notional_cap() -> None:
    # 3 ES = 750k > 600k cap -> 2 contracts.
    assert run([0, 0, 0, 0], [3, 0, 0, 0]).approved["ES"] == 2


def test_sector_limit_scales_the_whole_sector() -> None:
    # ES 2 (500k) + NQ 1 (400k) = 900k in US equity indices > 700k.
    result = run([0, 0, 0, 0], [2, 1, 0, 0])
    approved = result.approved
    assert approved["ES"] * 250_000 + approved["NQ"] * 400_000 <= 700_000
    assert any("sector us_equity_index" in a for a in result.actions)


def test_gross_limit_never_exceeded_after_rounding() -> None:
    result = run([0, 0, 0, 0], [2, 1, 5, -5], limits=replace(LIMITS, max_sector_gross=10_000_000))
    gross = (result.approved.abs() * VALUE).sum()
    assert gross <= LIMITS.max_gross_exposure


def test_net_limit_trims_the_dominant_side() -> None:
    limits = replace(LIMITS, max_gross_exposure=10e6, max_sector_gross=10e6, max_net_exposure=300_000)
    result = run([0, 0, 0, 0], [1, 0, 2, -1], limits=limits)  # net 250k+220k-70k = 400k long
    net = (result.approved * VALUE).sum()
    assert abs(net) <= 300_000
    assert result.approved["CL"] == -1  # the short side is left alone


def test_large_order_is_cut_not_rejected() -> None:
    limits = replace(LIMITS, max_order_contracts=2, max_gross_exposure=10e6, max_sector_gross=10e6,
                     max_net_exposure=10e6, max_position_notional=10e6)
    result = run([0, 0, 0, 0], [0, 0, 5, 0], limits=limits)
    assert result.orders["TY"] == 2


def test_utilisation_is_reported_for_every_limit() -> None:
    util = run([0, 0, 0, 0], [1, 0, 0, 0]).utilisation()
    assert util["gross exposure"] == pytest.approx(0.25)
    assert util["largest position"] == pytest.approx(250_000 / 600_000)


def test_replay_stops_a_losing_strategy_at_the_drawdown_limit() -> None:
    dates = pd.bdate_range("2024-01-01", periods=60)
    closes = pd.DataFrame({"ES": 5000.0 - 20.0 * np.arange(60)}, index=dates)  # steady fall
    targets = pd.DataFrame({"ES": 4.0}, index=dates)  # stubbornly long
    limits = replace(LIMITS, max_daily_loss=1e9, max_drawdown=0.20, max_order_contracts=10)
    result = replay_loss_limits(targets, closes, capital=1_000_000, limits=limits)
    assert result.stopped_on is not None
    assert result.positions["ES"].loc[result.stopped_on:].iloc[0] == 0.0  # closed the same day
    assert (result.positions["ES"].loc[result.stopped_on:] == 0).all()  # and stays flat
    assert result.utilisation["drawdown"].max() >= 1.0
