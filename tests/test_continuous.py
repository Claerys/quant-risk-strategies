import pandas as pd
import pytest

from quant_risk.continuous import ContractBars, back_adjust, reconcile, splice


def bars(dates: pd.DatetimeIndex, closes: list[float], volumes: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes},
        index=pd.DatetimeIndex(dates, name="date"),
    )


DAYS = pd.bdate_range("2025-01-06", periods=6)  # Mon..Mon


def two_contracts() -> list[ContractBars]:
    # Front month trades at 100..105; the next month is always 10 above (contango).
    # Volume crosses over on day 3, so that is the roll date.
    front = bars(DAYS, [100, 101, 102, 103, 104, 105], [900, 800, 300, 200, 100, 50])
    back = bars(DAYS, [110, 111, 112, 113, 114, 115], [100, 200, 400, 600, 800, 900])
    return [
        ContractBars("CLH5", pd.Timestamp("2025-02-20"), front),
        ContractBars("CLJ5", pd.Timestamp("2025-03-20"), back),
    ]


def test_rolls_when_next_contract_volume_overtakes() -> None:
    _, rolls = back_adjust(two_contracts())
    assert len(rolls) == 1
    assert rolls[0].date == DAYS[2]
    assert (rolls[0].from_contract, rolls[0].to_contract, rolls[0].gap) == ("CLH5", "CLJ5", 10.0)


def test_back_adjustment_removes_the_roll_gap() -> None:
    series, _ = back_adjust(two_contracts())
    # The latest prices are the real traded prices of the contract held now...
    assert series["close"].iloc[-1] == 115.0
    # ...and older prices are shifted by the gap, so every daily change is a real 1-point move.
    assert series["close"].diff().dropna().tolist() == [1.0] * 5


def test_forced_roll_before_expiry_even_without_volume_crossover() -> None:
    front = bars(DAYS, [100] * 6, [900] * 6)
    back = bars(DAYS, [110] * 6, [10] * 6)
    contracts = [
        ContractBars("NGF5", DAYS[3], front),  # expires on day 4
        ContractBars("NGG5", pd.Timestamp("2025-02-26"), back),
    ]
    _, rolls = back_adjust(contracts, force_days=2)
    assert rolls[0].date == DAYS[1]  # 2 business days before expiry


def test_never_rolls_backwards() -> None:
    contracts = two_contracts()
    later = bars(DAYS, [120] * 6, [0, 0, 0, 0, 0, 10_000])
    contracts.append(ContractBars("CLK5", pd.Timestamp("2025-04-21"), later))
    _, rolls = back_adjust(contracts)
    order = [r.to_contract for r in rolls]
    assert order == ["CLJ5", "CLK5"]


def test_splice_continues_history_from_its_last_level() -> None:
    history = bars(DAYS[:4], [50, 51, 52, 53], [1] * 4)
    extension = bars(DAYS[2:], [200, 201, 202, 203], [1] * 4)  # same moves, different level
    result = splice(history, extension)
    assert result.anchor == DAYS[3]
    assert result.added == 2
    assert result.bars["close"].tolist() == [50, 51, 52, 53, 54, 55]


def test_reconcile_accepts_two_sources_that_agree_on_moves() -> None:
    dates = pd.bdate_range("2025-01-01", periods=30)
    moves = pd.Series([(-1) ** i * (1 + i % 3) for i in range(30)]).cumsum()
    archive = bars(dates, (moves + 500).tolist(), [1] * 30)
    other_level = bars(dates, (moves + 80).tolist(), [1] * 30)
    report = reconcile(archive, other_level)
    assert report["ok"]
    assert report["mismatched_days"] == 0


def test_splice_needs_overlap() -> None:
    with pytest.raises(ValueError, match="download further back"):
        splice(bars(DAYS[:2], [1, 2], [1, 1]), bars(DAYS[3:], [1, 2, 3], [1, 1, 1]))


def test_reconcile_catches_a_units_mismatch() -> None:
    dates = pd.bdate_range("2025-01-01", periods=30)
    moves = [(-1) ** i * (1 + i % 3) for i in range(30)]
    archive = bars(dates, pd.Series(moves).cumsum().add(500).tolist(), [1] * 30)
    in_dollars = archive.copy()
    in_dollars[["open", "high", "low", "close"]] /= 100  # cents vs dollars
    report = reconcile(archive, in_dollars)
    assert not report["ok"]
    assert report["scale_ratio"] == pytest.approx(0.01)
