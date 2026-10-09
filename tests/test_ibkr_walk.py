import pandas as pd
import pytest

from quant_risk.ibkr.pacing import Pacer
from quant_risk.ibkr.walk import IncompleteCoverageError, chunk_budget, end_before, walk_backward

TODAY = pd.Timestamp("2025-06-30")


def frame(start: str, end: str) -> pd.DataFrame:
    index = pd.bdate_range(start, end, name="date")
    close = pd.Series(range(len(index)), index=index, dtype=float)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 0.0})


def test_end_before_is_the_previous_day_last_second():
    assert end_before(pd.Timestamp("2024-01-02")) == "20240101-23:59:59"


def test_walk_reaches_the_cutoff_and_stitches_chunks():
    chunks = [frame("2020-07-01", "2025-06-30"), frame("2015-06-01", "2020-06-30")]
    ends = []

    def fetch(duration, end):
        ends.append((duration, end))
        return chunks[len(ends) - 1]
    result = walk_backward(fetch, 10, today=TODAY)
    assert result.reached_target and result.is_complete
    assert ends == [("5 Y", ""), ("5 Y", "20200630-23:59:59")]
    assert result.bars.index.is_monotonic_increasing and result.bars.index.is_unique


def test_walk_stops_when_ibkr_has_nothing_older():
    calls = iter([frame("2023-01-02", "2025-06-30"), frame("2023-01-02", "2025-06-30")])
    result = walk_backward(lambda *_: next(calls), 20, today=TODAY)
    assert result.data_ran_out and "stopped moving" in result.stopped_because


def test_empty_response_is_a_complete_answer():
    result = walk_backward(lambda *_: pd.DataFrame(), 5, today=TODAY)
    assert result.data_ran_out and result.bars.empty


def test_budget_exhaustion_is_an_error_not_success():
    state = {"n": 0}

    def creeping(_duration, _end):  # always one more day of history, never reaching the cutoff
        state["n"] += 1
        oldest = pd.Timestamp("2025-06-30") - pd.offsets.BDay(state["n"])
        return frame(oldest, "2025-06-30")
    result = walk_backward(creeping, 20, today=TODAY)
    assert not result.is_complete and result.chunks == chunk_budget(20)
    with pytest.raises(IncompleteCoverageError, match="ran out"):
        result.require_coverage("EURUSD", 20)


def test_pacer_sleeps_only_after_the_window_fills():
    now, slept = [0.0], []
    pacer = Pacer(max_requests=3, window=10, clock=lambda: now[0], sleep=lambda s: (slept.append(s), now.__setitem__(0, now[0] + s)))
    for _ in range(3):
        pacer.wait()
    assert slept == []
    pacer.wait()  # the fourth must wait for the first to leave the window
    assert slept == [10]
