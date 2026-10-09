import numpy as np
import pandas as pd
import pytest
from fake_gateway import FakeClient

from quant_risk.bars import bars_path, normalize_bars, write_bars
from quant_risk.ibkr.pacing import Pacer
from quant_risk.ibkr.qualification import QualificationError
from quant_risk.ibkr.stocks import download_stock
from quant_risk.ibkr.tickers import TickerDefinition

TODAY = pd.Timestamp("2025-06-30")
AAPL = TickerDefinition("AAPL", primary_exchange="NASDAQ")
SERIES = pd.Series(100 + np.random.default_rng(7).normal(0, 1.5, 6000).cumsum().clip(-90),
                   index=pd.bdate_range("2005-01-03", periods=6000, name="date"))


def px(start, end, scale=1.0):
    close = SERIES.loc[start:end] * scale
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1e6})


def client(bars_fn):
    return FakeClient(months={"AAPL": [("AAPL", "", 265598)]}, bars_fn=bars_fn)


def run(fake, tmp_path, years=10):
    return download_stock(fake, AAPL, years=years, directory=tmp_path, today=TODAY, pacer=Pacer(max_requests=10**6))


def stored(tmp_path):
    return normalize_bars(pd.read_parquet(bars_path("AAPL", tmp_path)))


def by_window(key, end, duration):
    """What IBKR would answer: the history up to `end`, however far back the duration goes."""
    stop = pd.Timestamp(end[:8]) if end else TODAY
    years = {"5 Y": 5}.get(duration)
    start = stop - pd.DateOffset(years=years) if years else stop - pd.Timedelta(days=int(duration.split()[0]) * {"D": 1, "M": 31, "Y": 366}[duration.split()[1]])
    return px(start, stop)


def test_first_download_walks_back_and_trims_to_the_window(tmp_path):
    result = run(client(by_window), tmp_path, years=10)
    bars = stored(tmp_path)
    assert result.action == "full" and result.bars == len(bars)
    assert bars.index.min() >= TODAY - pd.DateOffset(years=10) and bars.index.max() == TODAY


def test_an_up_to_date_ticker_costs_no_request(tmp_path):
    write_bars("AAPL", px("2019-01-01", "2025-06-30"), tmp_path)
    fake = client(by_window)
    result = run(fake, tmp_path, years=5)
    assert result.action == "skip" and not any(c[0] == "bars" for c in fake.calls)


def test_forward_update_appends_only_the_new_days(tmp_path):
    write_bars("AAPL", px("2019-01-01", "2025-06-20"), tmp_path)
    before = stored(tmp_path)
    fake = client(by_window)
    result = run(fake, tmp_path, years=5)
    after = stored(tmp_path)
    assert result.action == "forward" and result.added == len(pd.bdate_range("2025-06-23", "2025-06-30"))
    pd.testing.assert_frame_equal(after.loc[: before.index[-1]], before, check_freq=False)
    assert [c[4] for c in fake.calls if c[0] == "bars"] == ["12 D"]  # one small request


def test_a_split_triggers_a_full_redownload_not_a_spliced_series(tmp_path):
    write_bars("AAPL", px("2019-01-01", "2025-06-20"), tmp_path)
    halved = lambda key, end, duration: by_window(key, end, duration) * [0.5, 0.5, 0.5, 0.5, 1.0]
    result = run(client(halved), tmp_path, years=5)
    bars = stored(tmp_path)
    assert result.action == "refresh" and "split" in result.reason
    np.testing.assert_allclose(bars["close"].to_numpy(), (SERIES.loc[bars.index] * 0.5).to_numpy())


def test_backfill_adds_older_history_and_keeps_what_is_stored(tmp_path):
    write_bars("AAPL", px("2023-01-02", "2025-06-30"), tmp_path)
    before = stored(tmp_path)
    fake = client(by_window)
    result = run(fake, tmp_path, years=5)
    after = stored(tmp_path)
    first_request_end = next(c[2] for c in fake.calls if c[0] == "bars")
    assert first_request_end == "20230102-23:59:59"  # starts on the oldest stored day, so one bar is shared
    assert result.action == "backfill" and after.index.min() < before.index.min()
    pd.testing.assert_frame_equal(after.loc[before.index[0]:], before, check_freq=False)
    assert after.index.is_unique and after.index.is_monotonic_increasing


def test_an_unresolvable_row_raises_before_anything_is_requested(tmp_path):
    fake = FakeClient(bars_fn=by_window)  # no contracts known
    with pytest.raises(QualificationError):
        run(fake, tmp_path)
    assert not any(c[0] == "bars" for c in fake.calls) and not bars_path("AAPL", tmp_path).exists()


def test_impossible_bars_block_the_write(tmp_path):
    def broken(key, end, duration):
        bars = by_window(key, end, duration).copy()
        bars.iloc[-3, bars.columns.get_loc("high")] = -1e6
        return bars
    with pytest.raises(RuntimeError, match="data-quality"):
        run(client(broken), tmp_path)
    assert not bars_path("AAPL", tmp_path).exists()


def test_forward_with_no_new_bars_changes_nothing(tmp_path):
    write_bars("AAPL", px("2019-01-01", "2025-06-20"), tmp_path)
    original = bars_path("AAPL", tmp_path).read_bytes()
    result = run(client(lambda *_: px("2025-06-01", "2025-06-01").iloc[0:0]), tmp_path, years=5)
    assert result.added == 0 and bars_path("AAPL", tmp_path).read_bytes() == original
