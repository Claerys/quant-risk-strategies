import numpy as np
import pandas as pd
import pytest
from fake_gateway import FakeClient

from quant_risk.bars import bars_path, normalize_bars
from quant_risk.continuous import ContractBars, back_adjust
from quant_risk.ibkr.download import download_fx, download_symbol
from quant_risk.ibkr.pacing import Pacer
from quant_risk.instruments import get_instrument

ES = get_instrument("ES")


def bars(start, end, base, volume=1000.0, seed=0):
    """A seeded random walk: the same start and seed give the same prefix whatever the end."""
    index = pd.bdate_range(start, end, name="date")
    steps = np.random.default_rng(seed).normal(0, 2, len(index))
    close = pd.Series(base + steps.cumsum(), index=index)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": volume})


def pacer():
    return Pacer(max_requests=10_000)


MONTHS = {"ES": [("ESZ4", "20241220", 4), ("ESH5", "20250321", 1), ("ESM5", "20250620", 2),
                 ("ESU5", "20250919", 3)]}


def client(second_run=False):
    esm5_end = "2025-04-30" if second_run else "2025-03-31"
    return FakeClient(months=dict(MONTHS), bars={
        "ESZ4": bars("2024-10-01", "2024-12-20", 90, 3000.0, seed=1),
        "ESH5": bars("2024-12-02", "2025-03-21", 100, 2000.0, seed=2),
        "ESM5": bars("2025-01-02", esm5_end, 105, 1000.0, seed=3),
    })


def first_run(tmp_path):
    return download_symbol(client(), ES, years=1, directory=tmp_path, today=pd.Timestamp("2025-03-31"), pacer=pacer())


def test_full_download_equals_back_adjust_of_the_same_bars(tmp_path):
    result = first_run(tmp_path)
    fake = client()
    expected, _ = back_adjust([ContractBars(local, pd.Timestamp(exp), fake.bars[local])
                               for local, exp, _ in MONTHS["ES"][:3]])
    written = normalize_bars(pd.read_parquet(bars_path("ES", tmp_path)))
    pd.testing.assert_frame_equal(written, expected.loc[expected.index >= pd.Timestamp("2024-03-31")],
                                  check_freq=False)
    assert (result.mode, result.months, result.bars) == ("full", 3, len(written))


def test_future_month_without_bars_is_skipped_but_expired_one_is_an_error(tmp_path):
    first_run(tmp_path)  # ESU5 has no bars and has not expired: skipped silently
    broken = client()
    broken.bars["ESH5"] = broken.bars["ESH5"].iloc[0:0]
    with pytest.raises(RuntimeError, match="ESH5"):
        download_symbol(broken, ES, years=1, directory=tmp_path / "b", today=pd.Timestamp("2025-03-31"),
                        pacer=pacer())
    assert not bars_path("ES", tmp_path / "b").exists()


def test_incremental_appends_only_new_days_and_skips_old_contracts(tmp_path):
    first_run(tmp_path)
    before = normalize_bars(pd.read_parquet(bars_path("ES", tmp_path)))
    second = client(second_run=True)
    result = download_symbol(second, ES, years=1, directory=tmp_path, today=pd.Timestamp("2025-05-01"),
                             pacer=pacer())
    after = normalize_bars(pd.read_parquet(bars_path("ES", tmp_path)))
    assert result.mode == "incremental" and result.added == len(pd.bdate_range("2025-04-01", "2025-04-30"))
    pd.testing.assert_frame_equal(after.loc[: before.index[-1]], before, check_freq=False)
    requested = {call[1] for call in second.calls if call[0] == "bars"}
    assert "ESZ4" not in requested and {"ESH5", "ESM5"} <= requested


def test_incremental_refuses_data_that_disagrees_with_the_archive(tmp_path):
    first_run(tmp_path)
    path = bars_path("ES", tmp_path)
    original = path.read_bytes()
    second = client(second_run=True)
    second.bars["ESM5"] = bars("2025-01-02", "2025-04-30", 105, 1000.0, seed=99)  # a different market
    with pytest.raises(RuntimeError, match="disagrees with the archive"):
        download_symbol(second, ES, years=1, directory=tmp_path, today=pd.Timestamp("2025-05-01"), pacer=pacer())
    assert path.read_bytes() == original


def test_impossible_bars_block_the_write(tmp_path):
    bad = client()
    bad.bars["ESM5"] = bad.bars["ESM5"].copy()
    bad.bars["ESM5"].iloc[-2, bad.bars["ESM5"].columns.get_loc("high")] = -1e6  # high far below low
    with pytest.raises(RuntimeError, match="data-quality"):
        download_symbol(bad, ES, years=1, directory=tmp_path, today=pd.Timestamp("2025-03-31"), pacer=pacer())
    assert not bars_path("ES", tmp_path).exists()


def test_price_scale_is_applied_for_the_yen(tmp_path):
    jy = get_instrument("JY")
    raw = bars("2025-01-02", "2025-03-31", 0.0064)
    fake = FakeClient(months={"6J": [("6JM5", "20250616", 9)]}, bars={"6JM5": raw}, multiplier="12500000")
    download_symbol(fake, jy, years=1, directory=tmp_path, today=pd.Timestamp("2025-03-31"), pacer=pacer())
    written = normalize_bars(pd.read_parquet(bars_path("JY", tmp_path)))
    np.testing.assert_allclose(written["close"].to_numpy(), raw["close"].to_numpy() * 100)


def test_download_fx_writes_a_daily_series(tmp_path):
    fx = bars("2020-01-01", "2025-03-31", 1.1, 0.0)
    fake = FakeClient(months={"EUR": [("EUR.USD", "", 5)]}, bars={"EUR.USD": fx})
    days = download_fx(fake, years=20, directory=tmp_path, today=pd.Timestamp("2025-03-31"), pacer=pacer())
    assert days == len(fx)
    assert any(c[:4] == ("bars", "EUR.USD", "", "MIDPOINT") for c in fake.calls)
    assert bars_path("EURUSD", tmp_path).exists()
