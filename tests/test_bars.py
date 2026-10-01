import numpy as np
import pandas as pd
import pytest

from quant_risk.bars import (
    available_symbols,
    load_closes,
    normalize_bars,
    pnl_per_contract,
    read_bars,
    write_bars,
)


def make_bars(dates: list[str], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(dates),
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [100] * len(closes),
        }
    )


def test_roundtrip(tmp_path) -> None:
    write_bars("ES", make_bars(["2024-01-02", "2024-01-03"], [4700.0, 4710.0]), tmp_path)
    bars = read_bars("ES", tmp_path)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.index.name == "date"
    assert bars["close"].tolist() == [4700.0, 4710.0]
    assert available_symbols(tmp_path) == ["ES"]


def test_normalize_sorts_and_drops_duplicate_dates() -> None:
    raw = make_bars(["2024-01-03", "2024-01-02", "2024-01-03"], [2.0, 1.0, 3.0])
    bars = normalize_bars(raw)
    assert bars.index.is_monotonic_increasing
    assert bars["close"].tolist() == [1.0, 3.0]  # the later duplicate wins


def test_normalize_strips_intraday_timestamp() -> None:
    raw = make_bars(["2024-01-02 22:59:00"], [1.1])
    assert normalize_bars(raw).index[0] == pd.Timestamp("2024-01-02")


def test_missing_file_points_to_data_readme(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        read_bars("ES", tmp_path)


def test_closes_align_on_union_of_dates(tmp_path) -> None:
    write_bars("ES", make_bars(["2024-01-02", "2024-01-03"], [4700.0, 4710.0]), tmp_path)
    write_bars("FDAX", make_bars(["2024-01-02", "2024-01-04"], [16700.0, 16800.0]), tmp_path)
    closes = load_closes(["ES", "FDAX"], tmp_path)
    assert len(closes) == 3
    assert np.isnan(closes.loc["2024-01-04", "ES"])  # not forward-filled


def test_pnl_per_contract_uses_multiplier() -> None:
    closes = pd.DataFrame(
        {"ES": [4700.0, 4710.0], "CL": [70.0, 69.5]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    pnl = pnl_per_contract(closes)
    assert pnl.loc["2024-01-03", "ES"] == 500.0  # 10 points x $50
    assert pnl.loc["2024-01-03", "CL"] == -500.0  # -$0.50 x 1,000 barrels


def test_pnl_is_correct_on_negative_back_adjusted_prices() -> None:
    # A back-adjusted crude series can sit below zero. The dollar move is still right,
    # while a percentage "return" from -2.0 to -1.0 would be a meaningless -50%.
    closes = pd.DataFrame({"CL": [-2.0, -1.0]}, index=pd.to_datetime(["2001-01-02", "2001-01-03"]))
    assert pnl_per_contract(closes).iloc[0]["CL"] == 1_000.0
