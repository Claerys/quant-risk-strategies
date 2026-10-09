import csv
from pathlib import Path

import pandas as pd
import pytest

from quant_risk.ibkr.plan import duration_for_days, plan_download
from quant_risk.ibkr.tickers import load_tickers
from quant_risk.instruments import load_equities

CONFIG = Path(__file__).resolve().parents[1] / "config"


def write(tmp_path, text):
    path = tmp_path / "t.csv"
    path.write_text(text)
    return path


def test_defaults_are_a_smart_routed_us_stock(tmp_path):
    ticker = load_tickers(write(tmp_path, "symbol\naapl\n"))[0]
    assert (ticker.symbol, ticker.sec_type, ticker.exchange, ticker.currency, ticker.con_id) == (
        "AAPL", "STK", "SMART", "USD", 0)


def test_con_id_and_primary_exchange_are_kept_and_make_the_identity(tmp_path):
    ticker = load_tickers(write(tmp_path, "symbol,primary_exchange,con_id\nAAPL,nasdaq,265598\n"))[0]
    assert (ticker.primary_exchange, ticker.con_id, ticker.to_key().identity) == ("NASDAQ", 265598, ("conid", 265598))


@pytest.mark.parametrize("text, message", [
    ("name\nAAPL\n", "'symbol' column"),
    ("symbol\n\n,\n", "empty symbol"),
    ("symbol,sec_type\nES,FUT\n", "not supported"),
    ("symbol,con_id\nAAPL,abc\n", "whole number"),
    ("symbol,con_id\nAAPL,-4\n", "cannot be negative"),
    ("symbol\nAAPL\nAAPL\n", "listed twice"),
    ("symbol\n", "no tickers"),
])
def test_bad_files_are_refused(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_tickers(write(tmp_path, text))


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tickers(tmp_path / "nope.csv")


def test_the_bundled_ticker_lists_load():
    sizes = {name: len(load_tickers(CONFIG / f"{name}.csv"))
             for name in ("tickers_nasdaq100", "tickers_nasdaq100_seed", "tickers_sp500_seed")}
    assert sizes["tickers_nasdaq100"] > 90 and sizes["tickers_nasdaq100_seed"] >= 10


def test_the_sp500_seed_list_matches_the_stock_universe():
    seed = {r["symbol"]: r["primary_exchange"] for r in csv.DictReader((CONFIG / "tickers_sp500_seed.csv").open())}
    assert seed == {s: i.primary_exchange for s, i in load_equities().items()}


TODAY = pd.Timestamp("2025-06-30")
FIRST_OLD, LAST_FRESH = pd.Timestamp("2015-01-02"), pd.Timestamp("2025-06-27")


def test_plan_full_when_nothing_is_stored():
    assert plan_download(None, None, 0, 10, TODAY).action == "full"


def test_plan_forward_when_stale_pads_the_gap():
    plan = plan_download(FIRST_OLD, pd.Timestamp("2025-06-20"), 2500, 10, TODAY)
    assert (plan.action, plan.duration, plan.missing_days) == ("forward", "12 D", 10)


def test_plan_backfill_when_history_is_short_but_current():
    plan = plan_download(pd.Timestamp("2023-01-03"), LAST_FRESH, 600, 10, TODAY)
    assert plan.action == "backfill" and plan.missing_days > 2000


def test_plan_skip_when_covered_and_a_weekend_is_not_stale():
    assert plan_download(FIRST_OLD, LAST_FRESH, 2600, 10, TODAY).action == "skip"


def test_duration_strings():
    assert [duration_for_days(d) for d in (1, 60, 61, 365, 366, 800)] == ["1 D", "60 D", "3 M", "13 M", "2 Y", "3 Y"]
