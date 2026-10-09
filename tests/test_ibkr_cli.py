import pandas as pd
from fake_gateway import FakeClient
from test_ibkr_download import bars

from quant_risk.bars import bars_path
from quant_risk.ibkr.cli import main

ES_MONTHS = [("ESH5", "20990321", 1), ("ESM5", "20990620", 2)]  # far future: always a valid front month


def fake(**kwargs):
    return FakeClient(months={"ES": ES_MONTHS}, **kwargs)


def test_dry_run_qualifies_and_writes_nothing(tmp_path, capsys):
    client = fake()
    assert main(["--symbols", "ES", "--dry-run", "--out", str(tmp_path)], lambda: client) == 0
    assert "ESH5" in capsys.readouterr().out
    assert not any(call[0] == "bars" for call in client.calls)
    assert list(tmp_path.iterdir()) == [] and client.disconnected


def test_unknown_symbol_exits_before_connecting(capsys):
    def never():
        raise AssertionError("connected")
    assert main(["--symbols", "NOPE"], never) == 1
    assert "unknown symbols" in capsys.readouterr().err


def test_live_port_in_environment_exits_before_connecting(monkeypatch, capsys):
    monkeypatch.setenv("IBKR_PORT", "4001")

    def never():
        raise AssertionError("connected")
    assert main(["--symbols", "ES"], never) == 1
    assert "4001" in capsys.readouterr().err


def test_connection_failure_exits_with_the_port(monkeypatch, capsys):
    monkeypatch.delenv("IBKR_PORT", raising=False)
    client = fake(connect_error=TimeoutError("timed out connecting to IBKR at 127.0.0.1:4002"))
    assert main(["--symbols", "ES"], lambda: client) == 1
    assert "127.0.0.1:4002" in capsys.readouterr().err


def test_one_failing_symbol_does_not_stop_the_others(tmp_path, capsys):
    today = pd.Timestamp.today().normalize()
    start = (today - pd.Timedelta(days=200)).strftime("%Y-%m-%d")
    ok = bars(start, today.strftime("%Y-%m-%d"), 100.0)
    expiry = (today + pd.Timedelta(days=100)).strftime("%Y%m%d")
    client = FakeClient(months={"ES": [("ESM9", expiry, 2)], "NQ": []}, bars={"ESM9": ok}, multiplier="50")
    code = main(["--symbols", "NQ", "ES", "--years", "1", "--out", str(tmp_path)], lambda: client)
    captured = capsys.readouterr()
    assert code == 1 and "FAILED NQ" in captured.err
    assert bars_path("ES", tmp_path).exists() and "full" in captured.out


STOCK_FILE = "symbol,primary_exchange\nAAPL,NASDAQ\nMSFT,NASDAQ\nXXXX,NYSE\n"


def stock_client():
    today = pd.Timestamp.today().normalize()
    series = bars((today - pd.Timedelta(days=400)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"), 100.0)
    return FakeClient(months={"AAPL": [("AAPL", "", 1)], "MSFT": [("MSFT", "", 2)]},
                      bars_fn=lambda key, end, duration: series)


def test_ticker_file_downloads_stocks_and_skips_what_ibkr_cannot_resolve(tmp_path, capsys):
    tickers = tmp_path / "t.csv"
    tickers.write_text(STOCK_FILE)
    out = tmp_path / "stocks"
    code = main(["--tickers-file", str(tickers), "--years", "1", "--out", str(out)], lambda: stock_client())
    captured = capsys.readouterr()
    assert code == 0  # a skipped row is reported, it does not fail the run
    assert "SKIPPED XXXX" in captured.err and "skipped" in captured.err
    assert (out / "AAPL.parquet").exists() and (out / "MSFT.parquet").exists() and not (out / "XXXX.parquet").exists()
    assert "full" in captured.out


def test_second_run_over_current_data_requests_nothing(tmp_path, capsys):
    tickers = tmp_path / "t.csv"
    tickers.write_text("symbol\nAAPL\n")
    out = tmp_path / "stocks"
    main(["--tickers-file", str(tickers), "--years", "1", "--out", str(out)], lambda: stock_client())
    again = stock_client()
    assert main(["--tickers-file", str(tickers), "--years", "1", "--out", str(out)], lambda: again) == 0
    assert "skip" in capsys.readouterr().out and not any(c[0] == "bars" for c in again.calls)


def test_stock_dry_run_qualifies_and_writes_nothing(tmp_path, capsys):
    tickers = tmp_path / "t.csv"
    tickers.write_text(STOCK_FILE)
    out = tmp_path / "stocks"
    fake = stock_client()
    assert main(["--tickers-file", str(tickers), "--dry-run", "--out", str(out)], lambda: fake) == 0
    assert "conId 1" in capsys.readouterr().out and not out.exists()
    assert not any(c[0] == "bars" for c in fake.calls)


def test_bad_ticker_file_and_unknown_symbol_exit_before_connecting(tmp_path, capsys):
    def never():
        raise AssertionError("connected")
    assert main(["--tickers-file", str(tmp_path / "missing.csv")], never) == 1
    good = tmp_path / "t.csv"
    good.write_text("symbol\nAAPL\n")
    assert main(["--tickers-file", str(good), "--symbols", "ZZZ"], never) == 1
    assert "unknown symbols" in capsys.readouterr().err
