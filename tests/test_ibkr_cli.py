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
