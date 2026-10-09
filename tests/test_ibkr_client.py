import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("ibapi")

from ibapi.client import EClient

from quant_risk.bars import normalize_bars
from quant_risk.ibkr.client import IbkrClient, IbkrRequestError
from quant_risk.ibkr.config import IbkrSettings
from quant_risk.ibkr.contract import ContractKey

KEY = ContractKey(symbol="ES", exchange="CME", local_symbol="ESM5", con_id=1)


def no_socket(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("socket opened")
    monkeypatch.setattr(EClient, "connect", boom)


def test_live_port_refused_before_any_socket(monkeypatch):
    no_socket(monkeypatch)
    client = IbkrClient()
    object.__setattr__(settings := IbkrSettings(), "port", 4001)  # bypass the settings check on purpose
    with pytest.raises(ValueError, match="4001"):
        client.connect_and_start(settings)


def test_connect_timeout_names_the_port(monkeypatch):
    monkeypatch.setattr(EClient, "connect", lambda *a, **k: None)
    monkeypatch.setattr(IbkrClient, "run", lambda self: None)
    monkeypatch.setattr(IbkrClient, "isConnected", lambda self: False)
    with pytest.raises(TimeoutError, match="127.0.0.1:4002"):
        IbkrClient().connect_and_start(IbkrSettings(timeout=0.05))


def pending_bars(client, req_id=7):
    event = threading.Event()
    client._done[req_id], client._bars[req_id] = event, []
    return event


def test_error_code_is_recorded_and_wakes_the_request():
    client = IbkrClient()
    event = pending_bars(client)
    client.error(7, 162, "pacing violation")
    assert event.is_set() and "162" in client._errors[7]


def test_harmless_notices_are_ignored():
    client = IbkrClient()
    event = pending_bars(client)
    client.error(7, 2104, "market data farm connection is OK")
    assert not event.is_set() and 7 not in client._errors


def test_historical_bars_parsed_like_normalize_bars(monkeypatch):
    client = IbkrClient(request_timeout=2)
    monkeypatch.setattr(client, "reqHistoricalData", lambda req_id, *a: [
        client.historicalData(req_id, SimpleNamespace(date="20250102", open=1, high=2, low=0.5, close=1.5, volume=9)),
        client.historicalData(req_id, SimpleNamespace(date="20250103", open=1.5, high=3, low=1, close=2, volume=7)),
        client.historicalDataEnd(req_id, "", ""),
    ])
    bars = client.historical_bars(KEY, end="", duration="1 Y")
    expected = normalize_bars(bars.reset_index())
    assert bars.equals(expected) and list(bars.index.strftime("%Y%m%d")) == ["20250102", "20250103"]


def test_empty_response_is_an_empty_frame(monkeypatch):
    client = IbkrClient(request_timeout=2)
    monkeypatch.setattr(client, "reqHistoricalData", lambda req_id, *a: client.historicalDataEnd(req_id, "", ""))
    assert client.historical_bars(KEY, end="", duration="1 Y").empty


def test_request_error_is_raised(monkeypatch):
    client = IbkrClient(request_timeout=2)
    monkeypatch.setattr(client, "reqHistoricalData", lambda req_id, *a: client.error(req_id, 162, "no permissions"))
    with pytest.raises(IbkrRequestError, match="162"):
        client.historical_bars(KEY, end="", duration="1 Y")


def test_order_reports_only_what_filled(monkeypatch):
    client = IbkrClient(request_timeout=2)

    def place(order_id, contract, order):
        client.orderStatus(order_id, "Filled", 2.0, 1.0, 5000.25, 0, 0, 5000.25, 1, "")
    monkeypatch.setattr(client, "placeOrder", place)
    result = client.place_market_order(KEY, 3, "DU1")
    assert (result.filled, result.avg_price) == (2.0, 5000.25)
