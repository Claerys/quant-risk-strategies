import numpy as np
import pandas as pd
import pytest
from fake_gateway import FakeClient

from quant_risk.ibkr.broker import IbkrBroker
from quant_risk.ibkr.config import IbkrSettings
from quant_risk.ibkr.contract import OrderResult
from quant_risk.instruments import load_instruments
from quant_risk.paper import Fill

TODAY = pd.Timestamp("2025-03-03")
SETTINGS = IbkrSettings(account="DU1234567")
MONTHS = {"ES": [("ESH5", "20250321", 1), ("ESM5", "20250620", 2)],
          "ZN": [("ZNH5", "20250321", 7), ("ZNM5", "20250620", 8)]}


def broker(tmp_path, client=None, settings=SETTINGS):
    client = client or FakeClient(months=MONTHS, multiplier="50")
    return IbkrBroker(client, settings, state_path=tmp_path / "ibkr_state.json"), client


class MixedClient(FakeClient):
    """ES at multiplier 50, ZN (TY) at multiplier 1000, as IBKR reports them."""

    def candidates(self, key):
        self.multiplier = "1000" if key.symbol == "ZN" else "50"
        return super().candidates(key)


def test_refuses_a_live_port():
    live = IbkrSettings(account="DU1")
    object.__setattr__(live, "port", 4001)
    with pytest.raises(ValueError, match="4001"):
        IbkrBroker(FakeClient(), live)


@pytest.mark.parametrize("account", [None, "", "U1234567"])
def test_refuses_a_non_paper_account(account):
    with pytest.raises(ValueError, match="not a paper account"):
        IbkrBroker(FakeClient(), IbkrSettings(account=account))


def test_refuses_an_account_the_gateway_does_not_know():
    with pytest.raises(ValueError, match="connected accounts"):
        IbkrBroker(FakeClient(accounts=["DU999"]), SETTINGS)


def test_positions_are_summed_per_project_symbol_and_unknowns_are_ignored(tmp_path):
    client = FakeClient(months=MONTHS, held={"ESH5": 2.0, "ESM5": 1.0})
    b, _ = broker(tmp_path, client)
    assert b.positions().to_dict() == {"ES": 3.0}


def test_orders_go_to_the_front_month_and_fills_are_reported(tmp_path):
    client = MixedClient(months=MONTHS, fill_price=5000.0)
    b, _ = broker(tmp_path, client)
    fills = b.execute(pd.Series({"ES": 3.0, "TY": -2.0, "CL": 0.0}), pd.Series(dtype=float), TODAY)
    assert client.orders == [("ESH5", 3, "DU1234567"), ("ZNH5", -2, "DU1234567")]
    assert fills == [Fill("ES", 3.0, 5000.0, 0.0), Fill("TY", -2.0, 5000.0, 0.0)]
    assert b.last_date == TODAY


def test_a_failed_lookup_sends_no_orders_at_all(tmp_path):
    client = MixedClient(months={"ES": MONTHS["ES"]})  # no ZN contracts exist
    b, _ = broker(tmp_path, client)
    with pytest.raises(Exception, match="no IBKR contract"):
        b.execute(pd.Series({"ES": 3.0, "TY": 1.0}), pd.Series(dtype=float), TODAY)
    assert client.orders == [] and b.last_date is None


def test_partial_fill_is_reported_as_the_filled_quantity(tmp_path):
    client = FakeClient(months=MONTHS, multiplier="50", fills={"ESH5": 2.0})
    b, _ = broker(tmp_path, client)
    fills = b.execute(pd.Series({"ES": 5.0}), pd.Series(dtype=float), TODAY)
    assert fills == [Fill("ES", 2.0, 100.0, 0.0)]


def test_a_rejected_order_with_no_fill_returns_nothing(tmp_path):
    client = FakeClient(months=MONTHS, multiplier="50")
    client.place_market_order = lambda key, qty, account: OrderResult(0.0, 0.0, "Inactive")
    b, _ = broker(tmp_path, client)
    assert b.execute(pd.Series({"ES": 5.0}), pd.Series(dtype=float), TODAY) == []


def test_fractional_orders_are_refused_before_anything_is_sent(tmp_path):
    b, client = broker(tmp_path)
    with pytest.raises(ValueError, match="whole number"):
        b.execute(pd.Series({"ES": 1.5}), pd.Series(dtype=float), TODAY)
    assert client.orders == []


def test_zero_orders_send_nothing(tmp_path):
    b, client = broker(tmp_path)
    assert b.execute(pd.Series({"ES": 0.0}), pd.Series(dtype=float), TODAY) == []
    assert client.orders == [] and set(load_instruments()) >= {"ES", "TY"}


def test_the_full_cycle_runs_through_the_ibkr_broker_with_limits_applied(tmp_path):
    from test_cycle import DATES, LIMITS

    from quant_risk.cycle import run_cycle
    from quant_risk.paper import Journal

    rng = np.random.default_rng(5)
    frame = pd.DataFrame({
        "ES": 4000 + np.cumsum(rng.normal(2, 30, len(DATES))),
        "TY": 110 + np.cumsum(rng.normal(-0.01, 0.4, len(DATES))),
        "CL": 70 + np.cumsum(rng.normal(0.02, 1.5, len(DATES))),
    }, index=DATES)
    months = {"ES": [("ESZ99", "20991217", 1)], "ZN": [("ZNZ99", "20991217", 2)], "CL": [("CLZ99", "20991217", 3)]}
    client = FakeClient(months=months, multipliers={"ES": "50", "ZN": "1000", "CL": "1000"})
    b, _ = broker(tmp_path, client)
    journal = Journal(tmp_path / "journal.sqlite")
    day = frame.index[-1]
    first = run_cycle(day, frame, strategy="time_series_momentum", capital=10_000_000, limits=LIMITS, broker=b,
                      journal=journal, send_alert=False)
    held = b.positions().reindex(first.approved.index).fillna(0.0)
    assert held.equals(first.approved) and client.orders and len(journal.read("fills")) == len(client.orders)
    second = run_cycle(day, frame, strategy="time_series_momentum", capital=10_000_000, limits=LIMITS, broker=b,
                       journal=journal, send_alert=False)
    assert not (second.orders != 0).any()  # already at target: nothing to trade
    journal.close()


def test_connect_helper_returns_a_connected_broker_and_disconnects_on_refusal(tmp_path):
    from quant_risk.ibkr.broker import connect_ibkr_broker

    client = FakeClient()
    got_client, got_broker = connect_ibkr_broker(lambda: client, SETTINGS)
    assert got_client is client and client.connected_with is SETTINGS and isinstance(got_broker, IbkrBroker)
    other = FakeClient(accounts=["DU999"])
    with pytest.raises(ValueError, match="connected accounts"):
        connect_ibkr_broker(lambda: other, SETTINGS)
    assert other.disconnected
