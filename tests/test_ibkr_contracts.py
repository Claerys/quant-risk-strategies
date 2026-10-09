import pandas as pd
import pytest
from fake_gateway import FakeClient

from quant_risk.ibkr.contracts import ContractMonth, front_contract, futures_chain, load_futures_map, root_key
from quant_risk.ibkr.qualification import QualificationError
from quant_risk.instruments import get_instrument, load_instruments

ES = get_instrument("ES")


def test_every_instrument_has_a_complete_mapping():
    mapping = load_futures_map()
    assert set(mapping) == set(load_instruments())
    for symbol, row in mapping.items():
        assert row.ibkr_symbol and row.exchange and row.trading_class and row.price_scale > 0, symbol


def test_root_key_asks_for_expired_months_and_the_trading_class():
    key = root_key(get_instrument("TY"), include_expired=True)
    assert (key.symbol, key.exchange, key.trading_class, key.include_expired) == ("ZN", "CBOT", "ZN", True)


def test_chain_is_sorted_by_expiry():
    client = FakeClient(months={"ES": [("ESM5", "20250620", 2), ("ESH5", "20250321", 1)]})
    assert [m.local_symbol for m in futures_chain(client, ES)] == ["ESH5", "ESM5"]


def test_no_contracts_raises():
    with pytest.raises(QualificationError):
        futures_chain(FakeClient(), ES)


def test_duplicate_expiry_raises():
    client = FakeClient(months={"ES": [("ESM5", "20250620", 2), ("MESM5", "20250620", 3)]})
    with pytest.raises(ValueError, match="more than one trading class"):
        futures_chain(client, ES)


def test_multiplier_mismatch_raises():
    client = FakeClient(months={"ES": [("ESM5", "20250620", 2)]}, multiplier="5")
    with pytest.raises(ValueError, match="instruments.csv says 50"):
        futures_chain(client, ES)


def test_price_scale_makes_the_yen_multiplier_match():
    jy = get_instrument("JY")
    ok = FakeClient(months={"6J": [("6JM5", "20250616", 9)]}, multiplier="12500000")
    assert len(futures_chain(ok, jy)) == 1
    wrong = FakeClient(months={"6J": [("6JM5", "20250616", 9)]}, multiplier="125000")
    with pytest.raises(ValueError, match="price scale"):
        futures_chain(wrong, jy)


def test_partial_expiry_date_raises():
    client = FakeClient(months={"ES": [("ESM5", "202506", 2)]})
    with pytest.raises(ValueError, match="not a full date"):
        futures_chain(client, ES)


def test_unmapped_instrument_raises(monkeypatch):
    monkeypatch.setattr("quant_risk.ibkr.contracts.load_futures_map", dict)
    with pytest.raises(ValueError, match="no IBKR mapping"):
        futures_chain(FakeClient(), ES)


def test_front_contract_skips_months_about_to_expire():
    client = FakeClient(months={"ES": [("ESH5", "20250321", 1), ("ESM5", "20250620", 2)]})
    months = futures_chain(client, ES)
    assert front_contract(months, pd.Timestamp("2025-03-01")).local_symbol == "ESH5"
    assert front_contract(months, pd.Timestamp("2025-03-18")).local_symbol == "ESM5"
    with pytest.raises(ValueError):
        front_contract(months, pd.Timestamp("2025-06-19"))
    assert isinstance(months[0], ContractMonth)
