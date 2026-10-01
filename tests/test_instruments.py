import pytest

from quant_risk.instruments import get_instrument, load_instruments


def test_universe_loads_with_positive_multipliers() -> None:
    instruments = load_instruments()
    assert len(instruments) == 35
    assert all(i.multiplier > 0 for i in instruments.values())


def test_pnl_uses_the_contract_multiplier() -> None:
    # ES: one index point is $50, so a 10-point rally on 2 contracts makes $1,000.
    assert get_instrument("ES").pnl(10.0, contracts=2) == 1_000.0
    # Corn is quoted in cents per bushel; 5,000 bushels means 1 cent is $50.
    assert get_instrument("C").pnl(1.0) == 50.0


def test_short_position_loses_when_price_rises() -> None:
    assert get_instrument("CL").pnl(1.0, contracts=-3) == -3_000.0


def test_notional() -> None:
    assert get_instrument("GC").notional(2_000.0) == 200_000.0


def test_eurex_contracts_are_in_euros() -> None:
    assert {get_instrument(s).currency for s in ("FDAX", "FESX", "FGBL")} == {"EUR"}


def test_unknown_symbol_has_a_helpful_error() -> None:
    with pytest.raises(KeyError, match="instruments.csv"):
        get_instrument("XYZ")
