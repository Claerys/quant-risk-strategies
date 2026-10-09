import pytest

pytest.importorskip("ibapi")
from fake_gateway import FakeTransport, make_contract, make_details

from quant_risk.ibkr.contract import ContractKey
from quant_risk.ibkr.qualification import (
    ContractDetailsCollector,
    ContractQualifier,
    QualificationError,
    QualificationStatus,
)

KEY = ContractKey(symbol="ES", exchange="CME")


def run(script, timeout=0.2, connected=True):
    collector = ContractDetailsCollector()
    transport = FakeTransport(collector, script, connected=connected)
    return ContractQualifier(transport, timeout, collector), collector, transport


def answer(*details):
    def script(transport, req_id, _contract):
        for d in details:
            transport.collector.on_details(req_id, d)
        transport.collector.on_end(req_id)
    return script


def test_one_candidate_is_qualified():
    qualifier, collector, _ = run(answer(make_details(con_id=11)))
    assert qualifier.qualify(KEY).con_id == 11
    assert collector.in_flight == 0


def test_zero_candidates_is_not_found_and_raises_on_qualify():
    qualifier, _, _ = run(answer())
    assert qualifier.lookup(KEY).status is QualificationStatus.NOT_FOUND
    with pytest.raises(QualificationError, match="no IBKR contract"):
        qualifier.qualify(KEY)


def test_many_candidates_is_ambiguous_but_candidates_returns_all_months():
    qualifier, _, _ = run(answer(make_details(con_id=1), make_details(con_id=2, expiry="20250919")))
    assert qualifier.lookup(KEY).status is QualificationStatus.AMBIGUOUS
    with pytest.raises(QualificationError):
        qualifier.qualify(KEY)
    assert [c.con_id for c in qualifier.candidates(KEY)] == [1, 2]


def test_duplicate_callback_does_not_create_an_ambiguity():
    qualifier, _, _ = run(answer(make_details(con_id=5), make_details(con_id=5)))
    assert qualifier.qualify(KEY).con_id == 5


def test_error_200_is_not_found_other_errors_are_errors():
    def err(code):
        return lambda t, req_id, _c: t.collector.on_error(req_id, code, f"code {code}")
    assert run(err(200))[0].lookup(KEY).status is QualificationStatus.NOT_FOUND
    assert run(err(321))[0].lookup(KEY).status is QualificationStatus.ERROR


def test_timeout_cancels_and_cleans_up():
    qualifier, collector, transport = run(None, timeout=0.05)
    assert qualifier.lookup(KEY).status is QualificationStatus.TIMEOUT
    assert transport.cancelled and collector.in_flight == 0


def test_disconnect_midway_and_not_connected():
    result = run(lambda t, *_: t.collector.on_disconnect())[0].lookup(KEY)
    assert result.status is QualificationStatus.DISCONNECTED
    assert run(None, connected=False)[0].lookup(KEY).status is QualificationStatus.DISCONNECTED


def test_late_callback_for_another_request_is_ignored():
    def script(transport, req_id, _contract):
        transport.collector.on_details(req_id + 50, make_details(con_id=99))  # not ours
        transport.collector.on_details(req_id, make_details(con_id=7))
        transport.collector.on_end(req_id)
    assert run(script)[0].qualify(KEY).con_id == 7


def test_key_identity_prefers_con_id():
    assert ContractKey("ES", local_symbol="ESM5", con_id=3) == ContractKey("NQ", local_symbol="X", con_id=3)
    assert ContractKey("ES", expiry="20250620") != ContractKey("ES", expiry="20250919")
    assert make_contract().conId == 1
