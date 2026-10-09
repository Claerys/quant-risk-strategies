import pytest

from quant_risk.ibkr.config import PAPER_PORTS, IbkrSettings, require_paper_port


def test_defaults_are_gateway_paper():
    settings = IbkrSettings.from_env({})
    assert (settings.host, settings.port, settings.client_id) == ("127.0.0.1", 4002, 1)
    assert settings.account is None


def test_env_overrides():
    settings = IbkrSettings.from_env({"IBKR_PORT": "7497", "IBKR_CLIENT_ID": "7", "IBKR_ACCOUNT": "DU123"})
    assert (settings.port, settings.client_id, settings.account) == (7497, 7, "DU123")


@pytest.mark.parametrize("port", [4001, 7496, 1234])
def test_live_or_unknown_ports_refused(port):
    with pytest.raises(ValueError, match=str(port)):
        require_paper_port(port)
    with pytest.raises(ValueError, match=str(port)):
        IbkrSettings.from_env({"IBKR_PORT": str(port)})


def test_paper_ports_accepted():
    for port in PAPER_PORTS:
        require_paper_port(port)
