import numpy as np
import pandas as pd
import pytest

from quant_risk.bars import write_bars

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

DATES = pd.bdate_range("2019-01-01", periods=900)


@pytest.fixture(autouse=True)
def synthetic_market(tmp_path, monkeypatch):
    monkeypatch.setenv("QRS_DATA_DIR", str(tmp_path / "daily"))
    rng = np.random.default_rng(11)
    for symbol, start, vol, drift in [("ES", 4000, 30, 1.5), ("TY", 110, 0.4, -0.02), ("CL", 70, 1.5, 0.03)]:
        close = start + np.cumsum(rng.normal(drift, vol, len(DATES)))
        bars = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000}, index=DATES)
        write_bars(symbol, bars, tmp_path / "daily")


def test_dashboard_renders_every_tab_without_errors() -> None:
    app = AppTest.from_file("../app/dashboard.py", default_timeout=120).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "Book & limits", "VaR & ES", "VaR backtest", "Stress & climate", "Performance", "Paper trading",
    ]
    assert app.metric[1].label == "VaR 99%"
    app.selectbox[0].set_value("multi_horizon_trend").run()
    assert not app.exception
