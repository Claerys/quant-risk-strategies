from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from quant_risk.sizing import ewma_covariance
from quant_risk.var import (
    RISKMETRICS_SPAN,
    historical_var_es,
    monte_carlo_var_es,
    parametric_var_es,
    risk_contributions,
    rolling_var,
    var_table,
)

DATES = pd.bdate_range("2020-01-01", periods=1000)


def normal_pnl(vols: dict[str, float], corr: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(vols)
    c = np.full((n, n), corr) + np.eye(n) * (1 - corr)
    draws = rng.multivariate_normal(np.zeros(n), c, size=len(DATES))
    return pd.DataFrame(draws * np.array(list(vols.values())), index=DATES, columns=list(vols))


def test_historical_var_and_es_on_a_known_distribution() -> None:
    scenarios = pd.Series(np.arange(-100, 0, dtype=float))  # 100 equally likely losses
    var, es = historical_var_es(scenarios, 0.95)
    assert var == 96.0  # the 5th worst of 100
    assert es == pytest.approx(np.mean([96, 97, 98, 99, 100]))


def test_es_is_never_below_var() -> None:
    scenarios = pd.Series(np.random.default_rng(1).standard_t(4, 5000))
    for level in (0.95, 0.99):
        var, es = historical_var_es(scenarios, level)
        assert es >= var


def test_parametric_var_of_one_position() -> None:
    covariance = np.array([[1_000.0**2]])
    var, es = parametric_var_es(pd.Series({"ES": 2.0}), covariance, 0.99)
    assert var == pytest.approx(2 * 1_000 * 2.3263, rel=1e-4)
    assert es == pytest.approx(2 * 1_000 * NormalDist().pdf(2.3263) / 0.01, rel=1e-3)


def test_hedged_book_has_less_var() -> None:
    covariance = np.array([[1.0, 0.9], [0.9, 1.0]]) * 1e6
    long_long = parametric_var_es(pd.Series({"ES": 1.0, "NQ": 1.0}), covariance, 0.99)[0]
    hedged = parametric_var_es(pd.Series({"ES": 1.0, "NQ": -1.0}), covariance, 0.99)[0]
    assert hedged < long_long / 3


def test_monte_carlo_normal_matches_parametric() -> None:
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]]) * 1e6
    positions = pd.Series({"ES": 1.0, "TY": -2.0})
    mc = monte_carlo_var_es(positions, covariance, 0.99, dof=None, simulations=400_000)[0]
    assert mc == pytest.approx(parametric_var_es(positions, covariance, 0.99)[0], rel=0.02)


def test_fat_tails_raise_99_es_more_than_95_var() -> None:
    covariance = np.array([[1e6]])
    positions = pd.Series({"ES": 1.0})
    normal_es = monte_carlo_var_es(positions, covariance, 0.99, dof=None)[1]
    fat_es = monte_carlo_var_es(positions, covariance, 0.99, dof=4)[1]
    assert fat_es > normal_es


def test_var_table_methods_are_consistent() -> None:
    pnl = normal_pnl({"ES": 1_500.0, "TY": 400.0, "CL": 1_200.0}, corr=0.3)
    table = var_table(pd.Series({"ES": 2.0, "TY": 5.0, "CL": -1.0}), pnl, window=1000)
    var99 = table["VaR 99.0%"]
    # On normal data the methods land in the same region; they differ by design, not by bug:
    # the parametric covariance is a short EWMA, and Monte Carlo uses fat (t) tails.
    assert var99.max() / var99.min() < 1.5
    assert var99["monte carlo (t, 5 dof)"] > var99["parametric (normal)"]
    for level in ("95.0%", "99.0%", "97.5%"):
        assert (table[f"ES {level}"] >= table[f"VaR {level}"]).all()


def test_risk_contributions_sum_to_var_and_show_hedges() -> None:
    pnl = normal_pnl({"ES": 1_500.0, "NQ": 1_800.0, "TY": 400.0}, corr=0.8)
    positions = pd.Series({"ES": 2.0, "NQ": -1.0, "TY": 3.0})
    contrib = risk_contributions(positions, pnl)
    cov = ewma_covariance(pnl, RISKMETRICS_SPAN)[-1]
    assert contrib["component VaR"].sum() == pytest.approx(parametric_var_es(positions, cov, 0.99)[0])
    assert contrib.loc["NQ", "component VaR"] < 0  # short NQ hedges long ES
    assert contrib["share of VaR"].sum() == pytest.approx(1.0)


def test_rolling_var_uses_only_the_past() -> None:
    pnl = normal_pnl({"ES": 1_000.0})
    positions = pd.DataFrame({"ES": 1.0}, index=DATES)
    full = rolling_var(positions, pnl, window=250)
    shocked = pnl.copy()
    shocked.iloc[600:] *= 10  # a future crisis
    assert rolling_var(positions, shocked, window=250).iloc[:600].equals(full.iloc[:600])
    assert full.iloc[:249].isna().all()


def test_rolling_parametric_var_tracks_volatility() -> None:
    pnl = normal_pnl({"ES": 1_000.0})
    pnl.iloc[500:] *= 3
    var = rolling_var(pd.DataFrame({"ES": 1.0}, index=DATES), pnl, method="parametric")
    assert var.iloc[900] / var.iloc[450] == pytest.approx(3.0, rel=0.35)
