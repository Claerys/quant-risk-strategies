"""Value-at-Risk and Expected Shortfall of a futures portfolio, three ways.

**VaR at 99%** is the loss that should be exceeded on only 1 day in 100. **Expected Shortfall (ES)**
is the average loss on those worst days: it answers "when it goes wrong, how badly?", which VaR
does not. Basel's market-risk framework (FRTB) moved from 99% VaR to 97.5% ES for that reason.

All numbers are 1-day, in USD, reported as positive losses. A futures position's P&L is linear in
the price change (contracts x change x multiplier x FX), so a portfolio scenario is just the
held contracts applied to one day's per-contract P&L, with no repricing model needed.

Methods:
    historical   replay the last `window` days of per-contract P&L on today's positions.
                 No distribution assumed; only as good as the window contains bad days.
    parametric   assume normally distributed P&L with an EWMA covariance matrix (RiskMetrics,
                 lambda = 0.94). Fast and reacts quickly to volatility, but normal tails are thin.
    monte carlo  simulate correlated scenarios from the same covariance, with Student-t tails
                 (5 degrees of freedom by default) to allow for fat tails.
    filtered     filtered historical simulation: each historical day's P&L is divided by the
    historical   volatility forecast for that day and multiplied by today's, instrument by
                 instrument. Keeps the real fat-tailed, correlated shapes of history but at
                 today's volatility, so a calm two-year window cannot hide a volatile market.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from quant_risk.instruments import get_instrument
from quant_risk.sizing import ewma_covariance

RISKMETRICS_SPAN = 32  # lambda = 1 - 2 / (span + 1) = 0.94


def portfolio_scenarios(positions: pd.Series, pnl_per_contract: pd.DataFrame, window: int = 500) -> pd.Series:
    """Hypothetical daily P&L of today's positions over the last `window` days of history."""
    history = pnl_per_contract.reindex(columns=positions.index).iloc[-window:].fillna(0.0)
    return history @ positions.fillna(0.0)


def ewma_vol_forecast(pnl_per_contract: pd.DataFrame, span: int = RISKMETRICS_SPAN) -> pd.DataFrame:
    """Per-instrument volatility forecast: the value on day t uses P&L up to and including day t."""
    return np.sqrt((pnl_per_contract.fillna(0.0) ** 2).ewm(span=span, adjust=False).mean())


def filtered_scenarios(
    positions: pd.Series, pnl_per_contract: pd.DataFrame, window: int = 500
) -> pd.Series:
    """Hypothetical P&L of today's positions with history rescaled to today's volatility."""
    pnl = pnl_per_contract.reindex(columns=positions.index).fillna(0.0)
    vol = ewma_vol_forecast(pnl)
    standardised = (pnl / vol.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rescaled = standardised.iloc[-window:] * vol.iloc[-1]
    return rescaled @ positions.fillna(0.0)


def historical_var_es(scenarios: pd.Series, level: float) -> tuple[float, float]:
    """VaR and ES from the empirical distribution of scenario P&L."""
    var = -float(np.quantile(scenarios, 1.0 - level, method="lower"))
    tail = scenarios[scenarios <= -var]
    return var, -float(tail.mean())


def parametric_var_es(positions: pd.Series, covariance: np.ndarray, level: float) -> tuple[float, float]:
    """Normal VaR = z * sigma; normal ES = sigma * pdf(z) / (1 - level)."""
    x = positions.fillna(0.0).to_numpy()
    sigma = float(np.sqrt(max(x @ covariance @ x, 0.0)))
    z = NormalDist().inv_cdf(level)
    return z * sigma, sigma * NormalDist().pdf(z) / (1.0 - level)


def monte_carlo_var_es(
    positions: pd.Series,
    covariance: np.ndarray,
    level: float,
    *,
    simulations: int = 100_000,
    dof: float | None = 5.0,
    seed: int = 7,
) -> tuple[float, float]:
    """VaR and ES from simulated correlated P&L; Student-t with `dof` degrees of freedom, or normal if None.

    The t draws are scaled to have the same covariance as the normal case, so the two differ only in
    the shape of the tails, not in volatility.
    """
    rng = np.random.default_rng(seed)
    x = positions.fillna(0.0).to_numpy()
    n = len(x)
    # Cholesky can fail on a singular matrix (an instrument with no history); eigen-decompose instead.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    shocks = rng.standard_normal((simulations, n))
    if dof is not None:
        chi = rng.chisquare(dof, size=(simulations, 1))
        shocks = shocks / np.sqrt(chi / dof) * np.sqrt((dof - 2.0) / dof)
    pnl = shocks @ root.T @ x
    return historical_var_es(pd.Series(pnl), level)


def var_table(
    positions: pd.Series,
    pnl_per_contract: pd.DataFrame,
    *,
    levels: tuple[float, ...] = (0.95, 0.99, 0.975),
    window: int = 500,
) -> pd.DataFrame:
    """VaR and ES by method and confidence level for today's positions, in USD."""
    pnl = pnl_per_contract.reindex(columns=positions.index).fillna(0.0)
    covariance = ewma_covariance(pnl, RISKMETRICS_SPAN)[-1]
    scenarios = portfolio_scenarios(positions, pnl, window)
    filtered = filtered_scenarios(positions, pnl, window)
    rows = {}
    for name, fn in {
        "historical": lambda lv: historical_var_es(scenarios, lv),
        "filtered historical": lambda lv: historical_var_es(filtered, lv),
        "parametric (normal)": lambda lv: parametric_var_es(positions, covariance, lv),
        "monte carlo (t, 5 dof)": lambda lv: monte_carlo_var_es(positions, covariance, lv),
    }.items():
        row = {}
        for level in levels:
            var, es = fn(level)
            row[f"VaR {level:.1%}"] = var
            row[f"ES {level:.1%}"] = es
        rows[name] = row
    return pd.DataFrame(rows).T


def risk_contributions(positions: pd.Series, pnl_per_contract: pd.DataFrame, level: float = 0.99) -> pd.DataFrame:
    """How much each position adds to parametric VaR (Euler allocation). Contributions sum to VaR.

    component_i = x_i * (S x)_i / sigma * z. A negative contribution is a hedge: that position
    lowers the portfolio's VaR.
    """
    pnl = pnl_per_contract.reindex(columns=positions.index).fillna(0.0)
    covariance = ewma_covariance(pnl, RISKMETRICS_SPAN)[-1]
    x = positions.fillna(0.0).to_numpy()
    sigma = float(np.sqrt(max(x @ covariance @ x, 0.0)))
    z = NormalDist().inv_cdf(level)
    marginal = covariance @ x / sigma if sigma > 0 else np.zeros_like(x)
    component = x * marginal * z
    out = pd.DataFrame({
        "contracts": x,
        "standalone VaR": z * np.sqrt(np.clip(np.diag(covariance), 0, None)) * np.abs(x),
        "component VaR": component,
        "share of VaR": component / component.sum() if component.sum() else 0.0,
    }, index=positions.index)
    out["sector"] = [get_instrument(s).sector for s in out.index]
    return out.sort_values("component VaR", ascending=False)


def rolling_var(
    positions: pd.DataFrame,
    pnl_per_contract: pd.DataFrame,
    *,
    level: float = 0.99,
    method: str = "historical",
    window: int = 500,
) -> pd.Series:
    """The VaR forecast made at each close for the next day's P&L of the positions held then.

    Only data up to and including that close is used, so the series can be backtested honestly.
    """
    frame = pnl_per_contract.reindex(index=positions.index, columns=positions.columns).fillna(0.0)
    pnl = frame.to_numpy()
    x = positions.fillna(0.0).to_numpy()
    out = np.full(len(x), np.nan)
    # Same order statistic as np.quantile(..., method="lower") in historical_var_es.
    k = int(np.floor((1.0 - level) * (window - 1)))
    if method == "historical":
        for t in range(window - 1, len(x)):
            scenarios = pnl[t - window + 1 : t + 1] @ x[t]
            out[t] = -np.partition(scenarios, k)[k]
    elif method == "filtered":
        vol = ewma_vol_forecast(frame).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.nan_to_num(pnl / np.vstack([np.full(pnl.shape[1], np.nan), vol[:-1]]), posinf=0.0, neginf=0.0)
        for t in range(window - 1, len(x)):
            scenarios = (z[t - window + 1 : t + 1] * vol[t]) @ x[t]
            out[t] = -np.partition(scenarios, k)[k]
    elif method == "parametric":
        z = NormalDist().inv_cdf(level)
        covariance = ewma_covariance(pd.DataFrame(pnl), RISKMETRICS_SPAN)
        variance = np.einsum("ti,tij,tj->t", x, covariance, x)
        out = z * np.sqrt(np.clip(variance, 0.0, None))
    else:
        raise ValueError(f"unknown method {method!r}; use 'historical', 'filtered' or 'parametric'")
    return pd.Series(out, index=positions.index, name=f"{method} VaR {level:.0%}")
