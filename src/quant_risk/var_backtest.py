"""Backtesting a VaR model: did losses exceed VaR as often as the model promised?

A 99% VaR should be exceeded on about 1% of days. Too many exceptions and the model understates
risk (capital is too low); too few and it overstates it (capital is wasted). Three standard tests:

**Kupiec proportion-of-failures (POF).** Is the exception rate consistent with 1 - level?
    LR = -2 ln[(1-p)^(n-x) p^x] + 2 ln[(1-x/n)^(n-x) (x/n)^x],  ~ chi-squared(1)

**Christoffersen independence.** Do exceptions cluster? A model can get the count right and still
fail if all exceptions arrive together in a crisis, which is when it matters most.

**Basel traffic light.** The supervisory test on the last 250 days of 99% VaR:
    green 0-4 exceptions, yellow 5-9, red 10 or more.

The P&L compared with each day's VaR is the *hypothetical* P&L: the positions held at the close on
which the VaR was computed, revalued at the next close. Trading during the day does not count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

BASEL_WINDOW = 250


def exceptions(var_forecast: pd.Series, next_day_pnl: pd.Series) -> pd.Series:
    """True on each day where the P&L was a bigger loss than the VaR forecast made the day before."""
    var_prev = var_forecast.shift(1)
    valid = var_prev.notna() & next_day_pnl.notna()
    return (next_day_pnl < -var_prev).where(valid)


def _chi2_1_pvalue(lr: float) -> float:
    """P(chi-squared with 1 degree of freedom > lr), using the normal distribution identity."""
    return math.erfc(math.sqrt(max(lr, 0.0) / 2.0))


def _log_likelihood(count: int, total: int, p: float) -> float:
    def xlogy(x: float, y: float) -> float:
        return 0.0 if x == 0 else x * math.log(y)

    return xlogy(total - count, 1.0 - p) + xlogy(count, p)


def kupiec_pof(n_exceptions: int, n_days: int, level: float) -> tuple[float, float]:
    """Kupiec likelihood ratio and its p-value. A p-value below 0.05 rejects the model."""
    p = 1.0 - level
    observed = n_exceptions / n_days
    lr = -2.0 * (_log_likelihood(n_exceptions, n_days, p) - _log_likelihood(n_exceptions, n_days, observed))
    return lr, _chi2_1_pvalue(lr)


def christoffersen_independence(hits: pd.Series) -> tuple[float, float]:
    """LR test that an exception today does not make one tomorrow more likely."""
    h = hits.dropna().astype(int).to_numpy()
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / len(curr) if len(curr) else 0.0
    unrestricted = _log_likelihood(n01, n00 + n01, pi0) + _log_likelihood(n11, n10 + n11, pi1)
    restricted = _log_likelihood(n01 + n11, len(curr), pi)
    lr = -2.0 * (restricted - unrestricted)
    return lr, _chi2_1_pvalue(lr)


def basel_zone(n_exceptions: int) -> str:
    """Traffic-light zone for exceptions of a 99% VaR over 250 days."""
    if n_exceptions <= 4:
        return "green"
    if n_exceptions <= 9:
        return "yellow"
    return "red"


@dataclass
class VarBacktest:
    level: float
    days: int
    exceptions: int
    expected: float
    kupiec_lr: float
    kupiec_p: float
    independence_lr: float
    independence_p: float
    last_250_exceptions: int
    basel_zone: str
    worst_excess: float  # largest loss beyond VaR, in USD
    hits: pd.Series

    @property
    def verdict(self) -> str:
        if self.kupiec_p < 0.05:
            side = "understates" if self.exceptions > self.expected else "overstates"
            return f"rejected: the model {side} risk"
        if self.independence_p < 0.05:
            return "count is fine but exceptions cluster"
        return "not rejected"

    def summary(self) -> dict:
        return {
            "days": self.days,
            "exceptions": self.exceptions,
            "expected": round(self.expected, 1),
            "exception rate": self.exceptions / self.days if self.days else np.nan,
            "Kupiec p-value": self.kupiec_p,
            "independence p-value": self.independence_p,
            "last 250d exceptions": self.last_250_exceptions,
            "Basel zone": self.basel_zone,
            "worst excess loss": self.worst_excess,
            "verdict": self.verdict,
        }


def backtest_var(var_forecast: pd.Series, pnl: pd.Series, level: float = 0.99) -> VarBacktest:
    """Run every test on a VaR forecast series against realised (hypothetical) daily P&L."""
    hits = exceptions(var_forecast, pnl)
    valid = hits.dropna()
    n, x = len(valid), int(valid.sum())
    lr, p = kupiec_pof(x, n, level) if n else (np.nan, np.nan)
    ind_lr, ind_p = christoffersen_independence(valid) if n > 1 else (np.nan, np.nan)
    last = int(valid.iloc[-BASEL_WINDOW:].sum())
    excess = (-pnl - var_forecast.shift(1))[hits.fillna(False).astype(bool)]
    return VarBacktest(
        level=level,
        days=n,
        exceptions=x,
        expected=n * (1.0 - level),
        kupiec_lr=lr,
        kupiec_p=p,
        independence_lr=ind_lr,
        independence_p=ind_p,
        last_250_exceptions=last,
        basel_zone=basel_zone(last) if level == 0.99 else "n/a",
        worst_excess=float(excess.max()) if len(excess) else 0.0,
        hits=hits,
    )


def rolling_basel_zone(hits: pd.Series) -> pd.Series:
    """Exceptions in the trailing 250 days and the zone they put the model in, day by day."""
    count = hits.astype(float).rolling(BASEL_WINDOW, min_periods=BASEL_WINDOW).sum()
    return count.map(lambda c: basel_zone(int(c)) if pd.notna(c) else None)
