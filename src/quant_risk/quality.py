"""Data-quality checks on daily bars.

A risk number is only as good as the prices behind it. One bad tick can dominate a historical
VaR; a feed that silently stopped updating makes yesterday's risk look like today's. These checks
run before any data is used and report what they find instead of fixing it quietly.

Severity:
    error    the bar is internally impossible (high below low, missing close). Do not use it.
    warning  plausible but suspicious (a large gap, a stale price, an extreme move). Look at it.
    info     expected features of the data worth knowing about (back-adjusted negative prices).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

ERROR, WARNING, INFO = "error", "warning", "info"


@dataclass(frozen=True)
class Issue:
    symbol: str
    severity: str
    check: str
    date: pd.Timestamp | None
    detail: str


def check_bars(
    symbol: str,
    bars: pd.DataFrame,
    *,
    max_gap_days: int = 7,
    stale_run: int = 5,
    jump_sigmas: float = 10.0,
    as_of: pd.Timestamp | None = None,
    max_age_days: int = 5,
) -> list[Issue]:
    """Run every check on one instrument's normalised bars (see ``bars.normalize_bars``)."""
    issues: list[Issue] = []

    def add(severity: str, check: str, date: pd.Timestamp | None, detail: str) -> None:
        issues.append(Issue(symbol, severity, check, date, detail))

    if bars.empty:
        add(ERROR, "empty", None, "no bars")
        return issues

    # --- impossible bars -----------------------------------------------------------------------
    for date in bars.index[bars[["open", "high", "low", "close"]].isna().any(axis=1)]:
        add(ERROR, "missing_price", date, "open/high/low/close has a missing value")
    for date in bars.index[bars["high"] < bars["low"]]:
        add(ERROR, "high_below_low", date, f"high {bars.at[date, 'high']} < low {bars.at[date, 'low']}")
    outside = (bars["close"] > bars["high"]) | (bars["close"] < bars["low"])
    for date in bars.index[outside]:
        add(ERROR, "close_outside_range", date, f"close {bars.at[date, 'close']} outside [low, high]")

    # --- suspicious bars -----------------------------------------------------------------------
    gaps = bars.index.to_series().diff().dt.days
    for date, days in gaps[gaps > max_gap_days].items():
        add(WARNING, "gap", date, f"{int(days)} calendar days since the previous bar")

    close = bars["close"]
    unchanged = close.diff().eq(0)
    run_id = (~unchanged).cumsum()
    run_length = unchanged.groupby(run_id).cumsum()
    for date in run_length.index[run_length == stale_run]:
        add(WARNING, "stale_price", date, f"close unchanged for {stale_run} consecutive bars")

    # A move is "extreme" relative to the instrument's own recent behaviour. The median absolute
    # move is used instead of the standard deviation so that one bad tick cannot hide itself by
    # inflating the yardstick it is measured against.
    move = close.diff().abs()
    typical = move.rolling(60, min_periods=20).median().shift(1)
    jumps = move > jump_sigmas * typical
    for date in bars.index[jumps.fillna(False)]:
        add(WARNING, "extreme_move", date, f"move of {move[date]:.4g} vs typical {typical[date]:.4g}")

    if as_of is not None:
        age = (pd.Timestamp(as_of).normalize() - bars.index[-1]).days
        if age > max_age_days:
            add(WARNING, "stale_series", bars.index[-1], f"last bar is {age} days old as of {as_of:%Y-%m-%d}")

    # --- facts worth knowing -------------------------------------------------------------------
    non_positive = int((close <= 0).sum())
    if non_positive:
        add(INFO, "non_positive_price", None,
            f"{non_positive} closes <= 0 (back-adjusted series): use price differences, not % returns")
    zero_volume = int((bars["volume"] == 0).sum())
    if zero_volume:
        add(INFO, "zero_volume", None, f"{zero_volume} bars with zero volume")

    return issues


def summarize(issues: list[Issue]) -> pd.DataFrame:
    """Count issues per symbol, check and severity, worst first."""
    if not issues:
        return pd.DataFrame(columns=["symbol", "severity", "check", "count", "first", "last"])
    df = pd.DataFrame([vars(i) for i in issues])
    order = {ERROR: 0, WARNING: 1, INFO: 2}
    out = (
        df.groupby(["symbol", "severity", "check"], dropna=False)
        .agg(count=("check", "size"), first=("date", "min"), last=("date", "max"))
        .reset_index()
    )
    return out.sort_values(["severity", "symbol"], key=lambda s: s.map(order) if s.name == "severity" else s)
