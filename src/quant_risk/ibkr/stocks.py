"""Download daily stock bars from IB Gateway into ``data/stocks/<TICKER>.parquet``.

Follows the BSQF ibkr_base flow: qualify the row first (a ticker IBKR will not resolve to exactly one
contract is skipped and reported, never guessed), plan the smallest request that brings it up to date
(``plan.py``), walk backward through history in chunks (``walk.py``), store.

One addition: IBKR re-adjusts a stock's whole history after a split. If the bars it returns now do not
match what is on disk over the days they share, appending would splice two price scales together, so
the ticker is downloaded again from scratch instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quant_risk.bars import bars_path, data_dir, normalize_bars, write_bars
from quant_risk.ibkr.pacing import Pacer
from quant_risk.ibkr.plan import plan_download
from quant_risk.ibkr.tickers import TickerDefinition
from quant_risk.ibkr.walk import walk_backward
from quant_risk.quality import ERROR, Issue, check_bars

PRICE_TOLERANCE = 1e-6  # relative; larger differences on shared days mean IBKR re-adjusted the history


def stocks_dir() -> Path:
    return data_dir().parent / "stocks"


@dataclass
class StockResult:
    symbol: str
    action: str  # skip | full | forward | backfill | refresh
    bars: int
    added: int
    reason: str = ""
    issues: list[Issue] = field(default_factory=list)


def _adjusted(history: pd.DataFrame, new: pd.DataFrame) -> bool:
    shared = history.index.intersection(new.index)
    if shared.empty:
        return False
    old, fresh = history.loc[shared, "close"], new.loc[shared, "close"]
    return bool(((old - fresh).abs() > PRICE_TOLERANCE * old.abs().clip(lower=1e-12)).any())


def download_stock(client, ticker: TickerDefinition, *, years: int = 20, directory: Path | None = None,
                   today: pd.Timestamp | None = None, pacer: Pacer | None = None) -> StockResult:
    """Raises ``QualificationError`` when IBKR cannot resolve the row to exactly one contract."""
    today = (today if today is not None else pd.Timestamp.today()).normalize()
    pacer = pacer or Pacer()
    key = client.qualify(ticker.to_key()).key
    path = bars_path(ticker.symbol, directory or stocks_dir())
    history = normalize_bars(pd.read_parquet(path)) if path.exists() else None
    cutoff = today - pd.DateOffset(years=years)

    def fetch(duration: str, end: str) -> pd.DataFrame:
        pacer.wait()
        return client.historical_bars(key, end=end, duration=duration, what="TRADES", use_rth=True)

    def full(action: str, reason: str) -> StockResult:
        bars = walk_backward(fetch, years, today=today).require_coverage(ticker.symbol, years)
        return finish(bars.loc[bars.index >= cutoff], action, reason)

    def finish(final: pd.DataFrame, action: str, reason: str) -> StockResult:
        if final.empty:
            raise RuntimeError(f"{ticker.symbol}: IBKR returned no bars; nothing was written")
        issues = check_bars(ticker.symbol, final, as_of=today)
        errors = [i for i in issues if i.severity == ERROR]
        if errors:
            raise RuntimeError(f"{ticker.symbol}: {len(errors)} data-quality errors, first: {errors[0].check} "
                               f"{errors[0].detail}; nothing was written")
        write_bars(ticker.symbol, final, directory or stocks_dir())
        before = 0 if history is None or action in ("full", "refresh") else len(history)
        return StockResult(ticker.symbol, action, len(final), len(final) - before, reason, issues)

    plan = plan_download(None if history is None else history.index[0], None if history is None else history.index[-1],
                         0 if history is None else len(history), years, today)
    if plan.action == "skip":
        return StockResult(ticker.symbol, "skip", len(history), 0, plan.reason)
    if plan.action == "full":
        return full("full", plan.reason)

    if plan.action == "forward":
        recent = fetch(plan.duration, "")
        if recent.empty:
            return StockResult(ticker.symbol, "forward", len(history), 0, "no new bars (halt or holiday)")
        if _adjusted(history, recent):
            return full("refresh", "IBKR re-adjusted the history (split); downloaded again")
        return finish(pd.concat([history, recent]).pipe(lambda d: d[~d.index.duplicated(keep="last")]).sort_index(),
                      "forward", plan.reason)

    # backfill: start the walk on the oldest stored day so one bar is shared and can be checked
    older = walk_backward(fetch, years, today=today, anchor=f"{history.index[0]:%Y%m%d}-23:59:59")
    older_bars = older.require_coverage(ticker.symbol, years)
    if _adjusted(history, older_bars):
        return full("refresh", "IBKR re-adjusted the history (split); downloaded again")
    merged = pd.concat([older_bars, history])
    return finish(merged[~merged.index.duplicated(keep="last")].sort_index().loc[lambda d: d.index >= cutoff],
                  "backfill", plan.reason)
