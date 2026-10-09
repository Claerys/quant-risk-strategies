"""Download daily futures and EUR/USD from IB Gateway into the project's parquet format.

Futures: every listed and expired month of an instrument is fetched (one request per month, ending
at its expiry), chained into one back-adjusted series with ``continuous.back_adjust``, and written
to ``data/daily/<SYMBOL>.parquet``. When an archive already exists only the recent months are
fetched and joined on with ``continuous.splice``, which first reconciles the overlap against the
archive and refuses to write if the two disagree.

Nothing is written unless the result passes the data-quality checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from quant_risk.bars import bars_path, normalize_bars, write_bars
from quant_risk.continuous import PRICE_COLUMNS, ContractBars, back_adjust, splice
from quant_risk.fx import fx_dir
from quant_risk.ibkr.contract import ContractKey
from quant_risk.ibkr.contracts import ContractMonth, futures_chain, futures_map_for
from quant_risk.ibkr.pacing import Pacer
from quant_risk.ibkr.walk import walk_backward
from quant_risk.instruments import Instrument
from quant_risk.quality import ERROR, Issue, check_bars

OVERLAP_DAYS = 60  # incremental runs refetch contracts expiring this long before the archive ends
LOOKAHEAD_DAYS = 270  # months expiring later than this have not started trading yet


@dataclass
class DownloadResult:
    symbol: str
    mode: str  # "full" or "incremental"
    bars: int
    added: int
    months: int
    issues: list[Issue] = field(default_factory=list)


def _contract_bars(client, month: ContractMonth, scale: float, today: pd.Timestamp, pacer: Pacer,
                   use_rth: bool) -> pd.DataFrame:
    end = "" if month.expiry >= today else f"{month.expiry:%Y%m%d}-23:59:59"
    pacer.wait()
    bars = client.historical_bars(month.key, end=end, duration="2 Y", use_rth=use_rth)
    if not bars.empty:
        bars = bars.copy()
        bars[PRICE_COLUMNS] = bars[PRICE_COLUMNS] * scale
    return bars


def download_symbol(client, instrument: Instrument, *, years: int = 20, directory: Path | None = None,
                    today: pd.Timestamp | None = None, pacer: Pacer | None = None,
                    use_rth: bool = False) -> DownloadResult:
    today = (today if today is not None else pd.Timestamp.today()).normalize()
    pacer = pacer or Pacer()
    scale = futures_map_for(instrument).price_scale
    months = futures_chain(client, instrument)

    path = bars_path(instrument.symbol, directory)
    history = normalize_bars(pd.read_parquet(path)) if path.exists() else None
    cutoff = today - pd.DateOffset(years=years)
    start = cutoff if history is None else history.index[-1] - pd.Timedelta(days=OVERLAP_DAYS)
    wanted = [m for m in months if start <= m.expiry <= today + pd.Timedelta(days=LOOKAHEAD_DAYS)]

    contracts, missing = [], []
    for month in wanted:
        bars = _contract_bars(client, month, scale, today, pacer, use_rth)
        if bars.empty:
            if month.expiry < today:
                missing.append(month.local_symbol)  # an expired month with no data is a hole
            continue
        contracts.append(ContractBars(month.local_symbol, month.expiry, bars))
    if missing:
        raise RuntimeError(f"{instrument.symbol}: IBKR returned no bars for expired months {missing}; "
                           "nothing was written")
    if not contracts:
        raise RuntimeError(f"{instrument.symbol}: no contract months returned any bars")

    series, _rolls = back_adjust(contracts)
    if history is None:
        series = series.loc[series.index >= cutoff]
        added, final = len(series), series
    else:
        # Compare only the recent window: further back the archive may have held an older contract.
        spliced = splice(history, series.loc[series.index >= history.index[-1] - pd.Timedelta(days=OVERLAP_DAYS)])
        if not spliced.reconciliation.get("ok"):
            raise RuntimeError(f"{instrument.symbol}: new data disagrees with the archive "
                               f"{spliced.reconciliation}; archive left untouched")
        added, final = spliced.added, spliced.bars

    issues = check_bars(instrument.symbol, final, as_of=today)
    errors = [i for i in issues if i.severity == ERROR]
    if errors:
        raise RuntimeError(f"{instrument.symbol}: {len(errors)} data-quality errors, first: "
                           f"{errors[0].check} {errors[0].detail}; nothing was written")
    write_bars(instrument.symbol, final, directory)
    return DownloadResult(instrument.symbol, "full" if history is None else "incremental", len(final), added,
                          len(contracts), issues)


def download_fx(client, *, years: int = 20, directory: Path | None = None, today: pd.Timestamp | None = None,
                pacer: Pacer | None = None) -> int:
    """Daily EUR/USD (midpoint) to ``data/fx/EURUSD.parquet``; returns the number of days written."""
    today = (today if today is not None else pd.Timestamp.today()).normalize()
    pacer = pacer or Pacer()
    pair = client.qualify(ContractKey(symbol="EUR", sec_type="CASH", exchange="IDEALPRO", currency="USD")).key

    def fetch(duration: str, end: str) -> pd.DataFrame:
        pacer.wait()
        return client.historical_bars(pair, end=end, duration=duration, what="MIDPOINT", use_rth=True)

    bars = walk_backward(fetch, years, today=today).require_coverage("EURUSD", years)
    if bars.empty:
        raise RuntimeError("EURUSD: IBKR returned no bars; nothing was written")
    bars = bars.assign(volume=0.0)  # FX midpoint bars carry no volume
    write_bars("EURUSD", bars, directory or fx_dir())
    return len(bars)
