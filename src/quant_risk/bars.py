"""Daily price bars: storage on disk and loading for analysis.

One Parquet file per instrument under ``data/daily/<SYMBOL>.parquet``, indexed by date with
columns open, high, low, close, volume.

The futures series are back-adjusted continuous contracts: past prices are shifted so that rolling
from one contract month to the next does not show up as a fake jump. The shift is additive, so old
prices can be far from what was traded at the time and can even be negative (crude oil, heating
oil, soybeans). Price *differences* survive the adjustment; price *ratios* do not. That is why
daily risk is measured as ``pnl_per_contract`` (price change x multiplier) rather than as a
percentage return.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from quant_risk.instruments import get_instrument

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Where daily bars live. Override with the QRS_DATA_DIR environment variable."""
    return Path(os.environ.get("QRS_DATA_DIR", REPO_ROOT / "data" / "daily"))


def bars_path(symbol: str, directory: Path | None = None) -> Path:
    return (directory or data_dir()) / f"{symbol}.parquet"


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Bring raw bars to the standard shape: a sorted, unique DatetimeIndex named 'date'."""
    df = df.copy()
    if "date" in df.columns:
        df = df.set_index("date")
    elif "ts" in df.columns:
        df = df.set_index("ts")
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"bars are missing columns {missing}")
    df.index = pd.to_datetime(df.index).normalize()
    df.index.name = "date"
    df = df[BAR_COLUMNS].astype({"open": float, "high": float, "low": float, "close": float})
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def write_bars(symbol: str, df: pd.DataFrame, directory: Path | None = None) -> Path:
    path = bars_path(symbol, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_bars(df).to_parquet(path)
    return path


def read_bars(symbol: str, directory: Path | None = None) -> pd.DataFrame:
    path = bars_path(symbol, directory)
    if not path.exists():
        raise FileNotFoundError(f"no bars for {symbol} at {path}; see data/README.md")
    return normalize_bars(pd.read_parquet(path))


def available_symbols(directory: Path | None = None) -> list[str]:
    return sorted(p.stem for p in (directory or data_dir()).glob("*.parquet"))


def load_closes(symbols: Iterable[str], directory: Path | None = None) -> pd.DataFrame:
    """Closing prices, one column per symbol, on the union of all trading dates.

    A date on which an instrument did not trade stays NaN. Nothing is forward-filled here:
    whether a missing price means "no change" or "unknown" is a decision for the caller.
    """
    closes = {s: read_bars(s, directory)["close"] for s in symbols}
    return pd.DataFrame(closes).sort_index()


def pnl_per_contract(closes: pd.DataFrame) -> pd.DataFrame:
    """Daily P&L of holding one long contract, in each contract's own currency.

    P&L on day t = (close_t - close_{t-1}) x multiplier. The first row is dropped. A day after a
    missing price is NaN rather than a two-day move counted as one.
    """
    multipliers = pd.Series({s: get_instrument(s).multiplier for s in closes.columns})
    return closes.diff().iloc[1:] * multipliers
