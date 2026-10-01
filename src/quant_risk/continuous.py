"""Build a continuous futures series from individual contract months, and splice it onto history.

A futures contract expires, so a long history needs a chain of contracts: hold the front month,
then roll into the next one before expiry. On the roll date the two contracts trade at different
prices (the "roll gap"), and gluing the raw closes together would book that gap as a fake P&L.

Back-adjustment removes it. Every price before a roll is shifted by that roll's gap, so the series
moves only when the held contract moves. Shifts are additive, which keeps daily P&L exact and is
why old back-adjusted prices can drift far from real traded levels, even below zero.

Roll rule: hold the contract with the most trading volume. Roll forward when a later contract
trades more than the one held, and in any case ``force_days`` trading days before the held
contract's last trade date. Never roll backwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class ContractBars:
    local_symbol: str  # e.g. "CLZ5"
    last_trade_date: pd.Timestamp
    bars: pd.DataFrame  # normalised daily bars (see bars.normalize_bars)


@dataclass(frozen=True)
class Roll:
    date: pd.Timestamp
    from_contract: str
    to_contract: str
    gap: float  # new contract close - old contract close on the roll date


def back_adjust(
    contracts: list[ContractBars], *, force_days: int = 2
) -> tuple[pd.DataFrame, list[Roll]]:
    """Chain contract months into one back-adjusted daily series. Returns (bars, rolls)."""
    if not contracts:
        raise ValueError("no contracts to chain")
    chain = sorted((c for c in contracts if not c.bars.empty), key=lambda c: c.last_trade_date)
    dates = sorted(set().union(*(c.bars.index for c in chain)))

    held = _first_held(chain, dates[0])
    segments: list[tuple[int, list[pd.Timestamp]]] = [(held, [])]
    rolls: list[Roll] = []

    for date in dates:
        current = chain[held]
        if date not in current.bars.index:
            continue
        target = _roll_target(chain, held, date, force_days)
        if target is not None:
            gap = chain[target].bars.at[date, "close"] - current.bars.at[date, "close"]
            rolls.append(Roll(date, current.local_symbol, chain[target].local_symbol, float(gap)))
            held = target
            segments.append((held, []))
        segments[-1][1].append(date)

    # Each segment is shifted by the sum of the gaps of every roll that came after it.
    pieces = []
    for k, (index, seg_dates) in enumerate(segments):
        if not seg_dates:
            continue
        shift = sum(r.gap for r in rolls[k:])
        piece = chain[index].bars.loc[seg_dates].copy()
        piece[PRICE_COLUMNS] += shift
        pieces.append(piece)
    return pd.concat(pieces).sort_index(), rolls


def _first_held(chain: list[ContractBars], date: pd.Timestamp) -> int:
    trading = [i for i, c in enumerate(chain) if date in c.bars.index]
    return max(trading, key=lambda i: chain[i].bars.at[date, "volume"])


def _roll_target(chain: list[ContractBars], held: int, date: pd.Timestamp, force_days: int) -> int | None:
    later = [i for i in range(held + 1, len(chain)) if date in chain[i].bars.index]
    if not later:
        return None
    busiest = max(later, key=lambda i: chain[i].bars.at[date, "volume"])
    current = chain[held]
    if chain[busiest].bars.at[date, "volume"] > current.bars.at[date, "volume"]:
        return busiest
    days_left = len(pd.bdate_range(date, current.last_trade_date)) - 1
    if days_left <= force_days:
        return busiest
    return None


@dataclass
class SpliceResult:
    bars: pd.DataFrame
    anchor: pd.Timestamp  # last archive date; new data starts the day after
    added: int  # bars appended
    reconciliation: dict = field(default_factory=dict)


def splice(history: pd.DataFrame, extension: pd.DataFrame, *, tolerance: float = 1e-6) -> SpliceResult:
    """Append ``extension`` to ``history`` after history's last date.

    The extension is shifted so that both series agree on the anchor date (the last date they
    share), which keeps the back-adjusted level continuous. Before trusting the new data, its daily
    price changes are reconciled against the history over the dates both cover: if the two sources
    disagree on how much prices moved, the splice is suspect.
    """
    common = history.index.intersection(extension.index)
    if common.empty:
        raise ValueError(
            f"no overlap: history ends {history.index[-1]:%Y-%m-%d}, "
            f"extension starts {extension.index[0]:%Y-%m-%d}; download further back"
        )
    anchor = common[-1]
    shift = history.at[anchor, "close"] - extension.at[anchor, "close"]
    new = extension.loc[extension.index > anchor].copy()
    new[PRICE_COLUMNS] += shift
    combined = pd.concat([history, new]).sort_index()
    return SpliceResult(combined, anchor, len(new), reconcile(history, extension, tolerance=tolerance))


def reconcile(a: pd.DataFrame, b: pd.DataFrame, *, tolerance: float = 1e-6) -> dict:
    """Compare daily close-to-close changes of two sources on the dates both cover.

    Roll dates can legitimately differ between two back-adjustment methods, so a handful of
    mismatched days is normal. A low correlation, or a typical-move ratio far from 1 (a units
    problem, such as cents vs dollars), is not.
    """
    common = a.index.intersection(b.index)
    da = a.loc[common, "close"].diff().iloc[1:]
    db = b.loc[common, "close"].diff().iloc[1:]
    if len(da) < 2:
        return {"overlap_days": len(common), "ok": False, "reason": "overlap too short to compare"}
    mismatch = (da - db).abs() > tolerance + 1e-9 * da.abs().max()
    scale = float(db.abs().median() / da.abs().median()) if da.abs().median() > 0 else float("nan")
    corr = float(da.corr(db))
    ok = bool(corr > 0.95 and 0.9 < scale < 1.1)
    return {
        "overlap_days": len(common),
        "correlation": round(corr, 4),
        "scale_ratio": round(scale, 4),
        "mismatched_days": int(mismatch.sum()),
        "mismatched_dates": [d.strftime("%Y-%m-%d") for d in da.index[mismatch]][:10],
        "ok": ok,
    }
