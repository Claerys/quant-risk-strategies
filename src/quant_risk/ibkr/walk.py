"""Walk backward through daily history, and know why you stopped.

IBKR caps how much one request may span, so years of history take several requests, each ending
where the previous began. The only ways to stop early are provable: IBKR returned nothing, or the
window stopped moving (the same oldest bar came back). Running out of request budget is neither,
and is reported as a failure, because a silently truncated series is worse than a failed download.

Adapted from the BSQF ibkr_base project, with its authors' permission.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

CHUNK_YEARS = 5  # one "5 Y" request per chunk
MAX_YEARS = 25
BUDGET_SLACK = 5


class IncompleteCoverageError(RuntimeError):
    """The walk ran out of requests before reaching the requested cutoff."""


def chunk_budget(years: int) -> int:
    return int(math.ceil(min(years, MAX_YEARS) / CHUNK_YEARS) * 2) + BUDGET_SLACK


@dataclass
class WalkResult:
    bars: pd.DataFrame
    chunks: int = 0
    reached_target: bool = False
    data_ran_out: bool = False
    stopped_because: str = ""

    @property
    def is_complete(self) -> bool:
        return self.reached_target or self.data_ran_out

    def require_coverage(self, name: str, years: int) -> pd.DataFrame:
        if not self.is_complete:
            raise IncompleteCoverageError(
                f"{name}: asked for {years}y of daily bars but the request budget ran out at "
                f"{self.bars.index.min():%Y-%m-%d}; rerun to continue"
            )
        return self.bars


def end_before(day: pd.Timestamp) -> str:
    """endDateTime for the next older chunk: the last second of the previous day, UTC."""
    return f"{(day - pd.Timedelta(days=1)):%Y%m%d}-23:59:59"


def walk_backward(fetch: Callable[[str, str], pd.DataFrame], years: int, *, today: pd.Timestamp) -> WalkResult:
    """``fetch(duration, end)`` returns daily bars ending at ``end`` ("" means now)."""
    target = min(max(int(years), 1), MAX_YEARS)
    cutoff = today - pd.DateOffset(years=target)
    frames: list[pd.DataFrame] = []
    result = WalkResult(bars=pd.DataFrame())
    end, previous_oldest = "", None
    for _ in range(chunk_budget(target)):
        result.chunks += 1
        bars = fetch(f"{CHUNK_YEARS} Y", end)
        if bars.empty:
            result.data_ran_out, result.stopped_because = True, "IBKR returned no more bars"
            break
        frames.append(bars)
        oldest = bars.index.min()
        if oldest <= cutoff:
            result.reached_target, result.stopped_because = True, f"reached the {target}y cutoff"
            break
        if previous_oldest is not None and oldest >= previous_oldest:
            result.data_ran_out = True
            result.stopped_because = f"the walk stopped moving at {oldest:%Y-%m-%d}: no earlier bars"
            break
        previous_oldest, end = oldest, end_before(oldest)
    if frames:
        merged = pd.concat(frames)
        result.bars = merged[~merged.index.duplicated(keep="last")].sort_index()
    return result
