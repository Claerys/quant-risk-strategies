"""Decide what part of a ticker's history is actually missing.

A rerun should cost one small request for the latest bars, not a full re-download. The planner
compares what is already stored with what was asked for and returns the smallest request that
closes the gap:

  full      nothing stored yet: pull the whole requested history
  forward   stored but stale: pull only the days since the last bar
  backfill  recent data is current but history is shorter than asked
  skip      already covers the requested window: request nothing

Adapted from the BSQF ibkr_base project, with its authors' permission.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# A daily series is "current" within this many calendar days: markets close for weekends and
# holidays, so demanding today's bar would re-request every symbol every Monday for nothing.
FRESH_TOLERANCE_DAYS = 4
BACKFILL_MIN_DAYS = 30  # only worth another request if a meaningful slice of history is absent


@dataclass(frozen=True)
class DownloadPlan:
    action: str  # "skip" | "full" | "forward" | "backfill"
    duration: str  # IBKR duration string, empty when action is skip
    reason: str
    missing_days: int = 0


def duration_for_days(days: int) -> str:
    """Smallest IBKR duration string that covers a gap of N calendar days."""
    days = max(1, days)
    if days <= 60:
        return f"{days} D"
    if days <= 365:
        return f"{max(1, (days + 29) // 30)} M"
    return f"{max(1, (days + 364) // 365)} Y"


def plan_download(first_bar: pd.Timestamp | None, last_bar: pd.Timestamp | None, bar_count: int, years: int,
                  today: pd.Timestamp) -> DownloadPlan:
    if bar_count <= 0 or first_bar is None or last_bar is None:
        return DownloadPlan("full", f"{max(1, years)} Y", "no bars stored yet")
    days_behind = (today.date() - last_bar.date()).days
    if days_behind > FRESH_TOLERANCE_DAYS:
        # pad by a couple of days so a partly written last bar is rewritten, not left half filled
        return DownloadPlan("forward", duration_for_days(days_behind + 2), f"{days_behind} days behind",
                            missing_days=days_behind)
    wanted_start = today - pd.Timedelta(days=365 * max(1, years))
    missing = (first_bar.date() - wanted_start.date()).days
    if missing > BACKFILL_MIN_DAYS:
        return DownloadPlan("backfill", duration_for_days(missing + 30),
                            f"history starts {first_bar.date()}, {missing} days short of the {years}y window",
                            missing_days=missing)
    return DownloadPlan("skip", "", f"up to date ({bar_count} bars, last {last_bar.date()})")
