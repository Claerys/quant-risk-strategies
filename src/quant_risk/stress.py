"""Stress testing: what the current portfolio would lose in a defined bad scenario.

VaR answers "how much on a normal bad day"; a stress test answers "how much if *this* happens".
Scenarios live in config/scenarios.toml so they can be reviewed and changed without code.

Historical scenarios replay each instrument's real price change between two dates on today's
positions. Price *changes* are unaffected by back-adjustment, so this is exact per contract; the
P&L is converted to USD at today's FX rate, as if the move happened now.

Shock scenarios move each price by a percentage of today's price. Today's back-adjusted price
equals the real traded price, so the dollar size of a percentage move is exact.

Reverse stress testing works the other way round: it searches the whole history for the worst 1,
5 and 20-day windows for today's book, with no scenario chosen in advance.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quant_risk.bars import REPO_ROOT
from quant_risk.fx import usd_per_unit
from quant_risk.instruments import get_instrument, load_instruments

DEFAULT_SCENARIOS_FILE = REPO_ROOT / "config" / "scenarios.toml"


@dataclass(frozen=True)
class Scenario:
    name: str
    category: str  # historical, hypothetical, climate
    kind: str  # "historical" or "shock"
    description: str
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    shocks: dict[str, float] = field(default_factory=dict)  # sector or symbol -> % move

    def shock_for(self, symbol: str) -> float:
        """Percentage move for one instrument: its own entry, else its sector's, else 0."""
        if symbol in self.shocks:
            return self.shocks[symbol]
        return self.shocks.get(get_instrument(symbol).sector, 0.0)


def load_scenarios(path: Path = DEFAULT_SCENARIOS_FILE) -> list[Scenario]:
    with path.open("rb") as handle:
        raw = tomllib.load(handle).get("scenario", [])
    known = set(load_instruments()) | {i.sector for i in load_instruments().values()}
    scenarios = []
    for item in raw:
        unknown = set(item.get("shocks", {})) - known
        if unknown:
            raise ValueError(f"scenario {item['name']!r}: unknown sector or symbol {sorted(unknown)}")
        if item["kind"] not in ("historical", "shock"):
            raise ValueError(f"scenario {item['name']!r}: kind must be 'historical' or 'shock'")
        scenarios.append(
            Scenario(
                name=item["name"],
                category=item.get("category", item["kind"]),
                kind=item["kind"],
                description=" ".join(item.get("description", "").split()),
                start=pd.Timestamp(item["start"]) if "start" in item else None,
                end=pd.Timestamp(item["end"]) if "end" in item else None,
                shocks={k: float(v) for k, v in item.get("shocks", {}).items()},
            )
        )
    return scenarios


def _usd_per_point(symbols: list[str], as_of: pd.Timestamp) -> pd.Series:
    """USD P&L of a 1.0 price move on one contract, at today's FX rate."""
    out = {}
    for s in symbols:
        inst = get_instrument(s)
        fx = 1.0 if inst.currency == "USD" else float(usd_per_unit(inst.currency, pd.DatetimeIndex([as_of])).iloc[0])
        out[s] = inst.multiplier * fx
    return pd.Series(out)


def scenario_price_moves(scenario: Scenario, closes: pd.DataFrame) -> pd.Series:
    """Price change per instrument, in each instrument's own price units."""
    latest = closes.ffill().iloc[-1]
    if scenario.kind == "shock":
        return pd.Series({s: latest[s] * scenario.shock_for(s) / 100.0 for s in closes.columns})
    if scenario.start < closes.index[0]:
        # Not covered by the data. NaN, not zero: "no loss" would be a false reassurance.
        return pd.Series(np.nan, index=closes.columns)
    end = closes.ffill().loc[: scenario.end].iloc[-1]
    start = closes.ffill().loc[: scenario.start].iloc[-1]
    return (end - start).fillna(0.0)  # an instrument not trading yet contributes nothing


@dataclass
class StressResult:
    scenario: Scenario
    pnl: float  # USD, negative is a loss
    by_instrument: pd.Series  # USD contribution of each position

    def top_contributors(self, n: int = 3) -> str:
        worst = self.by_instrument[self.by_instrument != 0].sort_values().head(n)
        return ", ".join(f"{s} {v / 1e3:+,.0f}k" for s, v in worst.items())


def run_scenario(scenario: Scenario, positions: pd.Series, closes: pd.DataFrame) -> StressResult:
    moves = scenario_price_moves(scenario, closes)
    per_point = _usd_per_point(list(positions.index), closes.index[-1])
    if moves.isna().all():
        return StressResult(scenario, float("nan"), pd.Series(0.0, index=positions.index))
    by_instrument = positions.fillna(0.0) * moves.reindex(positions.index).fillna(0.0) * per_point
    return StressResult(scenario, float(by_instrument.sum()), by_instrument)


def stress_table(positions: pd.Series, closes: pd.DataFrame, *, capital: float,
                 scenarios: list[Scenario] | None = None) -> pd.DataFrame:
    """One row per scenario: P&L in USD and % of capital, worst contributors, description."""
    rows = []
    for scenario in scenarios if scenarios is not None else load_scenarios():
        result = run_scenario(scenario, positions, closes)
        rows.append({
            "scenario": scenario.name,
            "category": scenario.category,
            "P&L": result.pnl,
            "% of capital": result.pnl / capital,
            "biggest losses": result.top_contributors() if result.pnl == result.pnl else "no data for this period",
            "description": scenario.description,
        })
    return pd.DataFrame(rows).set_index("scenario")


def reverse_stress(positions: pd.Series, pnl_per_contract: pd.DataFrame,
                   horizons: tuple[int, ...] = (1, 5, 20)) -> pd.DataFrame:
    """The worst historical windows for today's book: no scenario chosen in advance."""
    daily = pnl_per_contract.reindex(columns=positions.index).fillna(0.0) @ positions.fillna(0.0)
    rows = []
    for h in horizons:
        rolling = daily.rolling(h).sum()
        end = rolling.idxmin()
        start = daily.index[max(daily.index.get_loc(end) - h + 1, 0)]
        rows.append({"horizon (days)": h, "worst P&L": float(rolling.min()), "from": start.date(), "to": end.date()})
    return pd.DataFrame(rows).set_index("horizon (days)")


def sector_shock_grid(positions: pd.Series, closes: pd.DataFrame,
                      moves: tuple[float, ...] = (-20.0, -10.0, -5.0, 5.0, 10.0, 20.0)) -> pd.DataFrame:
    """P&L of moving one sector at a time by each percentage: a quick map of where the book is exposed."""
    latest = closes.ffill().iloc[-1].reindex(positions.index)
    per_point = _usd_per_point(list(positions.index), closes.index[-1])
    sectors = pd.Series({s: get_instrument(s).sector for s in positions.index})
    value = positions.fillna(0.0) * latest * per_point  # signed USD value of each position
    by_sector = value.groupby(sectors).sum()
    return pd.DataFrame({f"{m:+.0f}%": by_sector * m / 100.0 for m in moves}).loc[lambda d: d.abs().sum(axis=1) > 0]

