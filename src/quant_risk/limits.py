"""Pre-trade risk limits: every target portfolio is reviewed before a single order is sent.

The sizing step proposes how many contracts to hold. This module decides how many may be held.

**Fail closed.** Limits are loaded from config/limits.toml and every limit must be set. A missing
value is not read as "no limit": the engine refuses to start and lists what is missing. If the
day's P&L or the current drawdown cannot be measured, nothing that adds risk is approved.

**Hard limits** stop risk from growing:
    kill switch      no orders at all
    max daily loss   once hit, only orders that reduce positions are allowed
    max drawdown     stop-out: close every position (subject to the order-size limit) and stay flat
                     while the drawdown, measured from the peak of cumulative P&L, is beyond it

**Soft limits** shrink the proposal until it fits, instead of rejecting it:
    max VaR (99%, 1-day)      enforced by the trading cycle, which holds the history VaR needs
    max order size            contracts in a single order
    max position notional     USD value of one instrument's position
    max sector gross          USD value of all positions in one sector
    max gross exposure        sum of |position values|
    max net exposure          |sum of signed position values|

Each review reports every limit's utilisation (value / limit), so a breach is visible as a number
close to or above 100%, not just as a yes/no.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from quant_risk.bars import REPO_ROOT
from quant_risk.fx import usd_per_unit
from quant_risk.instruments import get_instrument

DEFAULT_LIMITS_FILE = REPO_ROOT / "config" / "limits.toml"


class LimitsNotConfigured(RuntimeError):
    pass


@dataclass(frozen=True)
class RiskLimits:
    max_gross_exposure: float  # USD
    max_net_exposure: float  # USD
    max_position_notional: float  # USD per instrument
    max_sector_gross: float  # USD per sector
    max_order_contracts: int
    max_daily_loss: float  # USD, positive number
    max_drawdown: float  # fraction of capital, positive number (0.20 = 20%)
    max_var_99: float  # USD, 1-day 99% filtered historical VaR of the book after trading
    kill_switch: bool = False

    @classmethod
    def from_mapping(cls, values: dict) -> RiskLimits:
        problems = []
        kwargs = {}
        for f in fields(cls):
            raw = values.get(f.name)
            if f.name == "kill_switch":
                kwargs[f.name] = bool(raw) if raw is not None else False
                continue
            if raw is None or raw == "":
                problems.append(f"{f.name} is not set")
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError):
                problems.append(f"{f.name}={raw!r} is not a number")
                continue
            if not number > 0:
                problems.append(f"{f.name}={raw!r} must be positive")
                continue
            kwargs[f.name] = int(number) if f.name == "max_order_contracts" else number
        if problems:
            raise LimitsNotConfigured("risk limits are incomplete, refusing to trade:\n  - " + "\n  - ".join(problems))
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Path | None = None) -> RiskLimits:
        """Read limits from ``path``, else $QRS_LIMITS_FILE, else config/limits.toml."""
        path = path or Path(os.environ.get("QRS_LIMITS_FILE", DEFAULT_LIMITS_FILE))
        if not path.exists():
            raise LimitsNotConfigured(f"no limits file at {path}; copy config/limits.toml and set every value")
        with path.open("rb") as handle:
            return cls.from_mapping(tomllib.load(handle).get("limits", {}))


@dataclass(frozen=True)
class LimitCheck:
    name: str
    value: float
    limit: float

    @property
    def utilisation(self) -> float:
        return self.value / self.limit if self.limit else np.inf

    @property
    def breached(self) -> bool:
        return self.value > self.limit + 1e-9


@dataclass
class Review:
    approved: pd.Series  # contracts allowed to be held after this review
    orders: pd.Series  # approved - current, signed contracts
    checks: list[LimitCheck]  # utilisation of the approved book
    actions: list[str]  # what the engine changed, and why
    halted: bool  # True when risk-increasing orders were blocked

    def utilisation(self) -> dict[str, float]:
        return {c.name: c.utilisation for c in self.checks}


def contract_value_frame(closes: pd.DataFrame) -> pd.DataFrame:
    """USD value of one contract of each instrument on each date: |price| x multiplier x FX.

    Exact on the latest date, where back-adjusted prices equal traded prices. Further back, the
    back-adjustment shifts price levels, so historical notionals are only approximate.
    """
    multipliers = pd.Series({s: get_instrument(s).multiplier for s in closes.columns})
    currencies = {s: get_instrument(s).currency for s in closes.columns}
    rates = {c: usd_per_unit(c, closes.index) for c in set(currencies.values())}
    fx = pd.DataFrame({s: rates[c] for s, c in currencies.items()})
    return closes.ffill().abs() * multipliers * fx


def _toward_zero(x: np.ndarray) -> np.ndarray:
    return np.trunc(x + np.sign(x) * 1e-9)


def review(
    current: pd.Series,
    target: pd.Series,
    contract_value: pd.Series,
    *,
    limits: RiskLimits,
    capital: float,
    daily_pnl: float | None,
    drawdown: float | None,
) -> Review:
    """Approve, shrink or block a move from `current` to `target` contracts.

    `contract_value` is the USD value of one contract (see ``contract_value_frame``). `daily_pnl` is
    today's P&L in USD; `drawdown` is the current fall from peak as a fraction of capital (<= 0).
    """
    symbols = list(target.index)
    cur = current.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
    tgt = target.fillna(0.0).to_numpy(dtype=float)
    value = contract_value.reindex(symbols).to_numpy(dtype=float)
    sectors = np.array([get_instrument(s).sector for s in symbols])
    actions: list[str] = []

    # --- hard limits ---------------------------------------------------------------------------
    halted = False
    if limits.kill_switch:
        actions.append("kill switch engaged: no orders")
        tgt = cur.copy()
        halted = True
    elif daily_pnl is None or drawdown is None or np.isnan(daily_pnl) or np.isnan(drawdown):
        actions.append("cannot measure daily P&L or drawdown: only risk-reducing orders allowed")
        halted = True
    elif daily_pnl <= -limits.max_daily_loss:
        actions.append(f"daily loss {daily_pnl:,.0f} hit the stop of {limits.max_daily_loss:,.0f}: reduce only")
        halted = True
    elif -drawdown >= limits.max_drawdown:
        actions.append(f"drawdown {drawdown:.1%} hit the limit of {limits.max_drawdown:.0%}: closing all positions")
        tgt = np.zeros_like(cur)
        halted = True
    if halted and not limits.kill_switch:
        # Allowed: move toward zero, never past it, never away from it.
        reducing = ((np.sign(tgt) == np.sign(cur)) & (np.abs(tgt) < np.abs(cur))) | (tgt == 0)
        tgt = np.where(reducing, tgt, cur)

    # --- soft limits ---------------------------------------------------------------------------
    per_position = np.floor(limits.max_position_notional / value)
    capped = np.clip(tgt, -per_position, per_position)
    for i in np.flatnonzero(capped != tgt):
        actions.append(f"{symbols[i]}: position capped at {int(per_position[i])} contracts (position notional)")
    tgt = capped

    for sector in np.unique(sectors):
        members = sectors == sector
        gross = np.sum(np.abs(tgt[members]) * value[members])
        if gross > limits.max_sector_gross:
            tgt[members] = _toward_zero(tgt[members] * limits.max_sector_gross / gross)
            actions.append(f"sector {sector}: scaled to {limits.max_sector_gross:,.0f} gross")

    gross = np.sum(np.abs(tgt) * value)
    if gross > limits.max_gross_exposure:
        tgt = _toward_zero(tgt * limits.max_gross_exposure / gross)
        actions.append(f"portfolio scaled from {gross:,.0f} to {limits.max_gross_exposure:,.0f} gross")

    net = np.sum(tgt * value)
    if abs(net) > limits.max_net_exposure:
        side = np.sign(tgt) == np.sign(net)
        side_value = np.sum(np.abs(tgt[side]) * value[side])
        keep = (side_value - (abs(net) - limits.max_net_exposure)) / side_value
        tgt[side] = _toward_zero(tgt[side] * keep)
        actions.append(f"{'long' if net > 0 else 'short'} side scaled to bring net within {limits.max_net_exposure:,.0f}")

    order = tgt - cur
    too_big = np.abs(order) > limits.max_order_contracts
    for i in np.flatnonzero(too_big):
        actions.append(f"{symbols[i]}: order cut from {int(order[i])} to {limits.max_order_contracts} contracts")
    order = np.clip(order, -limits.max_order_contracts, limits.max_order_contracts)
    approved = cur + order

    sector_gross = {
        sector: float(np.sum(np.abs(approved[sectors == sector]) * value[sectors == sector]))
        for sector in np.unique(sectors)
    }
    worst_sector = max(sector_gross, key=sector_gross.get)
    position_notional = np.abs(approved) * value
    checks = [
        LimitCheck("gross exposure", float(np.sum(position_notional)), limits.max_gross_exposure),
        LimitCheck("net exposure", float(abs(np.sum(approved * value))), limits.max_net_exposure),
        LimitCheck(f"sector gross ({worst_sector})", sector_gross[worst_sector], limits.max_sector_gross),
        LimitCheck("largest position", float(position_notional.max(initial=0.0)), limits.max_position_notional),
        LimitCheck("largest order", float(np.abs(order).max(initial=0.0)), limits.max_order_contracts),
        LimitCheck("daily loss", max(-(daily_pnl or 0.0), 0.0), limits.max_daily_loss),
        LimitCheck("drawdown", max(-(drawdown or 0.0), 0.0), limits.max_drawdown),
    ]
    return Review(
        approved=pd.Series(approved, index=symbols),
        orders=pd.Series(order, index=symbols),
        checks=checks,
        actions=actions,
        halted=halted,
    )


@dataclass
class Replay:
    positions: pd.DataFrame  # contracts held after review, each day
    utilisation: pd.DataFrame  # one column per limit
    actions: list[tuple[pd.Timestamp, str]]
    stopped_on: pd.Timestamp | None  # first day the drawdown limit closed the book


def replay_loss_limits(
    targets: pd.DataFrame, closes: pd.DataFrame, *, capital: float, limits: RiskLimits
) -> Replay:
    """Run the daily review through history to see what the hard loss limits would have done.

    Only the daily-loss, drawdown and order-size limits are applied. Notional limits are switched
    off here because back-adjusted historical price levels make notionals unreliable (see
    ``contract_value_frame``); they are enforced on the live book, where prices are exact.
    """
    from dataclasses import replace

    from quant_risk.bars import pnl_per_contract
    from quant_risk.fx import to_usd

    replay_limits = replace(
        limits,
        max_gross_exposure=np.inf,
        max_net_exposure=np.inf,
        max_position_notional=np.inf,
        max_sector_gross=np.inf,
    )
    values = contract_value_frame(closes).fillna(0.0)
    pnl_per = to_usd(pnl_per_contract(closes.ffill())).reindex(closes.index).fillna(0.0).to_numpy()
    targets = targets.reindex(index=closes.index, columns=closes.columns).fillna(0.0)

    held = pd.Series(0.0, index=closes.columns)
    cumulative = peak = 0.0
    rows, util, actions = [], [], []
    stopped_on = None
    for t, date in enumerate(closes.index):
        pnl = float(held.to_numpy() @ pnl_per[t])
        cumulative += pnl
        peak = max(peak, cumulative)
        drawdown = (cumulative - peak) / capital
        result = review(held, targets.iloc[t], values.iloc[t].replace(0.0, np.nan).fillna(1.0),
                        limits=replay_limits, capital=capital, daily_pnl=pnl, drawdown=drawdown)
        held = result.approved
        rows.append(held.to_numpy())
        util.append({c.name.split(" (")[0]: c.utilisation for c in result.checks if np.isfinite(c.limit)})
        actions += [(date, a) for a in result.actions]
        if stopped_on is None and -drawdown >= limits.max_drawdown:
            stopped_on = date
    return Replay(
        positions=pd.DataFrame(rows, index=closes.index, columns=closes.columns),
        utilisation=pd.DataFrame(util, index=closes.index),
        actions=actions,
        stopped_on=stopped_on,
    )
