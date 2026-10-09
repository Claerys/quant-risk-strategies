"""One immutable identity for a tradable contract.

A futures symbol is not an identity: ES has one contract per expiry. IBKR's ``conId`` is the
identity once a contract is qualified; before that the full description is. Routing details
(valid exchanges, tick size) belong to ``QualifiedContract``, never to the identity.

Adapted from the BSQF ibkr_base project, with its authors' permission.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class ContractKey:
    symbol: str
    sec_type: str = "FUT"
    exchange: str = ""
    currency: str = "USD"
    primary_exchange: str = ""  # stocks: SMART routes, this says which listing is meant
    expiry: str = ""  # lastTradeDateOrContractMonth
    multiplier: str = ""
    trading_class: str = ""
    local_symbol: str = ""
    con_id: int = 0  # 0 = not qualified by IBKR yet
    include_expired: bool = False  # ask for expired months too (futures history)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "sec_type", self.sec_type.strip().upper())

    @property
    def identity(self) -> tuple:
        """conId wins when known; otherwise every descriptive field takes part."""
        if self.con_id > 0:
            return ("conid", self.con_id)
        return ("composite", self.symbol, self.sec_type, self.exchange, self.currency, self.primary_exchange, self.expiry,
                self.multiplier, self.trading_class, self.local_symbol)

    def __hash__(self) -> int:
        return hash(self.identity)

    def __eq__(self, other: object) -> bool:
        return self.identity == other.identity if isinstance(other, ContractKey) else NotImplemented

    @property
    def label(self) -> str:
        if self.local_symbol:
            return f"{self.local_symbol} ({self.sec_type})"
        return " ".join(p for p in (self.symbol, self.sec_type, self.expiry) if p)

    def to_ib_contract(self) -> Any:
        from ibapi.contract import Contract

        contract = Contract()
        if self.con_id > 0:
            contract.conId = self.con_id
        contract.symbol = self.symbol
        contract.secType = self.sec_type
        contract.exchange = self.exchange
        contract.currency = self.currency
        contract.includeExpired = self.include_expired
        for attribute, value in (
            ("primaryExchange", self.primary_exchange),
            ("lastTradeDateOrContractMonth", self.expiry),
            ("multiplier", self.multiplier),
            ("tradingClass", self.trading_class),
            ("localSymbol", self.local_symbol),
        ):
            if value:
                setattr(contract, attribute, value)
        return contract

    @classmethod
    def from_ib_contract(cls, contract: Any) -> ContractKey:
        return cls(
            symbol=getattr(contract, "symbol", "") or "",
            sec_type=getattr(contract, "secType", "") or "FUT",
            exchange=getattr(contract, "exchange", "") or "",
            currency=getattr(contract, "currency", "") or "USD",
            primary_exchange=getattr(contract, "primaryExchange", "") or "",
            expiry=getattr(contract, "lastTradeDateOrContractMonth", "") or "",
            multiplier=str(getattr(contract, "multiplier", "") or ""),
            trading_class=getattr(contract, "tradingClass", "") or "",
            local_symbol=getattr(contract, "localSymbol", "") or "",
            con_id=int(getattr(contract, "conId", 0) or 0),
        )

    def with_con_id(self, con_id: int) -> ContractKey:
        return replace(self, con_id=int(con_id))


@dataclass(frozen=True)
class QualifiedContract:
    """A key plus what ``reqContractDetails`` said about it."""

    key: ContractKey
    min_tick: float = 0.0
    time_zone_id: str = ""
    long_name: str = ""

    @property
    def con_id(self) -> int:
        return self.key.con_id

    @classmethod
    def from_contract_details(cls, details: Any) -> QualifiedContract:
        contract = getattr(details, "contract", None)
        key = ContractKey.from_ib_contract(contract) if contract is not None else ContractKey("")
        return cls(
            key=key,
            min_tick=float(getattr(details, "minTick", 0.0) or 0.0),
            time_zone_id=str(getattr(details, "timeZoneId", "") or ""),
            long_name=str(getattr(details, "longName", "") or ""),
        )


@dataclass(frozen=True)
class OrderResult:
    filled: float  # contracts actually filled, unsigned
    avg_price: float
    status: str
