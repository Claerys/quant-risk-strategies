"""Ask IBKR what a contract actually is, and refuse to guess when it will not say.

``reqContractDetails`` turns a description into a fact: a conId, an expiry, a multiplier.
Everything that downloads or trades goes through here first. Fail closed:

  zero candidates   NOT_FOUND. Nothing is downloaded.
  exactly one       QUALIFIED. The only case ``qualify`` lets through.
  several           AMBIGUOUS for ``qualify``; this is the normal answer for a futures root
                    asked with ``include_expired``, where ``candidates`` returns every month.

The transport is injected, so the whole callback protocol (timeout, disconnect, errors, duplicate
callbacks, request-id isolation) is testable without a socket.

Adapted from the BSQF ibkr_base project, with its authors' permission.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from quant_risk.ibkr.contract import ContractKey, QualifiedContract

DEFAULT_TIMEOUT = 30.0
IGNORED_CODES = frozenset({2104, 2106, 2107, 2108, 2158, 2174})  # status notices, not failures
NO_SECURITY_DEFINITION = 200


class QualificationStatus(str, Enum):
    QUALIFIED = "qualified"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    ERROR = "error"
    TIMEOUT = "timeout"
    DISCONNECTED = "disconnected"


class QualificationError(RuntimeError):
    """A caller demanded a contract that could not be qualified."""


@dataclass(frozen=True)
class QualificationResult:
    request: ContractKey
    status: QualificationStatus
    candidates: tuple[QualifiedContract, ...] = ()
    error: str = ""

    @property
    def contract(self) -> QualifiedContract:
        """The single qualified contract. There is deliberately no "best candidate"."""
        if self.status is not QualificationStatus.QUALIFIED:
            raise QualificationError(self.describe())
        return self.candidates[0]

    def describe(self) -> str:
        label = self.request.label
        if self.status is QualificationStatus.NOT_FOUND:
            return f"{label}: no IBKR contract matches this description"
        if self.status is QualificationStatus.AMBIGUOUS:
            return f"{label}: {len(self.candidates)} contracts match; set con_id to choose one"
        if self.status is QualificationStatus.QUALIFIED:
            return f"{label}: conId {self.candidates[0].con_id}"
        return f"{label}: {self.status.value}: {self.error}"


class ContractDetailsTransport(Protocol):
    def allocate_request_id(self) -> int: ...
    def request_contract_details(self, req_id: int, contract: Any) -> None: ...
    def cancel_contract_details(self, req_id: int) -> None: ...
    def is_connected(self) -> bool: ...


@dataclass
class _Pending:
    req_id: int
    done: threading.Event = field(default_factory=threading.Event)
    candidates: list[QualifiedContract] = field(default_factory=list)
    seen_con_ids: set[int] = field(default_factory=set)
    error: str = ""
    error_code: int = 0
    disconnected: bool = False
    ended: bool = False


class ContractDetailsCollector:
    """Callback bookkeeping keyed by request id; callbacks for other requests are ignored."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}

    def open(self, req_id: int) -> _Pending:
        with self._lock:
            if req_id in self._pending:
                raise RuntimeError(f"request id {req_id} is already in flight")
            return self._pending.setdefault(req_id, _Pending(req_id))

    def close(self, req_id: int) -> None:
        with self._lock:
            self._pending.pop(req_id, None)

    @property
    def in_flight(self) -> int:
        with self._lock:
            return len(self._pending)

    def on_details(self, req_id: int, details: Any) -> None:
        qualified = QualifiedContract.from_contract_details(details)
        with self._lock:
            request = self._pending.get(req_id)
            if request is None or request.ended:
                return
            if qualified.con_id and qualified.con_id in request.seen_con_ids:
                return  # IBKR can repeat a callback; a duplicate must not create an ambiguity
            if qualified.con_id:
                request.seen_con_ids.add(qualified.con_id)
            request.candidates.append(qualified)

    def on_end(self, req_id: int) -> None:
        with self._lock:
            request = self._pending.get(req_id)
            if request is not None:
                request.ended = True
                request.done.set()

    def on_error(self, req_id: int, code: int, message: str) -> None:
        if code in IGNORED_CODES:
            return
        with self._lock:
            request = self._pending.get(req_id)
            if request is not None:
                request.error, request.error_code = message, code
                request.done.set()

    def on_disconnect(self) -> None:
        with self._lock:
            for request in self._pending.values():
                request.disconnected = True
                request.done.set()


class ContractQualifier:
    """Blocking, request-scoped ``reqContractDetails``."""

    def __init__(self, transport: ContractDetailsTransport, timeout: float = DEFAULT_TIMEOUT,
                 collector: ContractDetailsCollector | None = None) -> None:
        self.transport = transport
        self.timeout = timeout
        self.collector = collector or ContractDetailsCollector()

    def lookup(self, key: ContractKey) -> QualificationResult:
        """Every candidate IBKR returns. Status is QUALIFIED for one, AMBIGUOUS for many."""
        if not self.transport.is_connected():
            return QualificationResult(key, QualificationStatus.DISCONNECTED, error="not connected")
        req_id = self.transport.allocate_request_id()
        request = self.collector.open(req_id)
        try:
            try:
                self.transport.request_contract_details(req_id, key.to_ib_contract())
            except Exception as exc:  # noqa: BLE001 - any send failure is reported, never raised
                return QualificationResult(key, QualificationStatus.ERROR, error=f"{type(exc).__name__}: {exc}")
            if not request.done.wait(self.timeout):
                self._safe_cancel(req_id)
                return QualificationResult(key, QualificationStatus.TIMEOUT,
                                           error=f"no contractDetailsEnd within {self.timeout:g}s")
            if request.disconnected:
                return QualificationResult(key, QualificationStatus.DISCONNECTED,
                                           error="connection dropped before contractDetailsEnd")
            if request.error:
                status = (QualificationStatus.NOT_FOUND if request.error_code == NO_SECURITY_DEFINITION
                          else QualificationStatus.ERROR)
                return QualificationResult(key, status, error=request.error)
            candidates = tuple(request.candidates)
            if not candidates:
                return QualificationResult(key, QualificationStatus.NOT_FOUND)
            status = QualificationStatus.QUALIFIED if len(candidates) == 1 else QualificationStatus.AMBIGUOUS
            return QualificationResult(key, status, candidates)
        finally:
            self.collector.close(req_id)

    def qualify(self, key: ContractKey) -> QualifiedContract:
        """Exactly one contract, or ``QualificationError``."""
        return self.lookup(key).contract

    def candidates(self, key: ContractKey) -> tuple[QualifiedContract, ...]:
        """All contracts (e.g. every futures month). Zero is an error, never an empty list."""
        result = self.lookup(key)
        if result.status not in (QualificationStatus.QUALIFIED, QualificationStatus.AMBIGUOUS):
            raise QualificationError(result.describe())
        return result.candidates

    def _safe_cancel(self, req_id: int) -> None:
        try:
            self.transport.cancel_contract_details(req_id)
        except Exception:  # noqa: BLE001, S110 - cancelling a dead request must not mask the timeout
            pass
