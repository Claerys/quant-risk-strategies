"""Thin threaded wrapper around ``ibapi``: connect, contract lookup, daily bars, positions, orders.

Every call blocks until IBKR answers or ``timeout`` passes. Only paper ports connect (see
``config.require_paper_port``), and the guard runs before any socket is opened.

Adapted from the BSQF ibkr_base project, with its authors' permission.
"""

from __future__ import annotations

import threading
import time

import pandas as pd
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.wrapper import EWrapper

from quant_risk.bars import normalize_bars
from quant_risk.ibkr.config import IbkrSettings, require_paper_port
from quant_risk.ibkr.contract import ContractKey, OrderResult, QualifiedContract
from quant_risk.ibkr.qualification import IGNORED_CODES, ContractDetailsCollector, ContractQualifier


class IbkrRequestError(RuntimeError):
    """IBKR sent back an error for a request."""


FINAL_ORDER_STATES = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


class IbkrClient(EWrapper, EClient):
    def __init__(self, request_timeout: float = 60.0) -> None:
        EClient.__init__(self, self)
        self.request_timeout = request_timeout
        self.accounts: list[str] = []
        self._thread: threading.Thread | None = None
        self._connected = threading.Event()
        self._lock = threading.Lock()
        self._next_id = 1
        self._done: dict[int, threading.Event] = {}
        self._bars: dict[int, list] = {}
        self._errors: dict[int, str] = {}
        self._details = ContractDetailsCollector()
        self._positions: dict[ContractKey, float] = {}
        self._positions_done = threading.Event()
        self._orders: dict[int, dict] = {}

    # ---- connection ---------------------------------------------------------------------------

    def connect_and_start(self, settings: IbkrSettings) -> None:
        require_paper_port(settings.port)
        self.request_timeout = settings.timeout
        EClient.connect(self, settings.host, settings.port, settings.client_id)
        self._thread = threading.Thread(target=self.run, name="ibkr-api-loop", daemon=True)
        self._thread.start()
        if not self._connected.wait(self.request_timeout):
            self.disconnect_and_stop()
            raise TimeoutError(
                f"timed out connecting to IBKR at {settings.host}:{settings.port}; check that IB Gateway "
                "or TWS is running with the API socket enabled on that port"
            )

    def disconnect_and_stop(self) -> None:
        if self.isConnected():
            self.disconnect()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def nextValidId(self, orderId: int) -> None:
        with self._lock:
            self._next_id = max(self._next_id, orderId)
        self._connected.set()

    def managedAccounts(self, accountsList: str) -> None:
        self.accounts = [a for a in accountsList.split(",") if a]

    def connectionClosed(self) -> None:
        self._details.on_disconnect()
        with self._lock:
            for req_id, event in self._done.items():
                self._errors.setdefault(req_id, "IBKR connection closed")
                event.set()
        self._positions_done.set()

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson: str = "") -> None:
        if errorCode in IGNORED_CODES:
            return
        message = f"IBKR error {errorCode} for request {reqId}: {errorString}"
        self._details.on_error(reqId, errorCode, message)
        with self._lock:
            if reqId in self._done:
                self._errors[reqId] = message
                self._done[reqId].set()

    def _allocate_id(self) -> int:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
        return request_id

    # ---- contract details (ContractDetailsTransport) -------------------------------------------

    def contractDetails(self, reqId: int, contractDetails) -> None:
        self._details.on_details(reqId, contractDetails)

    def contractDetailsEnd(self, reqId: int) -> None:
        self._details.on_end(reqId)

    def allocate_request_id(self) -> int:
        return self._allocate_id()

    def request_contract_details(self, req_id: int, contract) -> None:
        self.reqContractDetails(req_id, contract)

    def cancel_contract_details(self, req_id: int) -> None:
        return None  # ibapi has no cancel; the request ends by itself

    def is_connected(self) -> bool:
        return bool(self.isConnected())

    def qualifier(self) -> ContractQualifier:
        return ContractQualifier(self, timeout=self.request_timeout, collector=self._details)

    def qualify(self, key: ContractKey) -> QualifiedContract:
        return self.qualifier().qualify(key)

    def candidates(self, key: ContractKey) -> tuple[QualifiedContract, ...]:
        return self.qualifier().candidates(key)

    # ---- daily bars ------------------------------------------------------------------------------

    def historicalData(self, reqId: int, bar) -> None:
        row = {"date": str(bar.date), "open": float(bar.open), "high": float(bar.high),
               "low": float(bar.low), "close": float(bar.close), "volume": float(bar.volume)}
        with self._lock:
            self._bars.setdefault(reqId, []).append(row)

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:
        with self._lock:
            event = self._done.get(reqId)
        if event is not None:
            event.set()

    def historical_bars(self, key: ContractKey, end: str, duration: str, what: str = "TRADES",
                        bar_size: str = "1 day", use_rth: bool = True) -> pd.DataFrame:
        """Daily bars as a normalised frame; empty when IBKR has none for the window."""
        req_id = self._allocate_id()
        event = threading.Event()
        with self._lock:
            self._done[req_id], self._bars[req_id] = event, []
            self._errors.pop(req_id, None)
        self.reqHistoricalData(req_id, key.to_ib_contract(), end, duration, bar_size, what, int(use_rth), 1,
                               False, [])
        if not event.wait(self.request_timeout):
            self.cancelHistoricalData(req_id)
            self._forget(req_id)
            raise TimeoutError(f"timed out waiting for bars for {key.label}")
        time.sleep(0.05)  # let the last callback land
        with self._lock:
            error, rows = self._errors.get(req_id), list(self._bars.get(req_id, []))
        self._forget(req_id)
        if error:
            raise IbkrRequestError(error)
        return normalize_bars(pd.DataFrame(rows)) if rows else pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"])

    def _forget(self, req_id: int) -> None:
        with self._lock:
            for table in (self._done, self._bars, self._errors):
                table.pop(req_id, None)

    # ---- positions ---------------------------------------------------------------------------------

    def position(self, account: str, contract: Contract, position: float, avgCost: float) -> None:
        with self._lock:
            self._positions[ContractKey.from_ib_contract(contract)] = float(position)

    def positionEnd(self) -> None:
        self._positions_done.set()

    def positions(self) -> dict[ContractKey, float]:
        """Open positions across the connected account(s), by contract; zeros dropped."""
        with self._lock:
            self._positions.clear()
        self._positions_done.clear()
        self.reqPositions()
        finished = self._positions_done.wait(self.request_timeout)
        self.cancelPositions()
        if not finished:
            raise TimeoutError("timed out waiting for positions")
        with self._lock:
            return {k: v for k, v in self._positions.items() if v}

    # ---- orders ------------------------------------------------------------------------------------

    def orderStatus(self, orderId: int, status: str, filled: float, remaining: float, avgFillPrice: float,
                    permId: int, parentId: int, lastFillPrice: float, clientId: int, whyHeld: str,
                    mktCapPrice: float = 0.0) -> None:
        with self._lock:
            self._orders[orderId] = {"status": status, "filled": float(filled), "avg": float(avgFillPrice)}
            event = self._done.get(orderId)
        if event is not None and status in FINAL_ORDER_STATES:
            event.set()

    def place_market_order(self, key: ContractKey, quantity: float, account: str) -> OrderResult:
        """Send a market order and wait for its final state. Never reports more than filled."""
        order = Order()
        order.action = "BUY" if quantity > 0 else "SELL"
        order.orderType = "MKT"
        order.totalQuantity = abs(quantity)
        order.account = account
        order.tif = "DAY"
        order.eTransmit = True
        order_id = self._allocate_id()
        event = threading.Event()
        with self._lock:
            self._done[order_id] = event
            self._orders[order_id] = {"status": "Submitted", "filled": 0.0, "avg": 0.0}
            self._errors.pop(order_id, None)
        self.placeOrder(order_id, key.to_ib_contract(), order)
        event.wait(self.request_timeout)
        with self._lock:
            error, state = self._errors.get(order_id), dict(self._orders[order_id])
        self._forget(order_id)
        if error and not state["filled"]:
            raise IbkrRequestError(error)
        return OrderResult(filled=state["filled"], avg_price=state["avg"], status=state["status"])
