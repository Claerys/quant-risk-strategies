"""IBKR allows about 60 historical-data requests per 10 minutes; exceeding it returns pacing errors."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class Pacer:
    """Blocks in ``wait()`` so that no more than ``max_requests`` start in any ``window`` seconds."""

    def __init__(self, max_requests: int = 55, window: float = 600.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.max_requests, self.window, self._clock, self._sleep = max_requests, window, clock, sleep
        self._starts: deque[float] = deque()

    def wait(self) -> None:
        now = self._clock()
        while self._starts and now - self._starts[0] >= self.window:
            self._starts.popleft()
        if len(self._starts) >= self.max_requests:
            self._sleep(self.window - (now - self._starts[0]))
            self._starts.popleft()
        self._starts.append(self._clock())
