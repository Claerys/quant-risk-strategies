"""Connection settings, read from the environment, and the paper-only guard.

The socket needs no username or password: IB Gateway / TWS holds the login. Only paper ports
are accepted anywhere in this package, so a typo cannot point the code at a live account.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

# IB Gateway paper, TWS paper. Live is 4001 (Gateway) and 7496 (TWS) and is refused.
PAPER_PORTS = frozenset({4002, 7497})


def require_paper_port(port: int) -> None:
    if port not in PAPER_PORTS:
        raise ValueError(
            f"port {port} is not a paper-trading port; only {sorted(PAPER_PORTS)} are allowed "
            "(live ports 4001 and 7496 are refused)"
        )


@dataclass(frozen=True)
class IbkrSettings:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 1
    timeout: float = 60.0
    account: str | None = None

    def __post_init__(self) -> None:
        require_paper_port(self.port)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> IbkrSettings:
        env = os.environ if env is None else env
        return cls(
            host=env.get("IBKR_HOST", cls.host),
            port=int(env.get("IBKR_PORT", cls.port)),
            client_id=int(env.get("IBKR_CLIENT_ID", cls.client_id)),
            timeout=float(env.get("IBKR_REQUEST_TIMEOUT", cls.timeout)),
            account=env.get("IBKR_ACCOUNT") or None,
        )
