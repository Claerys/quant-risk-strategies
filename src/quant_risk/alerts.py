"""Alerts at the end of each trading cycle.

Always printed. Also sent to Telegram when TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set in the
environment or .env (see .env.example). A failed send is reported, never raised: an alert channel
going down must not stop the risk checks that produced the alert.
"""

from __future__ import annotations

import json
import os
import urllib.request

TELEGRAM_LIMIT = 4000


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from .env into the environment, without overriding real variables."""
    from quant_risk.bars import REPO_ROOT

    path = REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


def send(text: str) -> list[str]:
    """Deliver `text` to every configured channel; returns one status line per channel."""
    _load_dotenv()
    print(text)
    statuses = ["console: ok"]
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat:
        try:
            request = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=json.dumps({"chat_id": chat, "text": text[:TELEGRAM_LIMIT]}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                statuses.append(f"telegram: {'ok' if response.status == 200 else response.status}")
        except OSError as exc:
            statuses.append(f"telegram: failed ({exc.__class__.__name__})")
    return statuses
