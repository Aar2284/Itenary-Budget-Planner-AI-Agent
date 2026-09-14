"""Shared worker scaffolding (step 1.8 support).

Every worker uses the same event contract, error type, and real-price
extraction rules so the manager/UI can treat them uniformly.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

# (event_type, detail) — event types: agent_start, agent_thought,
# agent_error, agent_retry (Phase 2 maps these onto the WebSocket).
EventEmit = Callable[[str, str], None]


class WorkerError(RuntimeError):
    """Raised when a worker's tool fails and the step must be retried."""


def default_emit(event_type: str, detail: str) -> None:
    return None


def log_event(state: dict[str, Any], agent: str, action: str, status: str) -> list[dict[str, Any]]:
    """Append one entry to the shared agent_log."""
    entry = {"agent": agent, "action": action, "status": status, "ts": time.time()}
    return list(state.get("agent_log") or []) + [entry]


# ---------------------------------------------------------------------
# Real-price extraction (shared by hotel/activity workers)
# ---------------------------------------------------------------------

# Real money figures only — same discipline as the Budget Worker.
MONEY_RE = re.compile(
    r"(?<![\w.])(?:[¤$€£¥]\s?|"
    r"\b(?:USD|BDT|INR|EUR|GBP|JPY|AED|SGD|THB|MYR)\s)"
    r"([\d][\d,]{1,9})(?:\s?(?:per night|/night|per person|per day|per adult|per child))?",
    re.IGNORECASE,
)


def extract_prices(text: str) -> list[float]:
    """Pull real price figures out of search snippets.

    Returns [] when nothing is found — an empty list is honest; a made
    up number never is.
    """
    prices: list[float] = []
    for match in MONEY_RE.finditer(text or ""):
        value = float(match.group(1).replace(",", ""))
        if 0 < value <= 1_000_000:
            prices.append(value)
    return prices


def first_price(text: str) -> float | None:
    """Cheapest real price in ``text``, or None."""
    prices = extract_prices(text)
    return min(prices) if prices else None
