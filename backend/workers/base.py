"""Shared worker scaffolding (step 1.8 support).

Every worker uses the same event contract and error type so the
manager/UI can treat them uniformly.
"""

from __future__ import annotations

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
