"""Flight Worker (step 1.7).

Finds live flight options via ``tools.flight_tool`` and structures them
for the shared trip-state.

- Emits status events through an optional ``emit(event_type, detail)``
  callback so the UI can show "what the agent is doing right now".
- Never invents a price: ``cost`` stays ``None`` when the API does not
  provide one — the Budget Worker then raises ``needs_recheck``.
- On tool failure, emits ``agent_error`` and raises ``WorkerError`` so
  the manager can run the step-back / recheck path.
"""

from __future__ import annotations

from typing import Any

from tools.flight_tool import search_flights

from backend.prompts import sanitize
from backend.workers.base import EventEmit, WorkerError, default_emit, log_event


def parse_flight_text(raw: str) -> list[dict[str, Any]]:
    """Turn the flight tool's text blocks into structured items.

    cost is always None here — AviationStack reports live status data,
    not ticket prices. Real fares (if any) must come from a real source;
    we never estimate them.
    """
    if not raw or raw.startswith(("Flight API error", "No live flight data")):
        return []

    items: list[dict[str, Any]] = []
    for block in raw.split("\n\n---\n\n"):
        block = block.strip()
        # drop route headers like "Live flights from DAC to NRT"
        if "Airline:" in block:
            block = block[block.index("Airline:") :].strip()
        if not block or not block.startswith("Airline:"):
            continue
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip().lower()] = value.strip()
        departure = ""
        arrival = ""
        for line in block.splitlines():
            if line.startswith("- Airport:"):
                if not departure:
                    departure = line.split(":", 1)[1].strip()
                else:
                    arrival = line.split(":", 1)[1].strip()
        description = (
            f"{fields.get('airline', 'Unknown')} "
            f"{fields.get('flight', '')}".strip()
        )
        if departure or arrival:
            description += f" ({departure} → {arrival})"
        items.append(
            {
                "type": "flight",
                "description": description,
                "cost": None,  # no real price available — gate will recheck
                "currency": None,
                "status": fields.get("status", "unknown"),
                "scheduled": fields.get("scheduled", ""),
            }
        )
    return items


def flight_worker(
    state: dict[str, Any],
    emit: EventEmit | None = None,
) -> dict[str, Any]:
    """Search flights for the trip and update the shared state."""
    emit = emit or default_emit
    query = sanitize(state.get("user_query", ""))
    budget = state.get("budget_total") or 0

    emit("agent_start", "Flight Worker is searching live flight options…")
    emit(
        "agent_thought",
        f"Route lookup for '{query[:80]}' "
        f"(budget {budget:,.0f} {state.get('currency', '')})",
    )

    try:
        raw = search_flights(query, limit=5)
    except Exception as exc:  # network / tool blow-up
        emit("agent_error", f"Flight search failed: {exc}")
        raise WorkerError(f"flight tool failed: {exc}") from exc

    items = parse_flight_text(raw)
    emit(
        "agent_thought",
        f"Collected {len(items)} option(s); prices not provided by the "
        "live API — Budget Worker must verify before accepting.",
    )

    # Do NOT auto-accept: present options, let the gate decide.
    # The worker only reports what it found.
    if items:
        emit("agent_start", "Flight Worker handed options to the Budget Worker…")
    else:
        emit(
            "agent_thought",
            "No live flight blocks returned — Budget Worker will flag "
            "for recheck.",
        )

    return {
        "flight_results": raw,
        "flight_items": items,
        "current_agent": "flight_worker",
        "agent_log": log_event(state, "flight_worker", "search_flights", "done"),
        "llm_calls": state.get("llm_calls", 0),
    }
