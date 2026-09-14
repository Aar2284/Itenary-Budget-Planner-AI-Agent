"""Hotel Worker (step 1.8).

Finds stay options via Tavily web search and structures them for the
shared trip-state.

- Same ``emit(event_type, detail)`` callback contract as the Flight
  Worker so the UI can show live agent activity.
- Never invents a price: ``cost`` stays ``None`` unless a real figure
  came back from search results — the Budget Worker then raises
  ``needs_recheck``.
- On tool failure, emits ``agent_error`` and raises ``WorkerError`` so
  the manager can run the step-back / recheck path.
"""

from __future__ import annotations

import re
from typing import Any

from tools.tavily_tool import tavily_search

from backend.prompts import sanitize
from backend.workers.base import EventEmit, WorkerError, default_emit, log_event


# Real money figures only — same discipline as the Budget Worker.
_MONEY_RE = re.compile(
    r"(?<![\w.])(?:[¤$€£¥]\s?|"
    r"\b(?:USD|BDT|INR|EUR|GBP|JPY|AED|SGD|THB|MYR)\s)"
    r"([\d][\d,]{1,9})(?:\s?(?:per night|/night|per person|per day))?",
    re.IGNORECASE,
)


def extract_prices(text: str) -> list[float]:
    """Pull real per-night price figures out of search snippets.

    Returns [] when nothing is found — an empty list is honest; a made
    up number never is.
    """
    prices: list[float] = []
    for match in _MONEY_RE.finditer(text or ""):
        value = float(match.group(1).replace(",", ""))
        if 0 < value <= 1_000_000:
            prices.append(value)
    return prices


def _first_price(text: str) -> float | None:
    prices = extract_prices(text)
    return min(prices) if prices else None


def parse_hotel_snippets(raw: str, max_items: int = 4) -> list[dict[str, Any]]:
    """Structure Tavily result lines into hotel items.

    Input lines look like: ``1. **Title**\\n   url\\n   snippet``
    """
    if not raw:
        return []

    items: list[dict[str, Any]] = []
    chunks = re.split(r"\n\s*\d+\.\s+", "\n" + raw)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        title_match = re.match(r"\*\*(.+?)\*\*", chunk)
        title = title_match.group(1) if title_match else chunk.splitlines()[0][:80]
        price = _first_price(chunk)
        items.append(
            {
                "type": "hotel",
                "name": title,
                "location": None,
                "nights": None,          # filled by manager from trip days
                "cost": price,           # None unless a REAL price was found
                "currency": None,
                "why": "Found via web search for this trip.",
                "source": chunk[:300],
            }
        )
        if len(items) >= max_items:
            break
    return items


def hotel_worker(
    state: dict[str, Any],
    emit: EventEmit | None = None,
) -> dict[str, Any]:
    """Search stays for the trip and update the shared state."""
    emit = emit or default_emit
    query = sanitize(state.get("user_query", ""))
    budget = state.get("budget_total") or 0
    destination = state.get("destination") or ""

    emit("agent_start", "Hotel Worker is searching stays…")
    emit(
        "agent_thought",
        f"Looking for stays in {destination or 'the destination'} "
        f"(budget {budget:,.0f} {state.get('currency', '')})",
    )

    search_query = f"best hotels {query}".strip()
    try:
        raw = tavily_search(search_query)
    except Exception as exc:
        emit("agent_error", f"Hotel search failed: {exc}")
        raise WorkerError(f"hotel tool failed: {exc}") from exc

    items = parse_hotel_snippets(raw)

    priced = sum(1 for item in items if item["cost"] is not None)
    emit(
        "agent_thought",
        f"Collected {len(items)} option(s); {priced} with verified "
        "prices — Budget Worker must gate before accepting.",
    )

    if items:
        emit("agent_start", "Hotel Worker handed options to the Budget Worker…")
    else:
        emit(
            "agent_thought",
            "No usable snippets returned — Budget Worker will flag "
            "for recheck.",
        )

    return {
        "hotel_results": raw,
        "hotel_items": items,
        "current_agent": "hotel_worker",
        "agent_log": log_event(state, "hotel_worker", "tavily_search", "done"),
        "llm_calls": state.get("llm_calls", 0),
    }
