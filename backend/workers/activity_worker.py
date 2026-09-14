"""Activity Worker (step 1.9).

Proposes day-by-day activities matching the user's stated interests,
using Tavily web search for real suggestions.

- Same ``emit(event_type, detail)`` callback contract as the other
  workers so the UI can show live agent activity.
- Structured output: ``{"days": [{"day": 1, "activities": [...]}]}``
  (the ``itinerary_json`` day-card shape used by UI and evals).
- Never invents a price: ``cost`` stays ``None`` unless a real figure
  appeared in search results — the Budget Worker then raises
  ``needs_recheck``.
- On tool failure, emits ``agent_error`` and raises ``WorkerError``.
"""

from __future__ import annotations

import re
from typing import Any

from tools.tavily_tool import tavily_search

from backend.prompts import sanitize
from backend.workers.base import (
    EventEmit,
    WorkerError,
    default_emit,
    first_price,
    log_event,
)

# Free activities are cost 0 — a real fact, not an estimate.
_FREE_RE = re.compile(
    r"\b(free (?:entry|entrance|walking tour|tour|museum|park|guides?)|"
    r"no entrance fee|free admission)\b",
    re.IGNORECASE,
)

_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")


def _parse_time(text: str, fallback_index: int) -> str:
    match = _TIME_RE.search(text)
    if match:
        return f"{match.group(1).zfill(2)}:{match.group(2)}"
    # deterministic fallback slots — times, not prices, so no honesty issue
    slots = ["09:00", "11:00", "14:00", "16:00", "18:00"]
    return slots[fallback_index % len(slots)]


def parse_activity_snippets(raw: str) -> list[dict[str, Any]]:
    """Structure Tavily result lines into flat activity candidates.

    Each candidate: {activity, cost|None, interest_hint, source}.
    """
    if not raw:
        return []

    candidates: list[dict[str, Any]] = []
    chunks = re.split(r"\n\s*\d+\.\s+", "\n" + raw)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        title_match = re.match(r"\*\*(.+?)\*\*", chunk)
        title = title_match.group(1) if title_match else chunk.splitlines()[0][:80]
        title = re.sub(r"\s*-\s*.*$", "", title).strip()  # drop trailing "- City"

        if _FREE_RE.search(chunk):
            cost: float | None = 0.0  # explicitly free = real zero
        else:
            cost = first_price(chunk)  # real figure or None, never invented

        candidates.append(
            {
                "activity": title,
                "cost": cost,
                "interest_hint": None,  # matched later against state interests
                "source": chunk[:300],
            }
        )
    return candidates


def _interest_match(candidate: dict[str, Any], interests: list[str]) -> str | None:
    """Return the first interest tag appearing in the snippet, else None.

    Tolerates simple singular/plural drift (\"temples\" matches \"temple\").
    """
    text = f"{candidate['activity']} {candidate.get('source', '')}".lower()
    for tag in interests or []:
        if not tag:
            continue
        needle = tag.lower().strip()
        for form in {needle, needle.rstrip("s"), needle + "s"}:
            if form and form in text:
                return tag.lower()
    return None


def build_days(
    candidates: list[dict[str, Any]],
    days: int,
    interests: list[str],
) -> dict[str, Any]:
    """Distribute candidates across trip days (3 activities/day max).

    Only candidates matching a stated interest are scheduled; leftovers
    are returned under "unscheduled" so the manager can re-query rather
    than pad the plan with filler.
    """
    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for cand in candidates:
        tag = _interest_match(cand, interests)
        if tag is not None or not interests:
            item = dict(cand)
            item["interest_tag"] = tag
            matched.append(item)
        else:
            unmatched.append(cand)

    day_count = max(int(days or 1), 1)
    day_cards: list[dict[str, Any]] = []
    cursor = 0
    for d in range(1, day_count + 1):
        activities: list[dict[str, Any]] = []
        for slot in range(3):
            if cursor >= len(matched):
                break
            cand = matched[cursor]
            cursor += 1
            activities.append(
                {
                    "time": _parse_time(cand["activity"], slot),
                    "activity": cand["activity"],
                    "cost": cand["cost"],       # None stays None — no fakes
                    "currency": None,
                    "interest_tag": cand.get("interest_tag"),
                }
            )
        day_cards.append({"day": d, "activities": activities})

    return {
        "days": day_cards,
        "unscheduled": unmatched,
        "matched_count": len(matched),
    }


def activity_worker(
    state: dict[str, Any],
    emit: EventEmit | None = None,
) -> dict[str, Any]:
    """Search interest-matching activities and update the shared state."""
    emit = emit or default_emit
    query = sanitize(state.get("user_query", ""))
    interests = list(state.get("interests") or [])
    days = int(state.get("days") or 1)
    destination = state.get("destination") or ""

    emit("agent_start", "Activity Worker is finding things to do…")
    emit(
        "agent_thought",
        f"Matching interests [{', '.join(interests) or 'any'}] in "
        f"{destination or 'the destination'} over {days} day(s)",
    )

    interest_text = ", ".join(interests) if interests else "popular"
    search_query = f"top things to do {destination} {interest_text} {query}".strip()
    try:
        raw = tavily_search(search_query)
    except Exception as exc:
        emit("agent_error", f"Activity search failed: {exc}")
        raise WorkerError(f"activity tool failed: {exc}") from exc

    candidates = parse_activity_snippets(raw)
    itinerary = build_days(candidates, days, interests)

    emit(
        "agent_thought",
        f"Found {len(candidates)} candidate(s); scheduled "
        f"{itinerary['matched_count']} interest-matched across "
        f"{len(itinerary['days'])} day(s).",
    )

    unpriced = sum(
        1
        for day in itinerary["days"]
        for act in day["activities"]
        if act["cost"] is None
    )
    if unpriced:
        emit(
            "agent_thought",
            f"{unpriced} activity price(s) unconfirmed — Budget Worker "
            "must gate before accepting.",
        )

    if itinerary["matched_count"]:
        emit("agent_start", "Activity Worker handed day-cards to the Budget Worker…")
    else:
        emit(
            "agent_thought",
            "No interest-matched activities found — Budget Worker will "
            "flag for recheck.",
        )

    return {
        "activity_results": raw,
        "activity_items": itinerary,
        "current_agent": "activity_worker",
        "agent_log": log_event(state, "activity_worker", "tavily_search", "done"),
        "llm_calls": state.get("llm_calls", 0),
    }
