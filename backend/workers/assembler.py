"""Itinerary Assembler (step 1.9b) — structured itinerary JSON.

Deterministic (no LLM): merges the **gated/approved** items in the
shared state into the final ``itinerary_json`` document the UI renders
as day-cards and the evals verify against.

Honesty rules (same as the Budget Worker):
- ``totals.verified_cost`` sums ONLY real (non-None) prices.
- Items without a confirmed price are listed under
  ``totals.unverified_items`` and shown as "price unverified" — never
  given an invented number.
"""

from __future__ import annotations

from typing import Any

from backend.workers.base import log_event


def _activity_day(item: dict[str, Any], fallback: int) -> int:
    day = item.get("day")
    if isinstance(day, int) and day >= 1:
        return day
    return fallback


def assemble_itinerary(state: dict[str, Any]) -> dict[str, Any]:
    """Build ``itinerary_json`` from gated items in shared state.

    Returns a state-update dict containing ``itinerary_json`` (and the
    usual observability fields).
    """
    days_count = max(int(state.get("days") or 1), 1)
    currency = state.get("currency")
    budget_total = float(state.get("budget_total") or 0)
    chosen: dict[str, list[dict[str, Any]]] = state.get("chosen_items") or {}

    flights = list(chosen.get("flights") or [])
    hotels = list(chosen.get("hotels") or [])
    activities = list(chosen.get("activities") or [])
    transport = list(chosen.get("transport") or [])

    # ---- day cards -------------------------------------------------
    day_cards: list[dict[str, Any]] = [
        {"day": d, "summary": "", "activities": []} for d in range(1, days_count + 1)
    ]

    # flights: outbound on day 1, return on last day when 2+ options
    for idx, flight in enumerate(flights):
        target = 0 if idx == 0 else (days_count - 1 if idx == 1 else 0)
        day_cards[target]["activities"].append(
            {
                "time": "—",
                "activity": f"✈ {flight.get('description', 'Flight')}",
                "cost": flight.get("cost"),        # None stays None
                "currency": flight.get("currency") or currency,
                "kind": "flight",
            }
        )

    # hotels: attach stay line to every day
    for hotel in hotels:
        for card in day_cards:
            card["activities"].append(
                {
                    "time": "—",
                    "activity": (
                        f"🏨 {hotel.get('name', 'Stay')}"
                        + (f" — {hotel.get('why')}" if hotel.get("why") else "")
                    ),
                    "cost": None,   # per-night split would misstate totals;
                    "currency": currency,  # hotel total counted in totals below
                    "kind": "hotel",
                }
            )

    # activities: by their own day field, else round-robin
    for idx, act in enumerate(activities):
        day = _activity_day(act, (idx % days_count) + 1)
        card = day_cards[day - 1]
        card["activities"].append(
            {
                "time": act.get("time") or "—",
                "activity": act.get("activity", "Activity"),
                "cost": act.get("cost"),           # None stays None
                "currency": act.get("currency") or currency,
                "kind": "activity",
                "interest_tag": act.get("interest_tag"),
            }
        )

    # local transport: first day
    for item in transport:
        day_cards[0]["activities"].append(
            {
                "time": "—",
                "activity": f"🚕 {item.get('description', 'Local transport')}",
                "cost": item.get("cost"),
                "currency": item.get("currency") or currency,
                "kind": "transport",
            }
        )

    # summaries
    for card in day_cards:
        n = len([a for a in card["activities"] if a["kind"] == "activity"])
        card["summary"] = f"{n} planned activity/activities"

    # ---- totals (real numbers only) --------------------------------
    def _real(value: Any) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return float(value)
        return None

    verified = 0.0
    unverified = 0

    for group in (flights, hotels, transport):
        for item in group:
            cost = _real(item.get("cost"))
            if cost is None:
                unverified += 1
            else:
                verified += cost

    for act in activities:
        cost = _real(act.get("cost"))
        if cost is None:
            unverified += 1
        else:
            verified += cost

    within_budget = verified <= budget_total if budget_total else False

    itinerary_json: dict[str, Any] = {
        "trip": {
            "origin": state.get("origin"),
            "destination": state.get("destination"),
            "days": days_count,
            "interests": list(state.get("interests") or []),
            "budget_total": budget_total,
            "currency": currency,
        },
        "flights": flights,
        "hotels": hotels,
        "transport": transport,
        "days": day_cards,
        "totals": {
            "verified_cost": round(verified, 2),
            "unverified_items": unverified,
            "budget_total": budget_total,
            "remaining_verified": round(max(budget_total - verified, 0.0), 2),
            "within_budget": within_budget,
            "budget_check_complete": unverified == 0,
            "currency": currency,
        },
    }

    return {
        "itinerary_json": itinerary_json,
        "current_agent": "assembler",
        "agent_log": log_event(state, "assembler", "assemble_itinerary", "done"),
        "llm_calls": state.get("llm_calls", 0),
    }
