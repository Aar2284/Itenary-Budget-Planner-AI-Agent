"""Planner Manager (step 1.10).

Central orchestrator of the manager-worker architecture.  The manager:

1. **Parses** the user's free-text request into structured trip parameters
   (origin, destination, days, budget, interests, transport modes) by
   invoking the LLM with ``PLANNER_SYSTEM``.
2. **Delegates** work sequentially to Flight → Hotel → Activity workers,
   running each result through the Budget Worker's ``gate()`` before
   accepting it into the trip state.
3. **Handles budget rejection** — when the gate rejects or asks for a
   recheck the manager follows the re-plan path:
   - ``needs_recheck``: step one worker back, emit a "recalculating"
     notice, re-run that worker, gate again.
   - ``rejected`` (over budget): ask the LLM for cheaper alternatives
     via ``REPLAN_SYSTEM``, re-delegate affected workers with the
     tighter constraints, max 2 re-plan attempts.
4. **Graceful failure** — if two re-plan rounds still can't fit the
   budget the manager returns a polite "couldn't fit your budget"
   result rather than crashing.

This module is a pure function library (no graph edges) — ``graph.py``
(step 1.11) will wire these into LangGraph nodes.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm import extract_json, invoke_llm, truncate
from backend.prompts import (
    PLANNER_SYSTEM,
    REPLAN_SYSTEM,
    sanitize,
)
from backend.workers.base import EventEmit, default_emit, log_event
from backend.workers.budget_worker import gate


# Maximum re-plan attempts before graceful failure
MAX_REPLAN = 2


# =====================================================================
# 1. Parse user request → trip parameters
# =====================================================================

def parse_request(
    user_query: str,
    emit: EventEmit | None = None,
) -> dict[str, Any]:
    """Use the LLM to extract structured trip parameters from free text.

    Returns a dict matching ``TripState`` seeding fields:
    origin, destination, days, budget_total, currency, interests,
    transport_modes — plus ``rationale`` for observability.

    If the LLM fails or returns garbage the function still returns a
    best-effort dict with sensible defaults so the flow doesn't crash.
    """
    emit = emit or default_emit
    emit("agent_start", "Planner Manager is understanding your request…")

    safe_query = sanitize(user_query)
    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=safe_query),
    ]

    try:
        response = invoke_llm(messages)
        content = getattr(response, "content", "") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        parsed = extract_json(str(content))
    except Exception as exc:
        emit("agent_error", f"Could not parse request: {exc}")
        parsed = {}

    # Normalise & enforce types with safe defaults
    plan: dict[str, Any] = {
        "origin": str(parsed.get("origin") or ""),
        "destination": str(parsed.get("destination") or ""),
        "days": _safe_int(parsed.get("days"), default=3),
        "budget_total": _safe_float(parsed.get("budget_total"), default=0.0),
        "currency": str(parsed.get("currency") or "USD"),
        "interests": _safe_string_list(parsed.get("interests")),
        "transport_modes": _safe_string_list(
            parsed.get("transport_modes"), default=["flight"]
        ),
        "rationale": str(parsed.get("rationale") or ""),
    }

    emit(
        "agent_thought",
        f"Parsed trip: {plan['origin']} → {plan['destination']}, "
        f"{plan['days']} day(s), budget {plan['budget_total']:,.0f} "
        f"{plan['currency']}, interests: {plan['interests']}",
    )

    return plan


# =====================================================================
# 2. Delegate to a single worker + gate the result
# =====================================================================

def delegate_and_gate(
    worker_fn: Callable[..., dict[str, Any]],
    worker_name: str,
    state: dict[str, Any],
    category: str,
    items_key: str,
    emit: EventEmit | None = None,
) -> dict[str, Any]:
    """Run *one* worker and push its output through the Budget Worker gate.

    Parameters
    ----------
    worker_fn : callable
        One of ``flight_worker``, ``hotel_worker``, ``activity_worker``.
    worker_name : str
        Human label for logging (``"flight_worker"`` etc.).
    state : dict
        Current trip state (will NOT be mutated).
    category : str
        Budget gate category (``"flights"``, ``"hotels"``, ``"activities"``).
    items_key : str
        State key written by this worker (``"flight_items"`` etc.).
    emit : callback
        Event emitter for the UI.

    Returns
    -------
    dict with keys:
        ``worker_update`` — the raw state-update dict from the worker,
        ``gate_result``   — the gate decision dict,
        ``items``         — extracted items list ready for ``chosen_items``.
    """
    emit = emit or default_emit

    # Run the worker
    worker_update = worker_fn(state, emit=emit)

    # Extract items the worker produced
    raw_items = worker_update.get(items_key)
    items = _normalise_items(raw_items, category)

    if not items:
        # Worker found nothing — gate can't evaluate, treat as needs_recheck
        gate_result: dict[str, Any] = {
            "approved": False,
            "needs_recheck": True,
            "running_total": None,
            "remaining": None,
            "reason": f"No {category} results to evaluate.",
            "suggestion": f"Re-run {worker_name} with a broader search.",
        }
    else:
        # Compute aggregate cost for the addition
        total_cost = _items_total_cost(items)
        addition: dict[str, Any] = {
            "category": category,
            "items": items,
            "cost": total_cost,
        }
        gate_result = gate(
            budget_total=float(state.get("budget_total") or 0),
            chosen_items=dict(state.get("chosen_items") or {}),
            addition=addition,
        )

    # Emit gate result
    if gate_result.get("approved"):
        emit("budget_update", f"✅ {category.title()} approved — {gate_result.get('reason', '')}")
    elif gate_result.get("needs_recheck"):
        emit("agent_retry", f"⟳ {category.title()} needs recheck — {gate_result.get('reason', '')}")
    else:
        emit("agent_error", f"❌ {category.title()} rejected — {gate_result.get('reason', '')}")

    return {
        "worker_update": worker_update,
        "gate_result": gate_result,
        "items": items,
    }


# =====================================================================
# 3. Re-plan path (cheaper alternatives)
# =====================================================================

def build_replan_instructions(
    state: dict[str, Any],
    gate_result: dict[str, Any],
    emit: EventEmit | None = None,
) -> dict[str, Any]:
    """Ask the LLM for cheaper-alternative instructions.

    Returns a dict with keys: ``cheaper_flight``, ``cheaper_hotel``,
    ``cheaper_activities``, ``max_nights_hint``, ``note``.
    """
    emit = emit or default_emit
    emit("agent_start", "Planner Manager is replanning for a tighter budget…")

    budget_total = state.get("budget_total", 0)
    reject_reason = gate_result.get("reason", "Over budget")
    suggestion = gate_result.get("suggestion", "")

    context = (
        f"Budget: {budget_total:,.0f} {state.get('currency', '')}\n"
        f"Rejection reason: {reject_reason}\n"
        f"Suggestion: {suggestion}\n"
        f"Destination: {state.get('destination', '')}\n"
        f"Days: {state.get('days', 3)}\n"
    )

    messages = [
        SystemMessage(content=REPLAN_SYSTEM),
        HumanMessage(content=truncate(context, 1200)),
    ]

    try:
        response = invoke_llm(messages)
        content = getattr(response, "content", "") or ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        plan = extract_json(str(content))
    except Exception as exc:
        emit("agent_error", f"Replan LLM call failed: {exc}")
        plan = {}

    result = {
        "cheaper_flight": str(plan.get("cheaper_flight") or ""),
        "cheaper_hotel": str(plan.get("cheaper_hotel") or ""),
        "cheaper_activities": str(plan.get("cheaper_activities") or ""),
        "max_nights_hint": plan.get("max_nights_hint"),
        "note": str(plan.get("note") or "Adjusting plan to fit your budget."),
    }

    emit(
        "agent_thought",
        f"Replan note: {result['note']}",
    )
    return result


# =====================================================================
# 4. Full orchestration: parse → delegate → gate → replan loop
# =====================================================================

def run_manager(
    user_query: str,
    worker_fns: dict[str, Callable[..., dict[str, Any]]] | None = None,
    emit: EventEmit | None = None,
) -> dict[str, Any]:
    """End-to-end manager orchestration.

    This is the high-level function called by the graph (step 1.11).
    It runs the full parse → delegate → gate → replan pipeline and
    returns the final state update.

    Parameters
    ----------
    user_query : str
        The user's free-text trip request.
    worker_fns : dict, optional
        Map of worker names to callables.  Defaults to importing the
        real workers.  Pass stubs in tests.
    emit : callable, optional
        Event emitter for the UI.

    Returns
    -------
    dict — state update with all worker outputs merged, ``chosen_items``
    populated, and ``phase`` set to ``"assembling"`` (ready for the
    assembler) or ``"failed"`` if the budget could never be met.
    """
    emit = emit or default_emit

    # Late imports to avoid circular dependencies (graph imports manager,
    # workers don't import manager).
    if worker_fns is None:
        from backend.workers.flight_worker import flight_worker
        from backend.workers.hotel_worker import hotel_worker
        from backend.workers.activity_worker import activity_worker

        worker_fns = {
            "flight_worker": flight_worker,
            "hotel_worker": hotel_worker,
            "activity_worker": activity_worker,
        }

    # ---- step 1: parse ----
    plan = parse_request(user_query, emit=emit)

    state: dict[str, Any] = {
        "user_query": user_query,
        "origin": plan["origin"],
        "destination": plan["destination"],
        "days": plan["days"],
        "budget_total": plan["budget_total"],
        "currency": plan["currency"],
        "interests": plan["interests"],
        "transport_modes": plan["transport_modes"],
        "budget_remaining": plan["budget_total"],
        "chosen_items": {"flights": [], "hotels": [], "activities": [], "transport": []},
        "current_agent": "manager",
        "agent_log": [],
        "error_log": [],
        "replan_count": 0,
        "budget_approved": False,
        "phase": "delegating",
        "llm_calls": 1,  # parse_request used 1 LLM call
    }

    # Worker pipeline in delegation order
    worker_pipeline = [
        ("flight_worker", "flights", "flight_items"),
        ("hotel_worker", "hotels", "hotel_items"),
        ("activity_worker", "activities", "activity_items"),
    ]

    # ---- step 2: delegate + gate with replan loop ----
    replan_count = 0

    for worker_name, category, items_key in worker_pipeline:
        worker_fn = worker_fns[worker_name]
        attempt = 0
        max_attempts = MAX_REPLAN + 1  # 1 initial + 2 replans

        while attempt < max_attempts:
            result = delegate_and_gate(
                worker_fn=worker_fn,
                worker_name=worker_name,
                state=state,
                category=category,
                items_key=items_key,
                emit=emit,
            )

            gate_result = result["gate_result"]

            # Merge worker state updates
            for key, value in result["worker_update"].items():
                if key != "agent_log":  # merge logs separately
                    state[key] = value
            state["agent_log"] = list(state.get("agent_log") or []) + list(
                result["worker_update"].get("agent_log") or []
            )

            if gate_result.get("approved"):
                # Accept items into chosen_items
                state["chosen_items"][category] = list(result["items"])
                state["budget_remaining"] = gate_result.get("remaining", state["budget_remaining"])
                state["agent_log"] = log_event(
                    state, "manager", f"gate_{category}", "approved"
                )
                break

            elif gate_result.get("needs_recheck"):
                # Step back one: re-run this worker
                attempt += 1
                emit(
                    "status",
                    f"Recalculating {category} with verified numbers "
                    f"(attempt {attempt + 1}/{max_attempts})…",
                )
                state["agent_log"] = log_event(
                    state, "manager", f"recheck_{category}", "retry"
                )
                continue

            else:
                # Rejected on budget — attempt a replan
                attempt += 1
                replan_count += 1
                state["replan_count"] = replan_count

                if attempt >= max_attempts:
                    # Exhausted replans for this worker
                    emit(
                        "agent_error",
                        f"Could not fit {category} within budget after "
                        f"{MAX_REPLAN} replan attempt(s).",
                    )
                    state["error_log"] = list(state.get("error_log") or []) + [
                        {
                            "agent": "manager",
                            "error": f"{category} budget overflow after {MAX_REPLAN} replans",
                            "action_taken": "graceful_skip",
                            "ts": time.time(),
                        }
                    ]
                    # Accept what we can (empty list) — don't crash
                    state["chosen_items"][category] = []
                    state["agent_log"] = log_event(
                        state, "manager", f"gate_{category}", "skipped"
                    )
                    break

                # Ask LLM for cheaper instructions and retry
                replan = build_replan_instructions(state, gate_result, emit=emit)
                state["llm_calls"] = state.get("llm_calls", 0) + 1
                state["phase"] = "replanning"
                emit(
                    "status",
                    f"💡 {replan.get('note', 'Adjusting plan…')}",
                )

                # Inject replan hints into user query for the next worker run
                hint_key = {
                    "flights": "cheaper_flight",
                    "hotels": "cheaper_hotel",
                    "activities": "cheaper_activities",
                }.get(category, "")

                if hint_key and replan.get(hint_key):
                    state["user_query"] = (
                        f"{state['user_query']} "
                        f"[Budget constraint: {replan[hint_key]}]"
                    )

                state["agent_log"] = log_event(
                    state, "manager", f"replan_{category}", "retry"
                )
                continue

    # ---- step 3: determine final phase ----
    has_any = any(
        bool(state["chosen_items"].get(cat))
        for cat in ("flights", "hotels", "activities")
    )

    if has_any:
        state["phase"] = "assembling"
        state["budget_approved"] = True
        emit("status", "All workers done — assembling your itinerary…")
    else:
        state["phase"] = "failed"
        state["budget_approved"] = False
        emit(
            "agent_error",
            "I'm sorry — I couldn't build an itinerary within your budget. "
            "Try increasing your budget or shortening the trip.",
        )

    state["current_agent"] = "manager"
    state["agent_log"] = log_event(state, "manager", "orchestration", state["phase"])

    return state


# =====================================================================
# Helpers (private)
# =====================================================================

def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce to int, with fallback."""
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce to float, with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_string_list(value: Any, default: list[str] | None = None) -> list[str]:
    """Coerce to list[str], with fallback."""
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str) and value:
        return [value]
    return list(default or [])


def _normalise_items(raw: Any, category: str) -> list[dict[str, Any]]:
    """Flatten worker output into a list of item dicts.

    Activity worker returns ``{days: [{day, activities:[...]}]}``
    while flight/hotel return a flat list.  This normalises both.
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        # Activity worker shape: {days: [{day: 1, activities: [...]}], ...}
        if "days" in raw:
            items: list[dict[str, Any]] = []
            for day_card in raw.get("days", []):
                day_num = day_card.get("day", 1)
                for act in day_card.get("activities", []):
                    item = dict(act)
                    item["day"] = day_num
                    items.append(item)
            return items
        return [raw]
    if isinstance(raw, list):
        return list(raw)
    return []


def _items_total_cost(items: list[dict[str, Any]]) -> float | None:
    """Sum real costs across items; None if ANY item lacks a price."""
    total = 0.0
    for item in items:
        cost = item.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
            total += float(cost)
        else:
            return None  # at least one missing → gate uses needs_recheck
    return total
