"""Budget Worker (step 1.6) — the gatekeeper.

Pure, deterministic functions (no LLM, no I/O) so the gate is always
testable and never hallucinates arithmetic.

**No invented numbers.** Every cost must come from a worker's real,
verified data. When a price is unavailable the gate returns a
``needs_recheck`` decision instead of approving: the manager then steps
one agent back, shows the user a plain "recalculating" notice, the
worker re-fetches real numbers, and only then does the flow continue.

- ``real_cost()``      — extract a verified numeric cost (None if absent)
- ``missing_costs()``  — which chosen categories still lack real prices
- ``running_total()``  — sum of verified real costs only
- ``gate()``           — approve / reject / needs_recheck + reason
"""

from __future__ import annotations

from typing import Any

CATEGORIES = ("flights", "hotels", "activities", "transport")

# Cheapening advice per category, used for gate rejections
_CHEAPER_HINTS: dict[str, str] = {
    "flights": "prefer economy, midweek departures, 1-stop options",
    "hotels": "switch to guesthouses/hostels or stay outside the center",
    "activities": "favor free walking tours, parks and free-entry museums",
    "transport": "use local trains/buses instead of taxis",
}

_RECHECK_REASON = (
    "A required price could not be confirmed, so the running total "
    "cannot be trusted yet. Recalculating with verified numbers."
)


def real_cost(item: dict[str, Any]) -> float | None:
    """Return the item's verified cost, or None when it is not a real number."""
    cost = item.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
        return float(cost)
    return None


def missing_costs(
    chosen_items: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Categories among chosen items that still lack a verified price."""
    missing: list[str] = []
    for category in CATEGORIES:
        for item in (chosen_items or {}).get(category, []):
            if real_cost(item) is None:
                if category not in missing:
                    missing.append(category)
    return missing


def running_total(chosen_items: dict[str, list[dict[str, Any]]]) -> float:
    """Sum of verified real costs only.

    Items without a real price contribute nothing — callers must check
    ``missing_costs()`` first (``gate()`` does) so unknowns never slip
    through as a silently-low total.
    """
    total = 0.0
    for category in CATEGORIES:
        for item in (chosen_items or {}).get(category, []):
            cost = real_cost(item)
            if cost is not None:
                total += cost
    return total


def _needs_recheck(missing: list[str], category: str) -> dict[str, Any]:
    return {
        "approved": False,
        "needs_recheck": True,
        "running_total": None,
        "remaining": None,
        "reason": _RECHECK_REASON,
        "suggestion": (
            f"Re-fetch verified prices for: {', '.join(dict.fromkeys(missing + [category]))}."
        ),
    }


def gate(
    budget_total: float,
    chosen_items: dict[str, list[dict[str, Any]]],
    addition: dict[str, Any],
) -> dict[str, Any]:
    """Decide whether ``addition`` may join ``chosen_items``.

    ``addition`` shape: {"category": "flights"|"hotels"|"activities"|"transport",
                         "items": [ {...}, ... ], "cost": float|None}

    Returns the Budget-Worker decision dict:
    - approved=True,  needs_recheck=False — within budget (real numbers)
    - approved=False, needs_recheck=False — over budget, with cheaper suggestion
    - approved=False, needs_recheck=True  — a price is missing; manager must
      step back one agent, show the "recalculating" excuse, and re-fetch
      real numbers before continuing
    """
    category = addition.get("category", "activities")
    if category not in CATEGORIES:
        category = "activities"

    # 1) the proposed addition itself must carry a verified cost
    items = addition.get("items") or []
    addition_unknown = real_cost(addition) is None and all(
        real_cost(item) is None for item in items
    )
    per_item_unknown = any(real_cost(item) is None for item in items)
    if addition_unknown or per_item_unknown:
        return _needs_recheck([category], category)

    # cost may be given directly on the addition, else sum its items
    proposed = real_cost(addition)
    if proposed is None:
        proposed = sum(real_cost(item) or 0.0 for item in items)

    # 2) already-chosen items must all be verified too
    missing = missing_costs(chosen_items)
    if missing:
        return _needs_recheck(missing, category)

    # 3) pure arithmetic on real numbers
    spent = running_total(chosen_items)
    projected = spent + proposed
    remaining = float(budget_total) - projected

    if projected <= budget_total:
        return {
            "approved": True,
            "needs_recheck": False,
            "running_total": round(projected, 2),
            "remaining": round(max(remaining, 0.0), 2),
            "reason": (
                f"Within budget: {projected:,.0f} of {budget_total:,.0f} "
                f"({max(remaining, 0):,.0f} left)."
            ),
            "suggestion": "",
        }

    over = projected - budget_total
    noun = category[:-1] if category.endswith("s") else category
    hint = _CHEAPER_HINTS.get(category, "choose a cheaper option")
    return {
        "approved": False,
        "needs_recheck": False,
        "running_total": round(projected, 2),
        "remaining": round(remaining, 2),
        "reason": (
            f"Rejected: adding this {noun} would total {projected:,.0f}, "
            f"which is {over:,.0f} over the {budget_total:,.0f} budget."
        ),
        "suggestion": f"Ask the worker to {hint}.",
    }
