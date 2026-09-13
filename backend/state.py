"""Shared trip-state — the single source of truth every worker reads/writes.

The manager seeds this state from the user request, each worker appends
its findings, and the Budget Worker is the only component allowed to
change budget_remaining / reject additions.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage


class TripState(TypedDict, total=False):
    # ---- input (seeded by manager) ----
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    origin: str                      # e.g. "DAC"
    destination: str                 # e.g. "Tokyo"
    days: int
    budget_total: float
    currency: str                    # e.g. "BDT"
    interests: list[str]             # e.g. ["food", "temples"]
    transport_modes: list[str]       # subset of ["flight", "train", "taxi", "own_vehicle"]

    # ---- shared memory (workers write, budget gates) ----
    budget_remaining: float
    chosen_items: dict[str, list[dict[str, Any]]]
    # chosen_items shape:
    # {
    #   "flights":  [{type, description, cost}],
    #   "hotels":   [{name, nights, cost, location}],
    #   "activities": [{day, time, activity, cost}],
    #   "transport": [{mode, description, cost}],
    # }

    # ---- worker outputs ----
    flight_results: str
    hotel_results: str
    activity_results: str
    transport_results: str
    flight_items: list[dict[str, Any]]       # structured findings, await gate
    hotel_items: list[dict[str, Any]]
    activity_items: dict[str, Any]           # {"days": [...]} shape
    transport_items: list[dict[str, Any]]
    itinerary_json: dict[str, Any]           # {"days": [{day, activities:[...]}], "total_cost": float}
    itinerary: str                           # final markdown rendering

    # ---- coordination / observability ----
    current_agent: str               # which worker is running right now
    agent_log: list[dict[str, Any]]  # [{agent, action, status, ts}]
    error_log: list[dict[str, Any]]  # [{agent, error, action_taken, ts}]
    replan_count: int
    budget_approved: bool
    budget_reject_reason: str
    phase: str                       # "parsing" | "delegating" | "replanning" | "assembling" | "done"

    # ---- bookkeeping ----
    llm_calls: int
