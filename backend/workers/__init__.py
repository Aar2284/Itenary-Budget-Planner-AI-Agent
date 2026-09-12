"""Workers package (step 1.5).

Each worker is an independent module with a single public entrypoint.
The Budget Worker is the only component allowed to mutate
``budget_remaining`` / reject additions — and it never invents numbers:
missing prices come back as ``needs_recheck`` for a re-fetch.
"""

from backend.workers.budget_worker import (
    CATEGORIES,
    gate,
    missing_costs,
    real_cost,
    running_total,
)

__all__ = [
    "CATEGORIES",
    "gate",
    "missing_costs",
    "real_cost",
    "running_total",
]
