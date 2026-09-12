"""Travel Itinerary Planning Concierge — backend package.

Manager-worker multi-agent architecture built on LangGraph.

During the migration from the monolithic ``backend.py`` this package
bridges attribute access to the legacy module, so imports like
``from backend import run_travel_agent`` keep working until
``backend/graph.py`` (step 1.11) and the legacy removal (step 1.14).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

__all__ = [
    "TripState",
    "run_travel_agent",
    "travel_graph",
]

_LAZY: dict[str, tuple[str, str]] = {
    "TripState": ("backend.state", "TripState"),
    "run_travel_agent": ("backend.graph", "run_travel_agent"),
    "travel_graph": ("backend.graph", "travel_graph"),
}

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "backend.py"
_legacy_cache: types.ModuleType | None = None


def _load_legacy() -> types.ModuleType | None:
    """Load root backend.py (monolith) under a private module name."""
    global _legacy_cache
    if _legacy_cache is not None:
        return _legacy_cache
    if not _LEGACY_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("backend._legacy", _LEGACY_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["backend._legacy"] = module
    spec.loader.exec_module(module)
    _legacy_cache = module
    return module


class _BackendPackage(types.ModuleType):
    """Module class that resolves attrs: new modules first, legacy fallback."""

    def __getattr__(self, name: str) -> Any:
        if name in _LAZY:
            module_name, attr = _LAZY[name]
            try:
                return getattr(importlib.import_module(module_name), attr)
            except ModuleNotFoundError:
                pass  # not built yet — fall through to legacy
        legacy = _load_legacy()
        if legacy is not None:
            try:
                return getattr(legacy, name)
            except AttributeError:
                pass
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        # Forward writes (e.g. monkeypatch.setattr(backend, "llm", ...))
        # to the legacy module when it owns that attribute, so patched
        # values are seen by legacy functions.
        legacy = _load_legacy()
        if legacy is not None and name in vars(legacy):
            setattr(legacy, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _BackendPackage
