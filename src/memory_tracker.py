"""
memory_tracker.py

Tracks memory footprint of loaded indices. Uses file-based sizing (sum of
artifact files on disk) for deterministic, repeatable measurements. RSS
diffing is available as a secondary tool but not used for budget decisions.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Any, Callable, Dict, Optional

import psutil

logger = logging.getLogger(__name__)


class MemoryTracker:
    """Tracks per-resource memory usage for budget enforcement."""

    def __init__(self, budget_mb: Optional[float] = None):
        self._process = psutil.Process(os.getpid())
        self._entries: Dict[str, int] = {}  # name → size in bytes
        self.budget: Optional[int] = (
            int(budget_mb * 1024 * 1024) if budget_mb is not None else None
        )

    # ---- tracking ----

    def track(self, name: str, size_bytes: int):
        """Record a known size for a named resource."""
        self._entries[name] = size_bytes
        logger.info("Tracking '%s': %.1f MB", name, size_bytes / (1024 * 1024))

    def track_unload(self, name: str) -> int:
        """Remove *name* from tracking and return its recorded size."""
        return self._entries.pop(name, 0)

    # ---- RSS (informational, not used for budget) ----

    def measure_rss(self) -> int:
        """Current process RSS in bytes."""
        return self._process.memory_info().rss

    # ---- queries ----

    def tracked_usage(self) -> int:
        """Total tracked memory in bytes."""
        return sum(self._entries.values())

    def tracked_usage_mb(self) -> float:
        return self.tracked_usage() / (1024 * 1024)

    def get_entry_bytes(self, name: str) -> int:
        return self._entries.get(name, 0)

    def get_entry_mb(self, name: str) -> float:
        return self.get_entry_bytes(name) / (1024 * 1024)

    def would_exceed_budget(self, additional_bytes: int = 0) -> bool:
        """Return True if adding *additional_bytes* would bust the budget."""
        if self.budget is None:
            return False
        return self.tracked_usage() + additional_bytes > self.budget

    def entries(self) -> Dict[str, int]:
        """Return a copy of {name: bytes} for all tracked resources."""
        return dict(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __repr__(self) -> str:
        items = ", ".join(
            f"{k}={v / (1024*1024):.1f}MB" for k, v in self._entries.items()
        )
        budget_str = (
            f", budget={self.budget / (1024*1024):.0f}MB"
            if self.budget is not None
            else ""
        )
        return f"MemoryTracker({items}{budget_str})"
