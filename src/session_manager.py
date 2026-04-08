"""
session_manager.py

Manages loaded course indices within a chat session. Supports lazy loading,
mid-session course switching, and basic LRU eviction under a memory budget.

Glues together: IndexRegistry (what exists), MemoryTracker (how much RAM),
and the artifact loading logic.
"""

from __future__ import annotations

import gc
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.index_registry import IndexRegistry, IndexEntry
from src.memory_tracker import MemoryTracker

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Tracks loaded course indices and enforces a memory budget with LRU eviction.

    Usage:
        sm = SessionManager(registry, tracker, load_fn, budget_mb=512)
        artifacts = sm.get_or_load("databases")   # lazy load
        artifacts = sm.get_or_load("algorithms")   # may evict "databases" if over budget
        sm.status()                                 # print loaded indices
    """

    def __init__(
        self,
        registry: IndexRegistry,
        tracker: MemoryTracker,
        load_fn: Callable[[str, IndexEntry], Dict[str, Any]],
        budget_mb: Optional[float] = None,
    ):
        self._registry = registry
        self._tracker = tracker
        self._load_fn = load_fn  # (course_name, entry) -> artifacts dict
        self._loaded: Dict[str, Dict[str, Any]] = {}  # course -> artifacts
        self._last_access: Dict[str, float] = {}  # course -> timestamp
        self._active_course: Optional[str] = None

        if budget_mb is not None:
            self._tracker.budget = int(budget_mb * 1024 * 1024)

    # ---- public API ----

    @property
    def active_course(self) -> Optional[str]:
        return self._active_course

    def get_or_load(self, course: str) -> Dict[str, Any]:
        """Return artifacts for *course*, loading from disk if needed."""
        if course in self._loaded:
            self._touch(course)
            self._active_course = course
            return self._loaded[course]

        entry = self._registry.get(course)
        if entry is None:
            raise KeyError(
                f"Course '{course}' not found in registry. "
                f"Available: {', '.join(self._registry.list_courses())}"
            )

        # Use on-disk file size as the cost estimate
        estimated_bytes = entry.disk_size_bytes()

        # Evict until we have room for the incoming index
        self._evict_if_needed(estimated_bytes)

        artifacts = self._load_fn(course, entry)
        self._loaded[course] = artifacts
        self._touch(course)
        self._active_course = course
        return artifacts

    def evict(self, course: str) -> bool:
        """Manually evict a loaded course. Returns True if it was loaded."""
        if course not in self._loaded:
            return False

        del self._loaded[course]
        self._last_access.pop(course, None)
        self._tracker.track_unload(course)
        gc.collect()

        if self._active_course == course:
            self._active_course = None

        logger.info("Evicted '%s'", course)
        print(f"Evicted index for '{course}'.")
        return True

    def loaded_courses(self) -> List[str]:
        """Return list of currently loaded course names."""
        return list(self._loaded.keys())

    def status(self):
        """Print current session state."""
        loaded = self.loaded_courses()
        if not loaded:
            print("No indices loaded.")
            return

        print(f"Active course: {self._active_course or '(none)'}")
        print(f"Loaded indices ({len(loaded)}):")
        for course in loaded:
            mb = self._tracker.get_entry_mb(course)
            last = self._last_access.get(course, 0)
            ago = time.time() - last
            marker = " *" if course == self._active_course else ""
            print(f"  {course}: {mb:.1f} MB, last accessed {ago:.0f}s ago{marker}")

        total = self._tracker.tracked_usage_mb()
        if self._tracker.budget is not None:
            budget_mb = self._tracker.budget / (1024 * 1024)
            print(f"Memory: {total:.1f} / {budget_mb:.0f} MB")
        else:
            print(f"Memory: {total:.1f} MB (no budget set)")

    # ---- internal ----

    def _touch(self, course: str):
        """Update last access timestamp."""
        self._last_access[course] = time.time()

    def _lru_course(self) -> Optional[str]:
        """Return the least recently used loaded course, excluding active."""
        candidates = {
            c: t for c, t in self._last_access.items()
            if c in self._loaded and c != self._active_course
        }
        if not candidates:
            # If all loaded courses are active (only one), allow evicting it
            candidates = {
                c: t for c, t in self._last_access.items()
                if c in self._loaded
            }
        if not candidates:
            return None
        return min(candidates, key=candidates.get)

    def _evict_if_needed(self, incoming_bytes: int = 0):
        """Evict LRU courses until there's room for *incoming_bytes*."""
        if self._tracker.budget is None:
            return

        while self._tracker.would_exceed_budget(incoming_bytes):
            victim = self._lru_course()
            if victim is None:
                logger.warning("Over budget but nothing to evict")
                break
            self.evict(victim)
