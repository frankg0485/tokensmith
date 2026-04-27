"""
session_manager.py

Manages loaded course indices within a chat session. Supports lazy loading,
mid-session course switching, and eviction under a memory budget.

Eviction policies:
  - lru:  basic least-recently-used
  - 2q:   LRU-2Q (Johnson & Shasha 1994) — new items enter a FIFO probation
           queue (A1in). A second access promotes them to a protected LRU queue
           (Am). Eviction prefers A1in, falling back to Am. A ghost queue
           (A1out) tracks recently evicted keys so that reloads go straight
           to Am.

Glues together: IndexRegistry (what exists), MemoryTracker (how much RAM),
and the artifact loading logic.
"""

from __future__ import annotations

import gc
import time
import logging
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from src.index_registry import IndexRegistry, IndexEntry
from src.memory_tracker import MemoryTracker

logger = logging.getLogger(__name__)

EVICTION_POLICIES = ("lru", "2q")


class SessionManager:
    """
    Tracks loaded course indices and enforces a memory budget with eviction.

    Usage:
        sm = SessionManager(registry, tracker, load_fn, budget_mb=512, policy="2q")
        artifacts = sm.get_or_load("databases")
        artifacts = sm.get_or_load("algorithms")   # may evict under budget
        sm.status()
    """

    def __init__(
        self,
        registry: IndexRegistry,
        tracker: MemoryTracker,
        load_fn: Callable[[str, IndexEntry], Dict[str, Any]],
        budget_mb: Optional[float] = None,
        policy: str = "lru",
    ):
        if policy not in EVICTION_POLICIES:
            raise ValueError(f"Unknown eviction policy '{policy}'. Choose from: {EVICTION_POLICIES}")

        self._registry = registry
        self._tracker = tracker
        self._load_fn = load_fn
        self._policy = policy
        self._loaded: Dict[str, Dict[str, Any]] = {}  # course -> artifacts
        self._active_course: Optional[str] = None

        if budget_mb is not None:
            self._tracker.budget = int(budget_mb * 1024 * 1024)

        # ---- basic LRU state ----
        self._last_access: Dict[str, float] = {}

        # ---- 2Q state ----
        # A1in: FIFO probation queue — newly loaded items land here
        # Am:   LRU protected queue — items promoted on second access
        # A1out: ghost FIFO — keys evicted from A1in (no data, just names)
        self._a1in: OrderedDict = OrderedDict()   # course -> True (insertion order = FIFO)
        self._am: OrderedDict = OrderedDict()      # course -> True (access order = LRU)
        self._a1out: OrderedDict = OrderedDict()   # course -> True (ghost keys)
        self._a1out_max = 50  # max ghost entries to track

    # ---- public API ----

    @property
    def active_course(self) -> Optional[str]:
        return self._active_course

    @property
    def policy(self) -> str:
        return self._policy

    def get_or_load(self, course: str) -> Dict[str, Any]:
        """Return artifacts for *course*, loading from disk if needed."""
        if course in self._loaded:
            self._on_access(course)
            self._active_course = course
            return self._loaded[course]

        entry = self._registry.get(course)
        if entry is None:
            raise KeyError(
                f"Course '{course}' not found in registry. "
                f"Available: {', '.join(self._registry.list_courses())}"
            )

        estimated_bytes = entry.disk_size_bytes()
        self._evict_if_needed(estimated_bytes)

        artifacts = self._load_fn(course, entry)
        self._loaded[course] = artifacts
        self._on_first_load(course)
        self._active_course = course
        return artifacts

    def evict(self, course: str) -> bool:
        """Manually evict a loaded course. Returns True if it was loaded."""
        if course not in self._loaded:
            return False

        del self._loaded[course]
        self._last_access.pop(course, None)
        self._tracker.track_unload(course)

        # Update 2Q queues
        if course in self._a1in:
            del self._a1in[course]
            # Add to ghost queue
            self._a1out[course] = True
            while len(self._a1out) > self._a1out_max:
                self._a1out.popitem(last=False)
        elif course in self._am:
            del self._am[course]

        gc.collect()

        if self._active_course == course:
            self._active_course = None

        logger.info("Evicted '%s'", course)
        print(f"Evicted index for '{course}'.")
        return True

    def loaded_courses(self) -> List[str]:
        return list(self._loaded.keys())

    def status(self):
        """Print current session state."""
        loaded = self.loaded_courses()
        if not loaded:
            print("No indices loaded.")
            return

        print(f"Active course: {self._active_course or '(none)'}")
        print(f"Eviction policy: {self._policy}")
        print(f"Loaded indices ({len(loaded)}):")
        for course in loaded:
            mb = self._tracker.get_entry_mb(course)
            last = self._last_access.get(course, 0)
            ago = time.time() - last
            marker = " *" if course == self._active_course else ""
            queue = self._queue_label(course)
            print(f"  {course}: {mb:.1f} MB, last accessed {ago:.0f}s ago, queue={queue}{marker}")

        total = self._tracker.tracked_usage_mb()
        if self._tracker.budget is not None:
            budget_mb = self._tracker.budget / (1024 * 1024)
            print(f"Memory: {total:.1f} / {budget_mb:.0f} MB")
        else:
            print(f"Memory: {total:.1f} MB (no budget set)")

        if self._policy == "2q" and self._a1out:
            print(f"Ghost queue (A1out): {list(self._a1out.keys())}")

    # ---- internal: access tracking ----

    def _on_access(self, course: str):
        """Called when an already-loaded course is accessed again."""
        self._last_access[course] = time.time()

        if self._policy == "2q" and course in self._a1in:
            # Second access while in probation → promote to Am
            del self._a1in[course]
            self._am[course] = True
            self._am.move_to_end(course)
            logger.info("Promoted '%s' from A1in to Am", course)
            print(f"Promoted '{course}' to protected queue.")
        elif self._policy == "2q" and course in self._am:
            # Already in Am — refresh LRU position
            self._am.move_to_end(course)

    def _on_first_load(self, course: str):
        """Called when a course is loaded for the first time (not cached)."""
        self._last_access[course] = time.time()

        if self._policy == "2q":
            if course in self._a1out:
                # Was in ghost queue — it's a proven re-access, go straight to Am
                del self._a1out[course]
                self._am[course] = True
                self._am.move_to_end(course)
                logger.info("Loaded '%s' directly into Am (was in A1out ghost)", course)
            else:
                # Brand new — enter probation
                self._a1in[course] = True

    # ---- internal: eviction ----

    def _pick_victim(self) -> Optional[str]:
        """Choose the next course to evict based on the active policy."""
        if self._policy == "lru":
            return self._pick_victim_lru()
        else:
            return self._pick_victim_2q()

    def _pick_victim_lru(self) -> Optional[str]:
        """Basic LRU: evict the least recently accessed loaded course."""
        candidates = {
            c: t for c, t in self._last_access.items()
            if c in self._loaded and c != self._active_course
        }
        if not candidates:
            candidates = {
                c: t for c, t in self._last_access.items()
                if c in self._loaded
            }
        if not candidates:
            return None
        return min(candidates, key=candidates.get)

    def _pick_victim_2q(self) -> Optional[str]:
        """
        LRU-2Q: prefer evicting from A1in (FIFO, oldest first).
        Fall back to Am (LRU, oldest first).
        Skip the active course if possible.
        """
        # Try A1in first (FIFO — front of OrderedDict is oldest)
        for course in self._a1in:
            if course in self._loaded and course != self._active_course:
                return course

        # Fall back to Am (LRU — front is least recently used)
        for course in self._am:
            if course in self._loaded and course != self._active_course:
                return course

        # Last resort: evict active course
        for course in list(self._a1in) + list(self._am):
            if course in self._loaded:
                return course

        return None

    def _evict_if_needed(self, incoming_bytes: int = 0):
        """Evict courses until there's room for *incoming_bytes*."""
        if self._tracker.budget is None:
            return

        while self._tracker.would_exceed_budget(incoming_bytes):
            victim = self._pick_victim()
            if victim is None:
                logger.warning("Over budget but nothing to evict")
                break
            self.evict(victim)

    def _queue_label(self, course: str) -> str:
        """Return which queue a course is in (for status display)."""
        if self._policy != "2q":
            return "lru"
        if course in self._a1in:
            return "A1in"
        if course in self._am:
            return "Am"
        return "?"
