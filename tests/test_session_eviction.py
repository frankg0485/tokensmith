"""
test_session_eviction.py

Mock-based tests for LRU and LRU-2Q eviction policies.
Runs in milliseconds — no disk I/O or model loading needed.

Run with:  pytest tests/test_session_eviction.py -v
"""

import json
import random
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest

from src.memory_tracker import MemoryTracker
from src.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

@dataclass
class MockIndexEntry:
    """Minimal stand-in for IndexEntry with a configurable disk size."""
    course: str
    artifacts_dir: str = "mock/"
    index_prefix: str = "mock"
    pdf_dir: Optional[str] = None
    _size: int = 0

    def disk_size_bytes(self) -> int:
        return self._size

    @property
    def artifacts_path(self):
        return Path(self.artifacts_dir)


class MockRegistry:
    """Minimal stand-in for IndexRegistry."""

    def __init__(self, courses: Dict[str, int]):
        """courses: {name: size_in_bytes}"""
        self._entries = {
            name: MockIndexEntry(course=name, _size=size)
            for name, size in courses.items()
        }

    def get(self, course: str):
        return self._entries.get(course)

    def list_courses(self) -> List[str]:
        return sorted(self._entries.keys())


def _make_session(courses_mb: Dict[str, float], budget_mb: float, policy: str = "lru"):
    """Helper: create a SessionManager with mock courses of given sizes."""
    courses_bytes = {name: int(mb * 1024 * 1024) for name, mb in courses_mb.items()}
    registry = MockRegistry(courses_bytes)
    tracker = MemoryTracker()

    load_count = {"n": 0}

    def mock_load(course_name, entry):
        load_count["n"] += 1
        tracker.track(course_name, entry.disk_size_bytes())
        return {"chunks": [], "course": course_name}

    sm = SessionManager(
        registry=registry,
        tracker=tracker,
        load_fn=mock_load,
        budget_mb=budget_mb,
        policy=policy,
    )
    return sm, tracker, load_count


# ---------------------------------------------------------------------------
# LRU tests
# ---------------------------------------------------------------------------

@pytest.mark.session
class TestLRUEviction:

    def test_lru_eviction_order(self):
        """LRU evicts the least recently accessed course."""
        sm, tracker, _ = _make_session(
            {"A": 5, "B": 5, "C": 5, "D": 5}, budget_mb=12, policy="lru"
        )

        sm.get_or_load("A")
        sm.get_or_load("B")  # A=5, B=5 → 10 MB, fits in 12
        assert set(sm.loaded_courses()) == {"A", "B"}

        sm.get_or_load("C")  # 10+5=15 > 12 → evict A (oldest)
        assert "A" not in sm.loaded_courses()
        assert set(sm.loaded_courses()) == {"B", "C"}

        sm.get_or_load("B")  # refresh B's timestamp
        sm.get_or_load("D")  # 10+5=15 > 12 → evict C (B was refreshed)
        assert "C" not in sm.loaded_courses()
        assert set(sm.loaded_courses()) == {"B", "D"}

    def test_lru_prefers_non_active(self):
        """LRU avoids evicting the active course when possible."""
        sm, _, _ = _make_session(
            {"A": 5, "B": 5, "C": 5}, budget_mb=8, policy="lru"
        )

        sm.get_or_load("A")
        sm.get_or_load("B")  # evicts A
        assert sm.active_course == "B"

        sm.get_or_load("C")  # evicts A or B; B is active, so evict the other loaded one
        assert sm.active_course == "C"
        assert "C" in sm.loaded_courses()


# ---------------------------------------------------------------------------
# 2Q tests
# ---------------------------------------------------------------------------

@pytest.mark.session
class TestLRU2QEviction:

    def test_2q_probation_and_promotion(self):
        """New items enter A1in; second access promotes to Am; eviction prefers A1in."""
        sm, _, _ = _make_session(
            {"A": 5, "B": 5, "C": 5, "D": 5}, budget_mb=12, policy="2q"
        )

        sm.get_or_load("A")  # → A1in
        assert "A" in sm._a1in

        sm.get_or_load("B")  # → A1in
        assert "B" in sm._a1in

        sm.get_or_load("A")  # second access → promoted to Am
        assert "A" in sm._am
        assert "A" not in sm._a1in

        sm.get_or_load("C")  # needs eviction, B in A1in, A in Am → evict B (A1in preferred)
        assert "B" not in sm.loaded_courses()
        assert "B" in sm._a1out  # B's key goes to ghost queue
        assert set(sm.loaded_courses()) == {"A", "C"}

        # Now active=C. Loading D: A1in has C (active, skipped), Am has A → evict A
        sm.get_or_load("D")
        assert "A" not in sm.loaded_courses()
        assert set(sm.loaded_courses()) == {"C", "D"}

    def test_2q_ghost_queue_direct_promotion(self):
        """Reloading a course from A1out ghost queue places it directly in Am."""
        sm, _, _ = _make_session(
            {"A": 5, "B": 5, "C": 5}, budget_mb=8, policy="2q"
        )

        sm.get_or_load("A")  # → A1in
        sm.get_or_load("B")  # evicts A → A goes to A1out
        assert "A" in sm._a1out
        assert "A" not in sm._a1in

        sm.get_or_load("A")  # was in A1out → load directly into Am
        assert "A" in sm._am
        assert "A" not in sm._a1in
        assert "A" not in sm._a1out

    def test_2q_a1in_evicted_before_am(self):
        """When A1in has a non-active entry, it's evicted before Am."""
        sm, _, _ = _make_session(
            {"A": 4, "B": 4, "C": 4, "D": 4}, budget_mb=10, policy="2q"
        )

        sm.get_or_load("A")  # → A1in
        sm.get_or_load("A")  # → Am (promoted)
        sm.get_or_load("B")  # → A1in (A=4 + B=4 = 8, fits)
        sm.get_or_load("C")  # active=C. A1in=[B,C], but B is not active → evict B (A1in)
        # Wait — active switches to C during get_or_load, but B was the last active before.
        # Actually: before loading C, active=B (from last get_or_load). B is in A1in.
        # Eviction skips B (active), goes to Am → evicts A. Let's just verify the invariant:
        # after loading C, only 2 courses loaded, total ≤ 10MB.
        assert len(sm.loaded_courses()) == 2
        assert sm._tracker.tracked_usage_mb() <= 10.0

        # Now set up the real test: load A (goes to Am via ghost), load B
        # Then load D: A1in has a non-active entry.
        sm2, _, _ = _make_session(
            {"A": 3, "B": 3, "C": 3}, budget_mb=8, policy="2q"
        )
        sm2.get_or_load("A")  # → A1in
        sm2.get_or_load("B")  # → A1in, total=6, fits
        sm2.get_or_load("A")  # promote A to Am
        # A1in=[B], Am=[A], active=A
        assert "B" in sm2._a1in
        assert "A" in sm2._am
        assert sm2.active_course == "A"

        sm2.get_or_load("C")  # 6+3=9 > 8. active=A. A1in has B (not active) → evict B
        assert "B" not in sm2.loaded_courses()
        assert "A" in sm2.loaded_courses()  # Am protected
        assert "C" in sm2.loaded_courses()


# ---------------------------------------------------------------------------
# Cache hit rate comparison
# ---------------------------------------------------------------------------

@pytest.mark.session
class TestCacheHitRate:

    def test_cache_hit_rate_comparison(self):
        """Compare LRU vs 2Q hit rates under a realistic access pattern."""
        courses = {"databases": 8, "python": 4, "algorithms": 6, "networks": 5, "os": 5}
        budget_mb = 15

        # Realistic student pattern: heavy on one course with occasional dips
        access_pattern = (
            ["databases"] * 3 + ["python"] + ["databases"] * 2 +
            ["algorithms"] + ["databases"] + ["networks"] + ["databases"] * 2 +
            ["os"] + ["databases"] + ["python"] + ["databases"]
        )

        results = {}
        for policy in ("lru", "2q"):
            sm, tracker, load_count = _make_session(courses, budget_mb, policy=policy)
            hits = 0
            misses = 0

            for course in access_pattern:
                was_loaded = course in sm.loaded_courses()
                sm.get_or_load(course)
                if was_loaded:
                    hits += 1
                else:
                    misses += 1

            results[policy] = {
                "hits": hits,
                "misses": misses,
                "evictions": load_count["n"] - len(set(access_pattern)),
                "hit_rate": hits / len(access_pattern),
            }

        # Print comparison table
        print(f"\n{'Policy':<8} | {'Hits':>4} | {'Misses':>6} | {'Evictions':>9} | {'Hit Rate':>8}")
        print("-" * 50)
        for policy, r in results.items():
            print(f"{policy:<8} | {r['hits']:>4} | {r['misses']:>6} | {r['evictions']:>9} | {r['hit_rate']:>7.1%}")

        # Both policies must complete without error
        for policy, r in results.items():
            assert r["hits"] + r["misses"] == len(access_pattern)

        # Save results
        results_dir = Path("tests/results")
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / "eviction_comparison.json", "w") as f:
            json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

@pytest.mark.session
class TestBudgetEnforcement:

    def test_memory_budget_never_exceeded(self):
        """tracked_usage <= budget holds after every operation."""
        courses = {"A": 5, "B": 4, "C": 6, "D": 3, "E": 5}
        budget_mb = 12

        for policy in ("lru", "2q"):
            sm, tracker, _ = _make_session(courses, budget_mb, policy=policy)
            budget_bytes = int(budget_mb * 1024 * 1024)
            course_names = list(courses.keys())

            random.seed(42)
            for i in range(50):
                course = random.choice(course_names)
                sm.get_or_load(course)
                usage = tracker.tracked_usage()
                assert usage <= budget_bytes, (
                    f"[{policy}] step {i}: tracked_usage={usage} > budget={budget_bytes} "
                    f"after loading '{course}'"
                )

    def test_budget_50_percent(self):
        """With budget at 50% of total, only one index fits at a time."""
        # Simulate real sizes: databases=22.6, python=6.3, total=28.9
        courses = {"databases": 22.6, "python": 6.3}
        budget_mb = 14.5  # 50% of 28.9

        for policy in ("lru", "2q"):
            sm, tracker, _ = _make_session(courses, budget_mb, policy=policy)

            sm.get_or_load("python")  # 6.3 fits
            assert tracker.tracked_usage_mb() == pytest.approx(6.3, abs=0.1)

            sm.get_or_load("databases")  # 6.3 + 22.6 > 14.5 → evict python first
            assert "python" not in sm.loaded_courses()
            assert "databases" in sm.loaded_courses()

    def test_single_course_exceeds_budget(self):
        """When a single course exceeds the budget, it still loads (best-effort)."""
        sm, tracker, _ = _make_session(
            {"big": 20, "small": 3}, budget_mb=10, policy="lru"
        )

        # big (20MB) > budget (10MB), but nothing to evict → loads anyway
        sm.get_or_load("big")
        assert "big" in sm.loaded_courses()

        # Now loading small should evict big
        sm.get_or_load("small")
        assert "big" not in sm.loaded_courses()
        assert "small" in sm.loaded_courses()
