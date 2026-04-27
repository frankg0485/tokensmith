"""
test_session_switching.py

Tests for course switching: latency, memory accounting, forced eviction,
and answer quality after evict/reload cycles.

Uses real IndexEntry.disk_size_bytes() but mocks the artifact loader for
speed in most tests. test_answer_quality_after_evict_reload uses the real
load_artifacts path.

Run with:  pytest tests/test_session_switching.py -v
"""

import json
import time
from pathlib import Path

import pytest

from src.index_registry import IndexRegistry
from src.memory_tracker import MemoryTracker
from src.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_real_session(budget_mb=None, policy="lru"):
    """Create a SessionManager backed by real registry entries but with a
    lightweight mock loader (returns empty artifact dicts with tracked sizes)."""
    registry = IndexRegistry()
    tracker = MemoryTracker()

    if not registry.list_courses():
        for entry in registry.discover():
            registry.register(
                course=entry.course,
                artifacts_dir=entry.artifacts_dir,
                index_prefix=entry.index_prefix,
            )

    courses = registry.list_courses()
    if len(courses) < 2:
        pytest.skip("Need at least 2 registered courses to test switching")

    def mock_load(course_name, entry):
        tracker.track(course_name, entry.disk_size_bytes())
        return {"chunks": [], "sources": [], "retrievers": [], "ranker": None, "meta": []}

    sm = SessionManager(
        registry=registry, tracker=tracker, load_fn=mock_load,
        budget_mb=budget_mb, policy=policy,
    )
    return sm, tracker, registry


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

@pytest.mark.session
class TestSwitchLatency:

    def test_switch_latency(self):
        """Measure cold-load vs warm cache-hit latency for each course."""
        sm, tracker, registry = _make_real_session()
        courses = registry.list_courses()
        results = {}

        for course in courses:
            # Cold load
            t0 = time.perf_counter()
            sm.get_or_load(course)
            cold = time.perf_counter() - t0

            # Warm hit
            t0 = time.perf_counter()
            sm.get_or_load(course)
            warm = time.perf_counter() - t0

            results[course] = {"cold_s": cold, "warm_s": warm}

        # Print table
        print(f"\n{'Course':<20} | {'Cold Load':>10} | {'Warm Hit':>10}")
        print("-" * 45)
        for course, r in results.items():
            print(f"{course:<20} | {r['cold_s']:>9.4f}s | {r['warm_s']:>9.6f}s")

        # Warm hits should be near-instant (dict lookup)
        for course, r in results.items():
            assert r["warm_s"] < 0.01, f"Warm hit for '{course}' took {r['warm_s']:.4f}s"

        # Save results
        results_dir = Path("tests/results")
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / "session_switching.json", "w") as f:
            json.dump(results, f, indent=2)


# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------

@pytest.mark.session
class TestSwitchMemory:

    def test_switch_memory_accounting(self):
        """tracked_usage_mb correctly reflects loads and evictions."""
        sm, tracker, registry = _make_real_session()
        courses = registry.list_courses()
        c1, c2 = courses[0], courses[1]

        entry1 = registry.get(c1)
        entry2 = registry.get(c2)
        size1_mb = entry1.disk_size_bytes() / (1024 * 1024)
        size2_mb = entry2.disk_size_bytes() / (1024 * 1024)

        # Load first course
        sm.get_or_load(c1)
        assert tracker.tracked_usage_mb() == pytest.approx(size1_mb, abs=0.1)

        # Load second course (no budget, both fit)
        sm.get_or_load(c2)
        assert tracker.tracked_usage_mb() == pytest.approx(size1_mb + size2_mb, abs=0.1)

        # Evict first course
        sm.evict(c1)
        assert c1 not in tracker
        assert tracker.tracked_usage_mb() == pytest.approx(size2_mb, abs=0.1)

        # Reload first course
        sm.get_or_load(c1)
        assert tracker.tracked_usage_mb() == pytest.approx(size1_mb + size2_mb, abs=0.1)

    def test_switch_forces_eviction(self):
        """Tight budget forces eviction when switching courses."""
        registry = IndexRegistry()
        if len(registry.list_courses()) < 2:
            pytest.skip("Need at least 2 courses")

        courses = registry.list_courses()
        c1, c2 = courses[0], courses[1]
        e1 = registry.get(c1)
        e2 = registry.get(c2)
        size1 = e1.disk_size_bytes() / (1024 * 1024)
        size2 = e2.disk_size_bytes() / (1024 * 1024)

        # Budget: fits the larger one but not both
        budget = max(size1, size2) + 1.0

        sm, tracker, _ = _make_real_session(budget_mb=budget)

        sm.get_or_load(c1)
        sm.get_or_load(c2)  # should evict c1 if both don't fit

        if size1 + size2 > budget:
            assert c1 not in sm.loaded_courses()
            assert c2 in sm.loaded_courses()
            assert tracker.tracked_usage_mb() <= budget
        else:
            # Both fit — no eviction needed
            assert c1 in sm.loaded_courses()
            assert c2 in sm.loaded_courses()


# ---------------------------------------------------------------------------
# Answer quality after evict/reload
# ---------------------------------------------------------------------------

@pytest.mark.session
class TestAnswerQualityAfterReload:

    def test_answer_quality_after_evict_reload(self):
        """Chunk IDs are identical before and after an evict/reload cycle."""
        try:
            from src.retriever import (
                load_artifacts, FAISSRetriever, BM25Retriever,
                filter_retrieved_chunks,
            )
            from src.ranking.ranker import EnsembleRanker
            from src.config import RAGConfig
        except ImportError:
            pytest.skip("Required modules not available")

        registry = IndexRegistry()
        courses = registry.list_courses()
        if not courses:
            pytest.skip("No registered courses")

        config_path = Path("config/config.yaml")
        if not config_path.exists():
            pytest.skip("config.yaml not found")

        cfg = RAGConfig.from_yaml(config_path)
        course = courses[0]
        entry = registry.get(course)

        test_queries = [
            "What are the ACID properties?",
            "How does a B+ tree work?",
        ]

        def run_retrieval(artifacts_dir, index_prefix, queries):
            """Run retrieval for given queries and return chunk ID lists."""
            faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(
                artifacts_dir, index_prefix
            )
            retrievers = [
                FAISSRetriever(faiss_idx, cfg.embed_model),
                BM25Retriever(bm25_idx),
            ]
            ranker = EnsembleRanker(
                ensemble_method=cfg.ensemble_method,
                weights=cfg.ranker_weights,
                rrf_k=int(cfg.rrf_k),
            )
            pool_n = max(cfg.num_candidates, cfg.top_k + 10)

            results = {}
            for q in queries:
                raw_scores = {}
                for retriever in retrievers:
                    raw_scores[retriever.name] = retriever.get_scores(q, pool_n, chunks)
                ordered, _ = ranker.rank(raw_scores=raw_scores)
                topk = list(filter_retrieved_chunks(cfg, chunks, ordered))
                results[q] = topk
            return results

        # First retrieval
        pre_results = run_retrieval(str(entry.artifacts_path), entry.index_prefix, test_queries)

        # Simulate evict/reload by running retrieval again with fresh load
        post_results = run_retrieval(str(entry.artifacts_path), entry.index_prefix, test_queries)

        # Compare chunk IDs
        for q in test_queries:
            assert pre_results[q] == post_results[q], (
                f"Chunk IDs differ for query '{q}':\n"
                f"  pre:  {pre_results[q]}\n"
                f"  post: {post_results[q]}"
            )
            print(f"  '{q[:40]}...' → {len(pre_results[q])} chunks, identical before/after reload")
