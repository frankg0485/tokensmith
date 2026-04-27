"""
test_session_equivalence.py

Full-pipeline tests comparing baseline (direct load_artifacts) vs session
manager path. Proves the session manager is a transparent wrapper with
no impact on retrieval quality.

Run with:  pytest tests/test_session_equivalence.py -v
(Requires index files on disk and embedding model)
"""

import json
import time
from pathlib import Path

import pytest

from src.config import RAGConfig
from src.index_registry import IndexRegistry
from src.memory_tracker import MemoryTracker
from src.session_manager import SessionManager
from src.retriever import (
    load_artifacts,
    FAISSRetriever,
    BM25Retriever,
    filter_retrieved_chunks,
)
from src.ranking.ranker import EnsembleRanker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        pytest.skip("config.yaml not found")
    return RAGConfig.from_yaml(config_path)


@pytest.fixture(scope="module")
def registry():
    reg = IndexRegistry()
    for entry in reg.discover():
        reg.register(
            course=entry.course,
            artifacts_dir=entry.artifacts_dir,
            index_prefix=entry.index_prefix,
        )
    if not reg.list_courses():
        pytest.skip("No indices available")
    return reg


@pytest.fixture(scope="module")
def course_entry(registry):
    """Use the first available course for equivalence tests."""
    course = registry.list_courses()[0]
    return registry.get(course)


@pytest.fixture(scope="module")
def test_queries():
    """Queries to use for equivalence testing."""
    return [
        "What are the ACID properties of transactions?",
        "How does a B+ tree index work?",
        "Explain primary keys and foreign keys",
    ]


def _run_retrieval(cfg, faiss_idx, bm25_idx, chunks, query):
    """Run retrieval for a single query and return top-k chunk IDs."""
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

    raw_scores = {}
    for retriever in retrievers:
        raw_scores[retriever.name] = retriever.get_scores(query, pool_n, chunks)
    ordered, scores = ranker.rank(raw_scores=raw_scores)
    topk = list(filter_retrieved_chunks(cfg, chunks, ordered))
    return topk


# ---------------------------------------------------------------------------
# Chunk retrieval equivalence
# ---------------------------------------------------------------------------

@pytest.mark.session
class TestChunkRetrievalEquivalence:

    def test_chunk_retrieval_equivalence(self, cfg, course_entry, test_queries):
        """Same query returns identical chunk IDs via baseline and session manager."""
        artifacts_dir = str(course_entry.artifacts_path)
        index_prefix = course_entry.index_prefix

        # Baseline: load artifacts directly
        faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(
            artifacts_dir, index_prefix
        )

        # Session manager: load via get_or_load
        tracker = MemoryTracker()
        registry = IndexRegistry()

        def real_load(course_name, entry):
            fi, bi, ch, so, me = load_artifacts(
                str(entry.artifacts_path), entry.index_prefix
            )
            tracker.track(course_name, entry.disk_size_bytes())
            return {
                "chunks": ch, "sources": so, "meta": me,
                "faiss_idx": fi, "bm25_idx": bi,
            }

        sm = SessionManager(registry=registry, tracker=tracker, load_fn=real_load)
        sm_artifacts = sm.get_or_load(course_entry.course)
        sm_faiss = sm_artifacts["faiss_idx"]
        sm_bm25 = sm_artifacts["bm25_idx"]
        sm_chunks = sm_artifacts["chunks"]

        print(f"\nComparing baseline vs session manager for '{course_entry.course}'")
        for query in test_queries:
            baseline_ids = _run_retrieval(cfg, faiss_idx, bm25_idx, chunks, query)
            session_ids = _run_retrieval(cfg, sm_faiss, sm_bm25, sm_chunks, query)

            assert baseline_ids == session_ids, (
                f"Chunk IDs differ for query: '{query}'\n"
                f"  baseline: {baseline_ids}\n"
                f"  session:  {session_ids}"
            )
            print(f"  '{query[:50]}...' → {len(baseline_ids)} chunks, IDENTICAL")


# ---------------------------------------------------------------------------
# Latency comparison
# ---------------------------------------------------------------------------

@pytest.mark.session
class TestLatencyComparison:

    def test_latency_comparison(self, cfg, registry, course_entry, test_queries):
        """Compare startup, first-query, and cached-access latency."""
        artifacts_dir = str(course_entry.artifacts_path)
        index_prefix = course_entry.index_prefix
        course = course_entry.course

        # 1. Baseline load latency
        t0 = time.perf_counter()
        faiss_idx, bm25_idx, chunks, sources, meta = load_artifacts(
            artifacts_dir, index_prefix
        )
        t_baseline_load = time.perf_counter() - t0

        # 2. Session manager: first load (lazy, from disk)
        tracker = MemoryTracker()

        def real_load(course_name, entry):
            fi, bi, ch, so, me = load_artifacts(
                str(entry.artifacts_path), entry.index_prefix
            )
            tracker.track(course_name, entry.disk_size_bytes())
            return {
                "chunks": ch, "sources": so, "meta": me,
                "faiss_idx": fi, "bm25_idx": bi,
            }

        sm = SessionManager(registry=registry, tracker=tracker, load_fn=real_load)

        t0 = time.perf_counter()
        sm.get_or_load(course)
        t_session_first = time.perf_counter() - t0

        # 3. Session manager: cached access
        t0 = time.perf_counter()
        sm.get_or_load(course)
        t_session_cached = time.perf_counter() - t0

        # 4. Per-query latency (retrieval only, no LLM)
        baseline_query_times = []
        session_query_times = []

        sm_artifacts = sm.get_or_load(course)
        sm_faiss = sm_artifacts["faiss_idx"]
        sm_bm25 = sm_artifacts["bm25_idx"]
        sm_chunks = sm_artifacts["chunks"]

        for query in test_queries:
            t0 = time.perf_counter()
            _run_retrieval(cfg, faiss_idx, bm25_idx, chunks, query)
            baseline_query_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            _run_retrieval(cfg, sm_faiss, sm_bm25, sm_chunks, query)
            session_query_times.append(time.perf_counter() - t0)

        baseline_query_median = sorted(baseline_query_times)[len(baseline_query_times) // 2]
        session_query_median = sorted(session_query_times)[len(session_query_times) // 2]

        # Print results
        print(f"\n{'Metric':<30} | {'Baseline':>10} | {'Session Mgr':>12}")
        print("-" * 58)
        print(f"{'Index load (disk→RAM)':<30} | {t_baseline_load:>9.3f}s | {t_session_first:>11.3f}s")
        print(f"{'Cached access':<30} | {'N/A':>10} | {t_session_cached:>11.6f}s")
        print(f"{'Query latency (median)':<30} | {baseline_query_median:>9.3f}s | {session_query_median:>11.3f}s")

        # Assertions
        assert t_session_cached < 0.001, (
            f"Cached access should be < 1ms, got {t_session_cached:.4f}s"
        )

        # Save results
        results = {
            "course": course,
            "baseline_load_s": t_baseline_load,
            "session_first_load_s": t_session_first,
            "session_cached_access_s": t_session_cached,
            "baseline_query_median_s": baseline_query_median,
            "session_query_median_s": session_query_median,
            "num_queries": len(test_queries),
        }

        results_dir = Path("tests/results")
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / "session_latency.json", "w") as f:
            json.dump(results, f, indent=2)
