"""
Supervisor Pattern Orchestrator — Advanced RAG Architecture.

Supervisor điều phối các worker tìm kiếm (Dense, Sparse, PageIndex Fallback)
song song và đưa ra quyết định tổng hợp kết quả (Hybrid RRF) hoặc chuyển giao sang
Vectorless Fallback nếu điểm số truy vấn thấp.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


class DenseWorker:
    """Worker phụ trách Semantic Search (Dense Retrieval)."""

    def execute(self, query: str, top_k: int) -> list[dict]:
        return semantic_search(query, top_k=top_k)


class SparseWorker:
    """Worker phụ trách Lexical Search (BM25 Sparse Retrieval)."""

    def execute(self, query: str, top_k: int) -> list[dict]:
        return lexical_search(query, top_k=top_k)


class FallbackWorker:
    """Worker phụ trách Vectorless Structural Fallback (PageIndex)."""

    def execute(self, query: str, top_k: int) -> list[dict]:
        return pageindex_search(query, top_k=top_k)


class SupervisorEngine:
    """
    Supervisor Pattern Engine điều phối song song các worker
    và tổng hợp kết quả RAG Pipeline.
    """

    def __init__(self, score_threshold: float = 0.3):
        self.dense_worker = DenseWorker()
        self.sparse_worker = SparseWorker()
        self.fallback_worker = FallbackWorker()
        self.score_threshold = score_threshold

    def orchestrate(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """
        Thực thi quy trình điều phối Supervisor:
            1. Chạy DenseWorker & SparseWorker song song
            2. Kiểm tra điểm Cosine tối ưu của DenseWorker
            3. Quyết định Fallback nếu điểm số < threshold
            4. Tổng hợp RRF & Rerank nếu đạt yêu cầu
        """
        start_time = time.time()
        fetch_k = top_k * 2
        decision_path = []

        # Step 1: Execute Workers in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(self.dense_worker.execute, query, fetch_k)
            future_sparse = executor.submit(self.sparse_worker.execute, query, fetch_k)
            dense_results = future_dense.result() or []
            sparse_results = future_sparse.result() or []

        decision_path.append(f"Workers executed: Dense ({len(dense_results)} items), Sparse ({len(sparse_results)} items)")

        best_dense_score = dense_results[0]["score"] if dense_results else 0.0

        # Step 2: Check fallback condition based on raw dense score
        if best_dense_score < self.score_threshold:
            decision_path.append(
                f"Best dense score ({best_dense_score:.3f}) < threshold ({self.score_threshold}) -> Triggering FallbackWorker"
            )
            fallback_results = self.fallback_worker.execute(query, top_k=top_k)
            for item in fallback_results:
                item["source"] = "pageindex"

            elapsed_ms = (time.time() - start_time) * 1000
            return {
                "results": fallback_results[:top_k],
                "metadata": {
                    "source": "pageindex",
                    "best_dense_score": best_dense_score,
                    "fallback_triggered": True,
                    "execution_time_ms": round(elapsed_ms, 2),
                    "decision_path": decision_path,
                },
            }

        # Step 3: Merge results with RRF (Hybrid)
        decision_path.append("Best dense score meets threshold -> Merging via RRF & Reranking")
        merged = rerank_rrf([dense_results, sparse_results], top_k=fetch_k)
        for item in merged:
            item["source"] = "hybrid"

        final_results = rerank(query, merged, top_k=top_k, method="rrf")
        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "results": final_results[:top_k],
            "metadata": {
                "source": "hybrid",
                "best_dense_score": best_dense_score,
                "fallback_triggered": False,
                "execution_time_ms": round(elapsed_ms, 2),
                "decision_path": decision_path,
            },
        }


def run_supervisor(query: str, top_k: int = 5, score_threshold: float = 0.3) -> dict[str, Any]:
    """Helper function cho Supervisor API/Streamlit UI."""
    engine = SupervisorEngine(score_threshold=score_threshold)
    return engine.orchestrate(query, top_k=top_k)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_queries = [
        "Thời gian thử việc tối đa đối với trình độ đại học là bao lâu?",
        "Mức trợ cấp thôi việc và điều kiện hưởng được quy định ra sao?",
        "xyzabc123nonsense",
    ]

    engine = SupervisorEngine(score_threshold=0.3)
    for q in test_queries:
        print(f"\n[Supervisor Query]: {q}")
        print("=" * 60)
        output = engine.orchestrate(q, top_k=3)
        meta = output["metadata"]
        print(f"  Execution Time : {meta['execution_time_ms']} ms")
        print(f"  Source         : {meta['source']}")
        print(f"  Decision Path  : {' -> '.join(meta['decision_path'])}")
        print("  Results:")
        for i, r in enumerate(output["results"], 1):
            print(f"    {i}. [{r['score']:.3f}] [{r.get('source')}] {r['content'][:80]}...")
