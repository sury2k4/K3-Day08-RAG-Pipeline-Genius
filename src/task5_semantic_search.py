"""
Task 5 - semantic search over the Task 4 vector index.

The primary path queries ChromaDB when it is installed. The repository's local
environment may not have ChromaDB/sentence-transformers, so the function also
supports the deterministic JSON index written by Task 4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from src.task4_chunking_indexing import (
        CHROMA_DIR,
        COLLECTION_NAME,
        LOCAL_INDEX_PATH,
        _hash_embeddings,
    )
except ModuleNotFoundError:
    from task4_chunking_indexing import (
        CHROMA_DIR,
        COLLECTION_NAME,
        LOCAL_INDEX_PATH,
        _hash_embeddings,
    )


def _validate_inputs(query: str, top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _search_local_index(query: str, top_k: int) -> list[dict]:
    if not LOCAL_INDEX_PATH.exists():
        raise RuntimeError("Chroma index is missing or empty. Run Task 4 first.")

    payload = json.loads(LOCAL_INDEX_PATH.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not chunks:
        raise RuntimeError("Chroma index is missing or empty. Run Task 4 first.")

    query_vector = _hash_embeddings([query])[0]
    results = []
    for chunk in chunks:
        embedding = np.asarray(chunk.get("embedding", []), dtype=float)
        if embedding.size == 0:
            continue
        score = float(np.dot(query_vector, embedding))
        score = max(0.0, min(1.0, score))
        metadata = dict(chunk.get("metadata") or {})
        metadata["distance"] = float(1.0 - score)
        results.append(
            {
                "content": chunk.get("content", ""),
                "score": score,
                "metadata": metadata,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _search_chroma(query: str, top_k: int) -> list[dict]:
    import chromadb
    from sentence_transformers import SentenceTransformer
    try:
        from src.task4_chunking_indexing import EMBEDDING_MODEL
    except ModuleNotFoundError:
        from task4_chunking_indexing import EMBEDDING_MODEL

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError("Chroma index is missing or empty. Run Task 4 first.") from exc

    if collection.count() == 0:
        raise RuntimeError("Chroma index is missing or empty. Run Task 4 first.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    query_vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    for document, metadata, distance in zip(documents, metadatas, distances):
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        enriched_metadata = dict(metadata or {})
        enriched_metadata["distance"] = float(distance)
        results.append({"content": document, "score": float(score), "metadata": enriched_metadata})

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks by vector similarity.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        sorted by score descending.
    """
    _validate_inputs(query, top_k)
    try:
        return _search_chroma(query, top_k)
    except ImportError:
        return _search_local_index(query, top_k)
    except RuntimeError:
        if LOCAL_INDEX_PATH.exists():
            return _search_local_index(query, top_k)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 5: semantic search")
    parser.add_argument("query", nargs="?", default="Thoi gian thu viec toi da la bao lau?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    for result in semantic_search(args.query, top_k=args.top_k):
        line = f"[{result['score']:.4f}] {result['metadata'].get('source')} :: {result['content'][:160]}"
        print(line.encode("ascii", errors="replace").decode("ascii"))
