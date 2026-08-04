"""
Task 5 - Semantic Search Module.
"""

try:
    from .task4_chunking_indexing import (
        CHROMA_DIR,
        COLLECTION_NAME,
        get_embedding_model,
    )
except ImportError:
    from task4_chunking_indexing import (
        CHROMA_DIR,
        COLLECTION_NAME,
        get_embedding_model,
    )


def _get_collection():
    try:
        import chromadb
    except ModuleNotFoundError:
        return None

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError("Vector index not found. Run Task 4 first.") from exc


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search indexed chunks using vector similarity.

    Args:
        query: Search query.
        top_k: Maximum number of results.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}, sorted by
        score descending.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty.")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    collection = _get_collection()
    if collection is None:
        return []

    count = collection.count()
    if count == 0:
        return []

    model = get_embedding_model()
    query_vector = model.encode(
        query.strip(),
        normalize_embeddings=True,
    ).tolist()

    n_results = min(top_k, count)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1.0 - float(distance)
        score = max(0.0, min(1.0, score))
        output.append({
            "content": document,
            "score": round(score, 4),
            "metadata": metadata or {},
        })

    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    results = semantic_search(
        "Thoi gian thu viec toi da la bao lau?",
        top_k=5,
    )

    for result in results:
        print(f"[{result['score']:.4f}] {result['metadata'].get('source', 'unknown')}")
        print(result["content"][:300])
        print("-" * 80)
