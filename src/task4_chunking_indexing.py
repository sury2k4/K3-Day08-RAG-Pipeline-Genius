"""
Task 4 - chunk standardized Markdown documents and build a local vector index.

Chunking strategy: recursive character splitting. Legal documents are long and
often have imperfect OCR/page layout, so paragraph/newline separators are safer
than header-only splitting.

Embedding model: local sklearn HashingVectorizer, 384 dimensions. This keeps
the lab runnable offline; if ChromaDB is installed, the same vectors are also
written to Chroma. Otherwise they are persisted to chroma_db/local_index.json.
"""

import json
import re
from pathlib import Path

import numpy as np

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# 500 chars gives compact legal chunks for retrieval; 80 chars keeps statute
# context across boundaries without creating too much duplication.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "sklearn-hashing-vectorizer"
EMBEDDING_DIM = 384

VECTOR_STORE = "chromadb_with_local_json_fallback"
COLLECTION_NAME = "labor_legal_docs"
LOCAL_INDEX_PATH = CHROMA_DIR / "local_index.json"


def _doc_type(md_file: Path) -> str:
    parts = {part.lower() for part in md_file.relative_to(STANDARDIZED_DIR).parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "unknown"


def load_documents() -> list[dict]:
    """
    Read all Markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        rel_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": rel_path,
                    "type": _doc_type(md_file),
                },
            }
        )
    return documents


def _split_long_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return [chunk for chunk in chunks if chunk]


def _recursive_split(text: str) -> list[str]:
    parts = re.split(r"(\n\n+|\n|(?<=[.!?])\s+)", text)
    units = []
    for i in range(0, len(parts), 2):
        unit = parts[i]
        if i + 1 < len(parts):
            unit += parts[i + 1]
        if unit.strip():
            units.append(unit.strip())

    chunks = []
    current = ""
    for unit in units:
        if len(unit) > CHUNK_SIZE:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_text(unit))
            continue

        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
        overlap = current[-CHUNK_OVERLAP:].strip() if current else ""
        candidate = f"{overlap}\n\n{unit}".strip() if overlap else unit
        current = candidate if len(candidate) <= CHUNK_SIZE else unit

    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents using the configured recursive character strategy.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    chunks = []
    for doc in documents:
        splits = _recursive_split(doc["content"])
        for index, chunk_text in enumerate(splits):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": index,
                        "chunking_method": CHUNKING_METHOD,
                    },
                }
            )
    return chunks


def _hash_embeddings(texts: list[str]) -> np.ndarray:
    try:
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.preprocessing import normalize

        vectorizer = HashingVectorizer(
            n_features=EMBEDDING_DIM,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            ngram_range=(1, 2),
        )
        matrix = vectorizer.transform(texts)
        return normalize(matrix, norm="l2", axis=1).toarray().astype(float)
    except Exception:
        vectors = np.zeros((len(texts), EMBEDDING_DIM), dtype=float)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                vectors[row, hash(token) % EMBEDDING_DIM] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add a 384-dimensional embedding vector to each chunk.

    Returns:
        Each chunk dict with key 'embedding': list[float]
    """
    if not chunks:
        return []

    embeddings = _hash_embeddings([chunk["content"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
        chunk["metadata"]["embedding_model"] = EMBEDDING_MODEL
        chunk["metadata"]["embedding_dim"] = EMBEDDING_DIM
    return chunks


def _chunk_id(chunk: dict) -> str:
    source = re.sub(r"[^A-Za-z0-9_.-]+", "_", chunk["metadata"]["source"])
    return f"{source}_chunk_{chunk['metadata']['chunk_index']}"


def _write_local_index(chunks: list[dict]) -> Path:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks": [
            {
                "id": _chunk_id(chunk),
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "embedding": chunk["embedding"],
            }
            for chunk in chunks
        ],
    }
    LOCAL_INDEX_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return LOCAL_INDEX_PATH


def index_to_vectorstore(chunks: list[dict]) -> Path:
    """Persist chunks and embeddings to ChromaDB when available, plus JSON."""
    local_path = _write_local_index(chunks)

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        collection.upsert(
            ids=[_chunk_id(chunk) for chunk in chunks],
            documents=[chunk["content"] for chunk in chunks],
            embeddings=[chunk["embedding"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks],
        )
    except Exception as exc:
        print(f"ChromaDB unavailable; wrote local JSON index instead: {exc}")

    return local_path


def run_pipeline() -> Path:
    """Run the full pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_path = index_to_vectorstore(chunks)
    print(f"Indexed to: {index_path}")
    return index_path


if __name__ == "__main__":
    run_pipeline()
