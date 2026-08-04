"""
Task 4 - chunk standardized Markdown documents and build a local vector index.

Chunking strategy: recursive character splitting. Labor-law documents contain
articles, clauses, and headings, so separators prioritize legal boundaries
before falling back to paragraphs/sentences/words.

Embedding model: local sklearn HashingVectorizer, 384 dimensions. This keeps
the lab runnable offline; if ChromaDB is installed, the same vectors are also
written to Chroma. Otherwise they are persisted to chroma_db/local_index.json.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# 1000 chars usually fits one legal provision with enough context. 150 chars of
# overlap keeps adjacent clauses connected without too much duplication.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "recursive"
SEPARATORS = ["\n# ", "\n## ", "\n### ", "\nĐiều ", "\nKhoản ", "\n\n", "\n", ". ", " ", ""]

EMBEDDING_MODEL = "sklearn-hashing-vectorizer"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 384
EMBEDDING_DIMENSION = 1024

VECTOR_STORE = "chromadb_with_local_json_fallback"
COLLECTION_NAME = "labor_law_documents"
LOCAL_INDEX_PATH = CHROMA_DIR / "local_index.json"


def _doc_type(md_file: Path) -> str:
    parts = {part.lower() for part in md_file.relative_to(STANDARDIZED_DIR).parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "unknown"


def _parse_front_matter(content: str) -> tuple[dict, str]:
    text = content.lstrip()
    if not text.startswith("---"):
        return {}, content
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, content
    metadata = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, parts[2].strip()


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
        raw_content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        front_matter, content = _parse_front_matter(raw_content)
        if not content.strip():
            continue
        rel_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        doc_type = front_matter.get("corpus_type") or _doc_type(md_file)
        documents.append(
            {
                "content": content.strip(),
                "metadata": {
                    "source": front_matter.get("source_file") or md_file.name,
                    "source_file": front_matter.get("source_file") or md_file.name,
                    "path": rel_path,
                    "source_path": rel_path,
                    "type": doc_type,
                    "corpus_type": doc_type,
                    "title": front_matter.get("title") or "",
                    "source_url": front_matter.get("source_url") or "",
                    "document_number": front_matter.get("document_number") or "",
                    "topic": front_matter.get("topic") or "",
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
        total_chunks = len(splits)
        for index, chunk_text in enumerate(splits):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": index,
                        "total_chunks": total_chunks,
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
    key = f"{chunk['metadata'].get('source_path')}:{chunk['metadata'].get('chunk_index')}:{chunk['content']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


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
                "metadata": {
                    key: value
                    for key, value in chunk["metadata"].items()
                    if isinstance(value, (str, int, float, bool))
                },
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


def load_markdown_documents(input_dir: str | Path = STANDARDIZED_DIR) -> list[dict]:
    global STANDARDIZED_DIR
    previous_dir = STANDARDIZED_DIR
    STANDARDIZED_DIR = Path(input_dir)
    try:
        return load_documents()
    finally:
        STANDARDIZED_DIR = previous_dir


def build_index(
    input_dir: str | Path = STANDARDIZED_DIR,
    persist_dir: str | Path = CHROMA_DIR,
    rebuild: bool = False,
) -> dict:
    global STANDARDIZED_DIR, CHROMA_DIR, LOCAL_INDEX_PATH
    previous_standardized, previous_chroma, previous_index = STANDARDIZED_DIR, CHROMA_DIR, LOCAL_INDEX_PATH
    STANDARDIZED_DIR, CHROMA_DIR = Path(input_dir), Path(persist_dir)
    LOCAL_INDEX_PATH = CHROMA_DIR / "local_index.json"
    try:
        if rebuild and LOCAL_INDEX_PATH.exists():
            LOCAL_INDEX_PATH.unlink()
        docs = load_documents()
        chunks = embed_chunks(chunk_documents(docs))
        index_to_vectorstore(chunks)
        return {
            "documents": len(docs),
            "chunks": len(chunks),
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": EMBEDDING_DIM,
        }
    finally:
        STANDARDIZED_DIR, CHROMA_DIR, LOCAL_INDEX_PATH = previous_standardized, previous_chroma, previous_index


def run_pipeline(rebuild: bool = False) -> Path:
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

    if rebuild and LOCAL_INDEX_PATH.exists():
        LOCAL_INDEX_PATH.unlink()
    index_path = index_to_vectorstore(chunks)
    print(f"Indexed to: {index_path}")
    return index_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 4: chunk and index Markdown files")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    run_pipeline(rebuild=args.rebuild)
