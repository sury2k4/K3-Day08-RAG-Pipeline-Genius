"""
Task 4 - Chunking & Indexing into ChromaDB.
"""

from functools import lru_cache
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

INVALID_TEXT_MARKERS = (
    "this source file appears to be a scanned",
    "no selectable text layer was found",
    "full text ocr is required",
    "image-based legal document",
)

# RecursiveCharacterTextSplitter is selected because the corpus mixes legal
# Markdown and news Markdown. Converted PDFs may not preserve headings
# consistently, so recursive splitting is more stable than header-only splitting.
# chunk_size=800 keeps enough context for an article/clause while avoiding overly
# broad retrieval results. chunk_overlap=100 preserves context at chunk edges
# without producing too much duplicated text.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

# BAAI/bge-m3 is multilingual, suitable for Vietnamese legal text, runs locally,
# needs no API key, and produces 1024-dimensional embeddings.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# ChromaDB is local, persistent, simple for this lab, and matches the README.
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "labor_law_docs"


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def _has_invalid_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in INVALID_TEXT_MARKERS)


def _parse_front_matter(content: str) -> tuple[dict, str]:
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    metadata = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"').strip("'")
        metadata[key.strip()] = value
    return metadata, parts[2].strip()


def _primitive_metadata(metadata: dict) -> dict:
    clean = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def load_documents() -> list[dict]:
    """
    Read valid Markdown files from data/standardized/.

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        raise RuntimeError(f"Standardized directory not found: {STANDARDIZED_DIR}")

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        raw_content = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if len(raw_content) < 200 or _has_invalid_marker(raw_content):
            continue

        parsed_metadata, body_text = _parse_front_matter(raw_content)
        body_text = body_text.strip()
        if len(body_text) < 200 or _has_invalid_marker(body_text):
            continue

        doc_type = md_file.parent.name
        metadata = {
            "source": md_file.name,
            "source_file": md_file.name,
            "source_path": str(md_file.relative_to(STANDARDIZED_DIR)),
            "type": doc_type,
            "corpus_type": doc_type,
            "title": parsed_metadata.get("title", md_file.stem),
            "source_url": parsed_metadata.get("source_url", ""),
        }
        documents.append({
            "content": body_text,
            "metadata": _primitive_metadata(metadata),
        })

    if not documents:
        raise RuntimeError("No valid Markdown documents found for indexing.")
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents using RecursiveCharacterTextSplitter.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n# ",
                "\n## ",
                "\n### ",
                "\nDieu ",
                "\nKhoan ",
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )
        split_text = splitter.split_text
    except ModuleNotFoundError:
        def split_text(text: str) -> list[str]:
            step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
            return [
                text[start : start + CHUNK_SIZE]
                for start in range(0, len(text), step)
            ]

    chunks = []
    for doc in documents:
        splits = [text.strip() for text in split_text(doc["content"]) if text.strip()]
        for index, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": index,
                    "total_chunks": len(splits),
                },
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an embedding vector to each chunk.
    """
    if not chunks:
        raise ValueError("chunks must not be empty.")

    model = get_embedding_model()
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=16,
    )
    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count.")

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Rebuild the ChromaDB collection and index all chunks.
    """
    if not chunks:
        raise ValueError("chunks must not be empty.")

    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [
        f"{chunk['metadata']['source_file']}_chunk_{chunk['metadata']['chunk_index']}"
        for chunk in chunks
    ]
    collection.add(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[_primitive_metadata(chunk["metadata"]) for chunk in chunks],
    )
    print(f"Indexed {len(chunks)} chunks into collection {COLLECTION_NAME}")
    return collection


def run_pipeline():
    """Run the full pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print(f"  Collection: {COLLECTION_NAME}")
    print("=" * 50)

    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Indexed to vector store")
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIM,
        "collection_name": COLLECTION_NAME,
    }


if __name__ == "__main__":
    run_pipeline()
