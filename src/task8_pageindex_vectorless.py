"""Task 8: PageIndex retrieval with a deterministic local structural fallback.

The hosted PageIndex API accepts PDF uploads.  The local fallback keeps the
same result contract and searches Markdown sections without embeddings.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = ROOT / "data" / "standardized"
LEGAL_DIR = ROOT / "data" / "landing" / "legal"
PAGEINDEX_DIR = ROOT / "chroma_db"
MANIFEST_PATH = PAGEINDEX_DIR / "pageindex_documents.json"
PAGEINDEX_BASE_URL = "https://api.vectify.ai/pageindex"


def _read_manifest() -> dict:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_manifest(manifest: dict) -> None:
    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_documents() -> list[dict]:
    """Upload raw PDFs to hosted PageIndex and persist doc IDs locally."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is not configured")
    import requests

    manifest = _read_manifest()
    uploaded = []
    for pdf_path in sorted(LEGAL_DIR.glob("*.pdf")):
        key = str(pdf_path.resolve())
        if manifest.get(key, {}).get("doc_id"):
            uploaded.append(manifest[key])
            continue
        with pdf_path.open("rb") as stream:
            response = requests.post(
                PAGEINDEX_BASE_URL + "/",
                headers={"api_key": PAGEINDEX_API_KEY},
                files={"file": (pdf_path.name, stream, "application/pdf")},
                timeout=120,
            )
        response.raise_for_status()
        payload = response.json()
        doc_id = payload.get("doc_id") or payload.get("id")
        if not doc_id:
            raise RuntimeError(f"No doc_id returned for {pdf_path.name}")
        manifest[key] = {"doc_id": doc_id, "filename": pdf_path.name, "uploaded_at": time.time()}
        uploaded.append(manifest[key])
    _write_manifest(manifest)
    return uploaded


def _local_structural_search(query: str, top_k: int) -> list[dict]:
    terms = set(re.findall(r"[\wÀ-ỹ]+", query.casefold()))
    results = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        sections = re.split(r"(?=^#{1,6}\s|^Điều\s)", text, flags=re.MULTILINE)
        for section in sections:
            content = section.strip()
            if not content:
                continue
            section_terms = set(re.findall(r"[\wÀ-ỹ]+", content.casefold()))
            overlap = len(terms & section_terms)
            if overlap:
                title = content.splitlines()[0].lstrip("# ").strip()[:200]
                results.append({
                    "content": content,
                    "score": float(overlap / max(len(terms), 1)),
                    "metadata": {"source": path.name, "path": str(path), "section": title},
                    "source": "pageindex",
                })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def _remote_search(query: str, top_k: int) -> list[dict]:
    import requests

    results = []
    for record in _read_manifest().values():
        doc_id = record.get("doc_id")
        if not doc_id:
            continue
        response = requests.get(f"{PAGEINDEX_BASE_URL}/{doc_id}", params={"query": query},
                                headers={"api_key": PAGEINDEX_API_KEY}, timeout=60)
        response.raise_for_status()
        retrieval_id = response.json().get("retrieval_id")
        if not retrieval_id:
            continue
        response = requests.get(f"{PAGEINDEX_BASE_URL}/retrieval/{retrieval_id}/",
                                headers={"api_key": PAGEINDEX_API_KEY}, timeout=60)
        response.raise_for_status()
        rank = 0
        for node in response.json().get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    content = item.get("relevant_content", "").strip()
                    if content:
                        rank += 1
                        results.append({"content": content, "score": 1.0 / rank,
                                        "metadata": {"source": record.get("filename", ""),
                                                      "section": item.get("section_title", "")},
                                        "source": "pageindex"})
    return results[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Return PageIndex-shaped results, using API only for uploaded documents."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if PAGEINDEX_API_KEY and _read_manifest():
        try:
            remote = _remote_search(query, top_k)
            if remote:
                return remote
        except Exception as exc:
            print(f"PageIndex unavailable; using local structural fallback: {exc}")
    return _local_structural_search(query, top_k)


if __name__ == "__main__":
    print("PageIndex results:")
    for result in pageindex_search("tuition fee payment", top_k=3):
        print(f"[{result['score']:.3f}] {result['metadata'].get('source')} :: {result['content'][:100]}")
