"""
Task 3 - convert files in data/landing/ to Markdown.

Output keeps the same child directory structure under data/standardized/.
MarkItDown is used when installed; JSON news files are converted directly
because they already contain Markdown content from Task 2.
"""

import json
import re
import warnings
from pathlib import Path
from urllib.parse import urljoin

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _get_markitdown():
    try:
        from markitdown import MarkItDown

        return MarkItDown()
    except Exception as exc:
        print(f"MarkItDown unavailable, using fallback conversion: {exc}")
        return None


def _fallback_markdown(filepath: Path, error: Exception | None = None) -> str:
    note = f"\n\nConversion note: {error}" if error else ""
    return (
        f"# {filepath.stem}\n\n"
        f"**Source file:** {filepath.name}\n\n"
        "This Markdown placeholder was created because the source document "
        "could not be parsed in the current environment. The original file is "
        f"available in `{filepath}` for later full conversion.{note}\n"
    )


def _legal_metadata(filepath: Path) -> dict:
    try:
        from src.task1_collect_legal_docs import LEGAL_DOCUMENTS
    except Exception:
        try:
            from task1_collect_legal_docs import LEGAL_DOCUMENTS
        except Exception:
            return {}

    for doc in LEGAL_DOCUMENTS:
        if filepath.name.startswith(doc["id"]) or filepath.name == doc["filename"]:
            return doc
    return {}


def _extract_pdf_text(filepath: Path) -> str:
    texts = []
    try:
        import pdfplumber

        with pdfplumber.open(filepath) as pdf:
            for page_number, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    texts.append(f"\n\n## Page {page_number}\n\n{page_text.strip()}")
    except Exception:
        texts = []

    if texts:
        return "".join(texts).strip()

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(filepath))
        for page_number, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                texts.append(f"\n\n## Page {page_number}\n\n{page_text.strip()}")
    except Exception:
        return ""

    return "".join(texts).strip()


def _fetch_url(url: str) -> str:
    import requests

    response = requests.get(url, timeout=30)
    if response.status_code >= 400:
        response.raise_for_status()
    return response.text


def _download_bytes(url: str) -> bytes:
    import requests

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.content
    except requests.exceptions.SSLError:
        # The official Cong Bao CDN can fail local CA verification on Windows.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            response = requests.get(url, timeout=60, verify=False)
        response.raise_for_status()
        return response.content


def _find_doc_links(page_url: str) -> list[str]:
    html = _fetch_url(page_url)
    links = []
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        anchors = soup.find_all("a", href=True)
        for anchor in anchors:
            text = anchor.get_text(" ", strip=True).lower()
            href = anchor["href"]
            if ".doc" in text or ".doc" in href.lower():
                links.append(urljoin(page_url, href))
    except Exception:
        for match in re.findall(r'href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, flags=re.I | re.S):
            href, text = match
            if ".doc" in text.lower() or ".doc" in href.lower():
                links.append(urljoin(page_url, href))

    deduped = []
    for link in links:
        if link not in deduped:
            deduped.append(link)
    return deduped


def _extract_doc_binary_text(data: bytes) -> str:
    """Extract readable text from legacy .doc by decoding UTF-16LE streams."""
    raw = data.decode("utf-16le", errors="ignore")
    starts = [
        raw.find(marker)
        for marker in (
            "QUỐC HỘI",
            "CHÍNH PHỦ",
            "BỘ LAO ĐỘNG",
            "CỘNG HÒA",
            "CỘNG HOÀ",
            "BỘ LUẬT",
        )
    ]
    starts = [index for index in starts if index >= 0]
    if starts:
        raw = raw[min(starts):]

    punctuation = set(".,;:!?()[]{}-/–—“”\"'\n\t %0123456789")
    chars = []
    for char in raw:
        if char.isalpha() or char.isspace() or char in punctuation:
            chars.append(char)
        else:
            chars.append("\n")

    text = "".join(chars).replace("\r", "\n")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if len(line) < 2:
            continue
        if sum(char.isalpha() for char in line) == 0 and len(line) > 20:
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_official_gazette_doc_text(meta: dict) -> str:
    page_url = meta.get("official_gazette_url")
    if not page_url:
        return ""

    parts = []
    for link in _find_doc_links(page_url):
        try:
            text = _extract_doc_binary_text(_download_bytes(link))
        except Exception as exc:
            print(f"Could not extract DOC from {link}: {exc}")
            continue
        if len(text) > 200:
            parts.append(text)
    return "\n\n".join(parts).strip()


def _legal_doc_to_markdown(filepath: Path, extracted_text: str = "") -> str:
    meta = _legal_metadata(filepath)
    title = meta.get("title", filepath.stem)
    lines = [
        f"# {title}",
        "",
        f"**Source file:** {filepath.name}",
    ]
    if meta.get("document_number"):
        lines.append(f"**Document number:** {meta['document_number']}")
    if meta.get("source_page_url"):
        lines.append(f"**Source page:** {meta['source_page_url']}")
    if meta.get("stable_source_url"):
        lines.append(f"**Stable source:** {meta['stable_source_url']}")
    if meta.get("official_gazette_url"):
        lines.append(f"**Official gazette:** {meta['official_gazette_url']}")
    if meta.get("download_url"):
        lines.append(f"**Download URL:** {meta['download_url']}")

    lines.extend(["", "---", ""])
    if extracted_text.strip():
        lines.append(extracted_text.strip())
    else:
        lines.extend(
            [
                "## Extraction status",
                "",
                "No selectable text layer was found in the local source file.",
                "If this file is a scanned PDF and no official DOC/HTML fallback is available, OCR is required for complete legal-content extraction.",
                "The metadata above is preserved so retrieval can still cite the official document source.",
            ]
        )
    return "\n".join(lines)


def _write_markdown(output_path: Path, content: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Saved: {output_path}")
    return output_path


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOC/DOCX files in data/landing/legal/ to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not legal_dir.exists():
        return []

    md_converter = _get_markitdown()
    saved_paths = []
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue

        print(f"Converting: {filepath.name}")
        output_path = output_dir / f"{filepath.stem}.md"
        try:
            meta = _legal_metadata(filepath)
            if filepath.suffix.lower() == ".pdf":
                extracted = _extract_pdf_text(filepath)
                if len(extracted) < 200:
                    extracted = _extract_official_gazette_doc_text(meta)
                content = _legal_doc_to_markdown(filepath, extracted)
            else:
                if md_converter is None:
                    raise RuntimeError("MarkItDown is not available.")
                result = md_converter.convert(str(filepath))
                content = _legal_doc_to_markdown(filepath, result.text_content)
        except Exception as exc:
            content = _fallback_markdown(filepath, exc)
        saved_paths.append(_write_markdown(output_path, content))

    return saved_paths


def _json_article_to_markdown(filepath: Path) -> str:
    data = json.loads(filepath.read_text(encoding="utf-8"))
    header = [
        f"# {data.get('title', 'Unknown')}",
        "",
        f"**Source:** {data.get('url', 'N/A')}",
        f"**Crawled:** {data.get('date_crawled', 'N/A')}",
        "",
        "---",
        "",
    ]
    return "\n".join(header) + data.get("content_markdown", "")


def convert_news_articles() -> list[Path]:
    """Convert crawled JSON/HTML/text news files to Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not news_dir.exists():
        return []

    saved_paths = []
    for filepath in sorted(news_dir.iterdir()):
        if not filepath.is_file() or filepath.name.startswith("."):
            continue

        print(f"Converting: {filepath.name}")
        output_path = output_dir / f"{filepath.stem}.md"
        if filepath.suffix.lower() == ".json":
            content = _json_article_to_markdown(filepath)
        elif filepath.suffix.lower() in {".md", ".txt", ".html"}:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        saved_paths.append(_write_markdown(output_path, content))

    return saved_paths


def convert_all() -> list[Path]:
    """Convert all supported landing files to Markdown."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_paths = convert_legal_docs()

    print("\n--- News Articles ---")
    news_paths = convert_news_articles()

    saved_paths = legal_paths + news_paths
    print(f"\nDone. Wrote {len(saved_paths)} files to: {OUTPUT_DIR}")
    return saved_paths


if __name__ == "__main__":
    convert_all()
