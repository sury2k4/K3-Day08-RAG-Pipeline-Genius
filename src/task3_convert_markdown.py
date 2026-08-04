"""
Task 3 - Convert landing legal/news files to Markdown.

Scanned PDFs are detected but not OCR'ed here. OCR is a manual prerequisite for
this lab step, so the script will not write Markdown that only contains a scan
warning.
"""

import json
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

INVALID_TEXT_MARKERS = (
    "this source file appears to be a scanned",
    "no selectable text layer was found",
    "full text ocr is required",
    "image-based legal document",
)


def is_valid_extracted_text(text: str, min_chars: int = 200) -> bool:
    normalized = (text or "").strip()
    lowered = normalized.lower()
    if len(normalized) < min_chars:
        return False
    return not any(marker in lowered for marker in INVALID_TEXT_MARKERS)


def _get_markitdown():
    from markitdown import MarkItDown

    return MarkItDown()


def _yaml_value(value: object) -> str:
    text = str(value or "").replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _write_markdown(output_path: Path, content: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"  Saved: {output_path}")
    return output_path


def _legal_markdown(filepath: Path, text: str) -> str:
    relative_path = filepath.relative_to(LANDING_DIR).as_posix()
    front_matter = [
        "---",
        f"title: {_yaml_value(filepath.stem)}",
        f"source_file: {_yaml_value(filepath.name)}",
        f"source_path: {_yaml_value(relative_path)}",
        'corpus_type: "legal"',
        "---",
        "",
    ]
    return "\n".join(front_matter) + text.strip()


def _news_content(data: dict) -> str:
    return (
        data.get("content_markdown")
        or data.get("content")
        or data.get("markdown")
        or data.get("text")
        or ""
    )


def _news_markdown(filepath: Path, data: dict, content: str) -> str:
    title = data.get("title", filepath.stem)
    source_url = data.get("source_url") or data.get("url", "")
    crawled_at = data.get("crawled_at") or data.get("date_crawled", "")
    topic = data.get("topic", "")
    source_name = data.get("source_name", "")

    front_matter = [
        "---",
        f"title: {_yaml_value(title)}",
        f"topic: {_yaml_value(topic)}",
        f"source_name: {_yaml_value(source_name)}",
        f"source_url: {_yaml_value(source_url)}",
        f"crawled_at: {_yaml_value(crawled_at)}",
        f"source_file: {_yaml_value(filepath.name)}",
        'corpus_type: "news"',
        "---",
        "",
        f"# {title}",
        "",
    ]
    return "\n".join(front_matter) + content.strip()


def convert_legal_docs():
    """Convert PDF/DOCX files in data/landing/legal/ to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"[WARN] Legal landing directory does not exist: {legal_dir}")
        return []

    md = _get_markitdown()
    saved_paths = []

    for filepath in sorted(legal_dir.iterdir()):
        suffix = filepath.suffix.lower()
        if suffix == ".doc":
            print(f"[WARN] {filepath.name}: convert .doc to .docx or PDF before Task 3.")
            continue
        if suffix not in (".pdf", ".docx"):
            continue

        print(f"Converting: {filepath.name}")
        output_path = output_dir / f"{filepath.stem}.md"
        try:
            result = md.convert(str(filepath))
            text = (result.text_content or "").strip()
            if not is_valid_extracted_text(text):
                print(f"[ERROR] {filepath.name}: no text layer; OCR is required first.")
                continue
            saved_paths.append(_write_markdown(output_path, _legal_markdown(filepath, text)))
        except Exception as exc:
            print(f"[ERROR] {filepath.name}: {exc}")

    return saved_paths


def convert_news_articles():
    """Convert crawled JSON articles in data/landing/news/ to Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print(f"[WARN] News landing directory does not exist: {news_dir}")
        return []

    saved_paths = []
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            content = str(_news_content(data)).strip()
            if len(content) < 200:
                print(f"[ERROR] {filepath.name}: content is too short.")
                continue

            output_path = output_dir / f"{filepath.stem}.md"
            saved_paths.append(_write_markdown(output_path, _news_markdown(filepath, data, content)))
        except Exception as exc:
            print(f"[ERROR] {filepath.name}: {exc}")

    return saved_paths


def _markdown_files_with_invalid_markers() -> list[Path]:
    bad_files = []
    if not OUTPUT_DIR.exists():
        return bad_files
    for md_file in OUTPUT_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8", errors="ignore").lower()
        if any(marker in content for marker in INVALID_TEXT_MARKERS):
            bad_files.append(md_file)
    return bad_files


def convert_all():
    """Convert all supported landing files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_paths = convert_legal_docs()

    print("\n--- News Articles ---")
    news_paths = convert_news_articles()

    output_paths = legal_paths + news_paths
    all_markdown = list(OUTPUT_DIR.rglob("*.md")) if OUTPUT_DIR.exists() else []
    bad_files = _markdown_files_with_invalid_markers()
    if bad_files:
        names = ", ".join(path.name for path in bad_files)
        raise RuntimeError(f"Invalid OCR warning text found in Markdown: {names}")
    if not all_markdown:
        raise RuntimeError("No Markdown files found after conversion.")

    print(f"\nLegal Markdown created: {len(legal_paths)}")
    print(f"News Markdown created: {len(news_paths)}")
    print(f"Total Markdown available: {len(all_markdown)}")
    print("Output directory:", OUTPUT_DIR)
    return output_paths


if __name__ == "__main__":
    convert_all()
