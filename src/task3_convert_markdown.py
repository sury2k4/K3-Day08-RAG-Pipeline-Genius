"""
Task 3 - convert files in data/landing/ to Markdown.

PDF conversion uses MarkItDown first. If the extracted text is unusable, the
script runs OCRmyPDF with Vietnamese and English OCR, converts the temporary
searchable PDF, and writes Markdown only when usable text is available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

SCAN_WARNING_MARKERS = (
    "this source file appears to be a scanned",
    "no selectable text layer was found",
    "full text ocr is required",
    "image-based legal document",
)


def is_usable_extracted_text(text: str, min_chars: int = 200) -> bool:
    normalized = (text or "").strip()
    lowered = normalized.lower()

    if len(normalized) < min_chars:
        return False
    if any(marker in lowered for marker in SCAN_WARNING_MARKERS):
        return False
    return True


def _get_markitdown():
    try:
        from markitdown import MarkItDown

        return MarkItDown()
    except Exception as exc:
        print(f"MarkItDown unavailable: {exc}")
        return None


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


def _check_ocr_environment() -> None:
    missing = []
    ocrmypdf = shutil.which("ocrmypdf")
    tesseract = shutil.which("tesseract")
    if ocrmypdf is None:
        missing.append("OCRmyPDF")
    if tesseract is None:
        missing.append("Tesseract")
    if missing:
        raise RuntimeError(
            "OCRmyPDF is required for scanned PDFs but was not found. "
            "Install OCRmyPDF, Tesseract, and Vietnamese Tesseract language data "
            f"(missing: {', '.join(missing)})."
        )

    completed = subprocess.run(
        [tesseract, "--list-langs"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or "vie" not in completed.stdout.lower():
        raise RuntimeError(
            "Vietnamese OCR language data is required. Install Tesseract language pack 'vie'."
        )


def run_ocrmypdf(source_path: Path, output_path: Path) -> None:
    _check_ocr_environment()
    executable = shutil.which("ocrmypdf")
    command = [
        executable,
        "--language",
        "vie+eng",
        "--skip-text",
        "--deskew",
        "--rotate-pages",
        "--output-type",
        "pdf",
        str(source_path),
        str(output_path),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"OCR failed for {source_path.name}: {completed.stderr.strip()}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"OCR did not create a valid PDF for {source_path.name}")


def create_searchable_pdf_with_ocr(source_path: Path) -> Path:
    temporary_directory = tempfile.mkdtemp(prefix="rag_ocr_")
    output_path = Path(temporary_directory) / f"{source_path.stem}.ocr.pdf"
    run_ocrmypdf(source_path=source_path, output_path=output_path)
    return output_path


def convert_pdf_with_ocr_fallback(source_path: Path, converter) -> tuple[str, bool]:
    direct_result = converter.convert(str(source_path))
    direct_text = (direct_result.text_content or "").strip()

    if is_usable_extracted_text(direct_text):
        return direct_text, False

    with tempfile.TemporaryDirectory(prefix="rag_ocr_") as temp_dir:
        ocr_path = Path(temp_dir) / f"{source_path.stem}.ocr.pdf"
        run_ocrmypdf(source_path=source_path, output_path=ocr_path)

        ocr_result = converter.convert(str(ocr_path))
        ocr_text = (ocr_result.text_content or "").strip()
        if not is_usable_extracted_text(ocr_text):
            raise RuntimeError(
                f"OCR completed but no usable text was extracted from {source_path.name}"
            )
        return ocr_text, True


def _front_matter(filepath: Path, ocr_applied: bool | None = None) -> list[str]:
    meta = _legal_metadata(filepath)
    lines = ["---", f'source_file: "{filepath.name}"']
    if meta.get("document_number"):
        lines.append(f'document_number: "{meta["document_number"]}"')
    if meta.get("source_page_url"):
        lines.append(f'source_page_url: "{meta["source_page_url"]}"')
    if meta.get("stable_source_url"):
        lines.append(f'stable_source_url: "{meta["stable_source_url"]}"')
    if meta.get("official_gazette_url"):
        lines.append(f'official_gazette_url: "{meta["official_gazette_url"]}"')
    if meta.get("download_url"):
        lines.append(f'download_url: "{meta["download_url"]}"')
    if ocr_applied is not None:
        lines.append(f"ocr_applied: {str(ocr_applied).lower()}")
        if ocr_applied:
            lines.append('ocr_engine: "OCRmyPDF/Tesseract"')
            lines.append('ocr_languages: "vie+eng"')
    lines.extend(["---", ""])
    return lines


def _legal_doc_to_markdown(filepath: Path, extracted_text: str, ocr_applied: bool | None = None) -> str:
    meta = _legal_metadata(filepath)
    title = meta.get("title", filepath.stem)
    lines = _front_matter(filepath, ocr_applied=ocr_applied)
    lines.extend([f"# {title}", "", extracted_text.strip()])
    return "\n".join(lines)


def _write_markdown(output_path: Path, content: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Saved: {output_path}")
    return output_path


def _json_article_to_markdown(filepath: Path) -> str:
    data = json.loads(filepath.read_text(encoding="utf-8"))
    content = data.get("content") or data.get("markdown") or data.get("text") or data.get("content_markdown") or ""
    source_url = data.get("source_url") or data.get("url") or ""
    header = [
        "---",
        f'title: "{data.get("title", "Unknown")}"',
        f'topic: "{data.get("topic", "")}"',
        f'source_name: "{data.get("source_name", "")}"',
        f'source_url: "{source_url}"',
        f'crawled_at: "{data.get("crawled_at") or data.get("date_crawled", "")}"',
        f'source_file: "{filepath.name}"',
        'corpus_type: "news"',
        "---",
        "",
    ]
    return "\n".join(header) + content


def convert_file(source_path: str | Path, output_path: str | Path) -> Path:
    """Convert one supported source file to Markdown."""
    source = Path(source_path)
    output = Path(output_path)
    converter = _get_markitdown()

    if source.suffix.lower() == ".json":
        return _write_markdown(output, _json_article_to_markdown(source))
    if source.suffix.lower() == ".pdf":
        if converter is None:
            raise RuntimeError("MarkItDown is required for PDF conversion.")
        text, ocr_applied = convert_pdf_with_ocr_fallback(source, converter)
        return _write_markdown(output, _legal_doc_to_markdown(source, text, ocr_applied=ocr_applied))
    if source.suffix.lower() in {".doc", ".docx"}:
        if converter is None:
            raise RuntimeError("MarkItDown is required for DOC/DOCX conversion.")
        result = converter.convert(str(source))
        text = (result.text_content or "").strip()
        if not is_usable_extracted_text(text):
            raise RuntimeError(f"No usable text was extracted from {source.name}")
        return _write_markdown(output, _legal_doc_to_markdown(source, text, ocr_applied=None))
    if source.suffix.lower() in {".md", ".txt", ".html"}:
        return _write_markdown(output, source.read_text(encoding="utf-8", errors="ignore"))
    raise ValueError(f"Unsupported file type: {source}")


def validate_markdown_file(filepath: Path, legal: bool = False) -> None:
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    lowered = content.lower()

    if len(content.strip()) <= 200:
        raise ValueError(f"{filepath.name} is too short")
    if any(marker in lowered for marker in SCAN_WARNING_MARKERS):
        raise ValueError(f"{filepath.name} contains scan warning text")

    body = content.strip()
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) < 3 or not parts[2].strip():
            raise ValueError(f"{filepath.name} has no content after YAML front matter")
    if legal and not body:
        raise ValueError(f"{filepath.name} has no legal content")


def _existing_markdown_is_valid(output_path: Path, legal: bool = False) -> bool:
    if not output_path.exists():
        return False
    try:
        validate_markdown_file(output_path, legal=legal)
        return True
    except ValueError:
        return False


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOC/DOCX files in data/landing/legal/ to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not legal_dir.exists():
        return []

    converter = _get_markitdown()
    saved_paths = []
    stats = {"success": 0, "ocr": 0, "errors": 0}

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue

        print(f"Converting: {filepath.name}")
        output_path = output_dir / f"{filepath.stem}.md"
        try:
            if converter is None:
                raise RuntimeError("MarkItDown is required for legal document conversion.")

            if filepath.suffix.lower() == ".pdf":
                extracted_text, ocr_applied = convert_pdf_with_ocr_fallback(filepath, converter)
            else:
                result = converter.convert(str(filepath))
                extracted_text = (result.text_content or "").strip()
                ocr_applied = None
                if not is_usable_extracted_text(extracted_text):
                    raise RuntimeError(f"No usable text was extracted from {filepath.name}")

            content = _legal_doc_to_markdown(filepath, extracted_text, ocr_applied=ocr_applied)
            validate_text = content.split("---", 2)[-1] if content.strip().startswith("---") else content
            if not is_usable_extracted_text(validate_text):
                raise RuntimeError(f"Converted Markdown is not usable for {filepath.name}")
            saved_paths.append(_write_markdown(output_path, content))
            stats["success"] += 1
            if ocr_applied:
                stats["ocr"] += 1
        except Exception as exc:
            stats["errors"] += 1
            print(f"ERROR converting {filepath.name}: {exc}")
            print("Install OCRmyPDF, Tesseract, and Tesseract language data 'vie' for scanned PDFs.")
            if _existing_markdown_is_valid(output_path, legal=True):
                print(f"Keeping existing valid Markdown: {output_path}")
                saved_paths.append(output_path)
            else:
                print(f"No valid Markdown was created for: {filepath.name}")

    validation_errors = validate_standardized_dir(OUTPUT_DIR / "legal", legal=True, raise_on_error=False)
    if validation_errors:
        stats["errors"] += len(validation_errors)
        print("Legal Markdown validation failures:")
        for error in validation_errors:
            print(f"  - {error}")

    print(
        "Legal conversion summary: "
        f"success={stats['success']}, ocr={stats['ocr']}, errors={stats['errors']}"
    )
    return saved_paths


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

    validate_standardized_dir(output_dir, legal=False, raise_on_error=False)
    return saved_paths


def validate_standardized_dir(directory: Path, legal: bool = False, raise_on_error: bool = True) -> list[str]:
    errors = []
    if not directory.exists():
        return errors

    for md_file in sorted(directory.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        try:
            validate_markdown_file(md_file, legal=legal)
        except ValueError as exc:
            errors.append(str(exc))

    if errors and raise_on_error:
        raise RuntimeError("; ".join(errors))
    return errors


def convert_all(
    landing_dir: str | Path = LANDING_DIR,
    output_dir: str | Path = OUTPUT_DIR,
) -> list[Path]:
    """Convert all supported landing files to Markdown."""
    global LANDING_DIR, OUTPUT_DIR
    previous_landing, previous_output = LANDING_DIR, OUTPUT_DIR
    LANDING_DIR, OUTPUT_DIR = Path(landing_dir), Path(output_dir)

    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_paths = convert_legal_docs()

    print("\n--- News Articles ---")
    news_paths = convert_news_articles()

    saved_paths = legal_paths + news_paths
    validation_errors = []
    validation_errors.extend(validate_standardized_dir(OUTPUT_DIR / "legal", legal=True, raise_on_error=False))
    validation_errors.extend(validate_standardized_dir(OUTPUT_DIR / "news", legal=False, raise_on_error=False))

    print(f"\nDone. Available Markdown files: {len(saved_paths)} in {OUTPUT_DIR}")
    if validation_errors:
        print("Validation completed with failures:")
        for error in validation_errors:
            print(f"  - {error}")
    LANDING_DIR, OUTPUT_DIR = previous_landing, previous_output
    return saved_paths


if __name__ == "__main__":
    convert_all()
