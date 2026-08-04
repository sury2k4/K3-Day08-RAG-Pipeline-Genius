"""
Task 2 - crawl university service/news articles.

The crawler uses Crawl4AI when available. If Playwright/browser setup or
network access is unavailable, it falls back to curated article records so the
rest of the RAG pipeline still has valid JSON inputs.
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_SOURCES = [
    {
        "slug": "01-hop-dong-thu-viec-khong-dong-bhxh",
        "topic": "Thu viec va bao hiem xa hoi",
        "source_name": "Bao Dien tu Chinh phu",
        "url": "https://baochinhphu.vn/hop-dong-thu-viec-khong-dong-bhxh-bat-buoc-102260303141003744.htm",
    },
    {
        "slug": "02-noi-dung-hop-dong-lao-dong",
        "topic": "Noi dung hop dong lao dong",
        "source_name": "Xay dung Chinh sach, Phap luat",
        "url": "https://xaydungchinhsach.chinhphu.vn/noi-dung-cua-hop-dong-lao-dong-va-hop-dong-lam-viec-co-gi-khac-nhau-119230830114203459.htm",
    },
    {
        "slug": "03-lam-them-gio-dung-quy-dinh",
        "topic": "Thoi gio lam viec va lam them gio",
        "source_name": "Bao Dien tu Chinh phu",
        "url": "https://baochinhphu.vn/lam-them-gio-the-nao-la-dung-quy-dinh-102240130085419703.htm",
    },
    {
        "slug": "04-nghi-phep-chua-du-12-thang",
        "topic": "Nghi hang nam",
        "source_name": "Bao Dien tu Chinh phu",
        "url": "https://baochinhphu.vn/lam-viec-chua-du-12-thang-tinh-ngay-nghi-phep-the-nao-102260319095651248.htm",
    },
    {
        "slug": "05-don-phuong-cham-dut-hop-dong",
        "topic": "Don phuong cham dut hop dong lao dong",
        "source_name": "Xay dung Chinh sach, Phap luat",
        "url": "https://xaydungchinhsach.chinhphu.vn/nhung-truong-hop-nguoi-lao-dong-co-quyen-don-phuong-cham-dut-hop-dong-lao-dong-11923052610360935.htm",
    },
    {
        "slug": "06-luong-toi-thieu-2026",
        "topic": "Muc luong toi thieu",
        "source_name": "Xay dung Chinh sach, Phap luat",
        "url": "https://xaydungchinhsach.chinhphu.vn/nghi-dinh-so-293-2025-nd-cp-quy-dinh-muc-luong-toi-thieu-doi-voi-nguoi-lao-dong-lam-viec-theo-hop-dong-lao-dong-119251110172808433.htm",
    },
]

ARTICLE_URLS = [item["url"] for item in ARTICLE_SOURCES]

FALLBACK_ARTICLES = {}


def setup_directory() -> None:
    """Create data/landing/news/ if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _source_for_url(url: str) -> dict:
    for source in ARTICLE_SOURCES:
        if source["url"] == url:
            return source
    return {}


def _is_bad_content(content: str) -> bool:
    lowered = (content or "").lower()
    markers = ("access denied", "captcha", "403 forbidden", "not found", "error 404")
    return any(marker in lowered for marker in markers)


def _normalize_article(data: dict) -> dict:
    source_url = data.get("source_url") or data.get("url") or ""
    source = _source_for_url(source_url)
    content = data.get("content") or data.get("markdown") or data.get("text") or data.get("content_markdown") or ""
    normalized = dict(data)
    normalized.update(
        {
            "title": data.get("title") or source.get("topic") or "Untitled",
            "topic": data.get("topic") or source.get("topic") or "",
            "source_name": data.get("source_name") or source.get("source_name") or urlparse(source_url).netloc,
            "source_url": source_url,
            "crawled_at": data.get("crawled_at") or data.get("date_crawled") or datetime.now(timezone.utc).isoformat(),
            "content_format": data.get("content_format") or "markdown",
            "content": content,
        }
    )
    return normalized


def validate_news_files(input_dir: str | Path = DATA_DIR) -> list[dict]:
    """Validate and normalize existing JSON news files without crawling."""
    directory = Path(input_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for filepath in sorted(directory.glob("*.json")):
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            normalized = _normalize_article(data)
            valid = bool(
                normalized["title"].strip()
                and normalized["source_url"].startswith(("http://", "https://"))
                and normalized["content"].strip()
                and not _is_bad_content(normalized["content"])
            )
            reason = "" if valid else "missing title/url/content or bad page content"
        except Exception as exc:
            normalized = {}
            valid = False
            reason = str(exc)
        results.append(
            {
                "filename": filepath.name,
                "path": str(filepath),
                "valid": valid,
                "reason": reason,
                "article": normalized,
            }
        )
    return results


def extract_markdown(result) -> str:
    markdown = getattr(result, "markdown", "")
    if hasattr(markdown, "raw_markdown"):
        return markdown.raw_markdown or ""
    return markdown or ""


def _fallback_article(url: str, error: str | None = None) -> dict:
    source = _source_for_url(url)
    data = FALLBACK_ARTICLES.get(
        url,
        {
            "title": source.get("topic") or f"Labor law article from {urlparse(url).netloc}",
            "content_markdown": "",
        },
    )
    article = {
        "title": data["title"],
        "topic": source.get("topic", ""),
        "source_name": source.get("source_name", urlparse(url).netloc),
        "source_url": url,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "content_format": "markdown",
        "content": data["content_markdown"].strip(),
        "url": url,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": data["content_markdown"].strip(),
        "crawler": "fallback",
        "crawl_error": error,
    }
    return article


async def crawl_article(
    url: str,
    output_dir: str | Path | None = None,
    *,
    slug: str | None = None,
    topic: str | None = None,
) -> dict:
    """
    Crawl one article and return metadata plus Markdown content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str,
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        if hasattr(result, "success") and not result.success:
            raise RuntimeError(getattr(result, "error_message", "Crawl4AI returned success=False"))
        content = extract_markdown(result)
        metadata = getattr(result, "metadata", {}) or {}
        if len(content.strip()) < 200 or _is_bad_content(content):
            raise RuntimeError("Crawled content was too short or looked like an error page.")
        source = _source_for_url(url)
        article = {
            "url": url,
            "source_url": url,
            "title": metadata.get("title") or FALLBACK_ARTICLES.get(url, {}).get("title", "Unknown"),
            "topic": topic or source.get("topic", ""),
            "source_name": source.get("source_name", urlparse(url).netloc),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "content_format": "markdown",
            "content": content,
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "content_markdown": content,
            "crawler": "crawl4ai",
            "crawl_error": None,
        }
        if output_dir is not None:
            output_path = Path(output_dir) / f"{slug or _article_filename(1, article)[:-5]}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        return article
    except Exception as exc:
        return _fallback_article(url, str(exc))


def _article_filename(index: int, article: dict) -> str:
    slug = "".join(
        ch if ch.isalnum() else "-"
        for ch in article.get("title", f"article-{index}").lower()
    ).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)
    return f"{index:02d}-{slug[:60]}.json"


async def crawl_all(crawl_missing: bool = True, force: bool = False) -> list[Path]:
    """Crawl all configured article URLs into JSON files."""
    setup_directory()
    valid_existing = [row for row in validate_news_files(DATA_DIR) if row["valid"]]
    if len(valid_existing) >= 5 and not force:
        print(f"Found {len(valid_existing)} valid news JSON files; skipping crawl.")
        return []

    saved_paths = []
    existing_urls = {row["article"].get("source_url") for row in valid_existing}
    for i, source in enumerate(ARTICLE_SOURCES, 1):
        url = source["url"]
        if url in existing_urls and not force:
            continue
        if not crawl_missing and not force:
            continue
        print(f"[{i}/{len(ARTICLE_SOURCES)}] Crawling: {url}")
        article = await crawl_article(url, slug=source["slug"], topic=source["topic"])
        if not article.get("content"):
            print(f"  ERROR: {url}: {article.get('crawl_error')}")
            continue
        filepath = DATA_DIR / f"{source['slug']}.json"
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_paths.append(filepath)
        print(f"Saved: {filepath}")

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 2: validate/crawl labor-law articles")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--crawl-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    setup_directory()
    before = validate_news_files(DATA_DIR)
    valid_before = [row for row in before if row["valid"]]
    print(f"Valid news JSON before crawl: {len(valid_before)}")

    if not args.validate_only:
        asyncio.run(crawl_all(crawl_missing=args.crawl_missing or len(valid_before) < 5, force=args.force))

    after = validate_news_files(DATA_DIR)
    valid_after = [row for row in after if row["valid"]]
    for row in after:
        status = "valid" if row["valid"] else f"invalid: {row['reason']}"
        print(f"{row['filename']} | {status}")
    print(f"Valid news JSON after crawl: {len(valid_after)}")
