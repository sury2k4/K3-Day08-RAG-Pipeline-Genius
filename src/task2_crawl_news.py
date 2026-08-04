"""
Task 2 - crawl university service/news articles.

The crawler uses Crawl4AI when available. If Playwright/browser setup or
network access is unavailable, it falls back to curated article records so the
rest of the RAG pipeline still has valid JSON inputs.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Fill this list after you choose the news/article sources for Task 2.
ARTICLE_URLS = []

FALLBACK_ARTICLES = {}


def setup_directory() -> None:
    """Create data/landing/news/ if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _fallback_article(url: str, error: str | None = None) -> dict:
    data = FALLBACK_ARTICLES.get(
        url,
        {
            "title": f"University service article from {urlparse(url).netloc}",
            "content_markdown": "University service information for students.",
        },
    )
    return {
        "url": url,
        "title": data["title"],
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": data["content_markdown"].strip(),
        "crawler": "fallback",
        "crawl_error": error,
    }


async def crawl_article(url: str) -> dict:
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
        content = getattr(result, "markdown", "") or ""
        metadata = getattr(result, "metadata", {}) or {}
        if len(content.strip()) < 200:
            return _fallback_article(url, "Crawled content was too short.")
        return {
            "url": url,
            "title": metadata.get("title") or FALLBACK_ARTICLES.get(url, {}).get("title", "Unknown"),
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "content_markdown": content,
            "crawler": "crawl4ai",
            "crawl_error": None,
        }
    except Exception as exc:
        return _fallback_article(url, str(exc))


def _article_filename(index: int, article: dict) -> str:
    slug = "".join(
        ch if ch.isalnum() else "-"
        for ch in article.get("title", f"article-{index}").lower()
    ).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)
    return f"{index:02d}-{slug[:60]}.json"


async def crawl_all() -> list[Path]:
    """Crawl all configured article URLs into JSON files."""
    setup_directory()
    if not ARTICLE_URLS:
        print("ARTICLE_URLS is empty. Add news/article URLs before running Task 2.")
        return []

    saved_paths = []

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)
        filepath = DATA_DIR / _article_filename(i, article)
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_paths.append(filepath)
        print(f"Saved: {filepath}")

    return saved_paths


if __name__ == "__main__":
    asyncio.run(crawl_all())
