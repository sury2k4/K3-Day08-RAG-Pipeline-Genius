"""
Task 2 - Crawl labor-law news/explainer articles into JSON landing files.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_SOURCES = [
    {
        "slug": "01_hop_dong_thu_viec_bhxh",
        "topic": "Thu viec va bao hiem xa hoi",
        "source_name": "Bao Dien tu Chinh phu",
        "url": (
            "https://baochinhphu.vn/"
            "hop-dong-thu-viec-khong-dong-bhxh-bat-buoc-"
            "102260303141003744.htm"
        ),
    },
    {
        "slug": "02_noi_dung_hop_dong_lao_dong",
        "topic": "Noi dung hop dong lao dong",
        "source_name": "Xay dung Chinh sach, Phap luat",
        "url": (
            "https://xaydungchinhsach.chinhphu.vn/"
            "noi-dung-cua-hop-dong-lao-dong-va-hop-dong-"
            "lam-viec-co-gi-khac-nhau-119230830114203459.htm"
        ),
    },
    {
        "slug": "03_lam_them_gio",
        "topic": "Thoi gio lam viec va lam them gio",
        "source_name": "Bao Dien tu Chinh phu",
        "url": (
            "https://baochinhphu.vn/"
            "lam-them-gio-the-nao-la-dung-quy-dinh-"
            "102240130085419703.htm"
        ),
    },
    {
        "slug": "04_nghi_phep_hang_nam",
        "topic": "Nghi hang nam",
        "source_name": "Bao Dien tu Chinh phu",
        "url": (
            "https://baochinhphu.vn/"
            "lam-viec-chua-du-12-thang-tinh-ngay-nghi-phep-"
            "the-nao-102260319095651248.htm"
        ),
    },
    {
        "slug": "05_don_phuong_cham_dut_hop_dong",
        "topic": "Don phuong cham dut hop dong lao dong",
        "source_name": "Xay dung Chinh sach, Phap luat",
        "url": (
            "https://xaydungchinhsach.chinhphu.vn/"
            "nhung-truong-hop-nguoi-lao-dong-co-quyen-"
            "don-phuong-cham-dut-hop-dong-lao-dong-"
            "11923052610360935.htm"
        ),
    },
    {
        "slug": "06_luong_toi_thieu_2026",
        "topic": "Muc luong toi thieu",
        "source_name": "Xay dung Chinh sach, Phap luat",
        "url": (
            "https://xaydungchinhsach.chinhphu.vn/"
            "nghi-dinh-so-293-2025-nd-cp-quy-dinh-muc-"
            "luong-toi-thieu-doi-voi-nguoi-lao-dong-lam-"
            "viec-theo-hop-dong-lao-dong-"
            "119251110172808433.htm"
        ),
    },
]

ARTICLE_URLS = [item["url"] for item in ARTICLE_SOURCES]


def setup_directory():
    """Create data/landing/news/ if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_markdown(result) -> str:
    """Handle Crawl4AI result.markdown as string or object with raw_markdown."""
    markdown = getattr(result, "markdown", "")
    if isinstance(markdown, str):
        return markdown.strip()

    raw_markdown = getattr(markdown, "raw_markdown", "")
    if isinstance(raw_markdown, str):
        return raw_markdown.strip()

    return ""


def _source_for_url(url: str) -> dict:
    for source in ARTICLE_SOURCES:
        if source["url"] == url:
            return source
    return {
        "slug": Path(url).stem or "article",
        "topic": "",
        "source_name": "",
        "url": url,
    }


def _json_has_valid_content(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    content = (
        data.get("content_markdown")
        or data.get("content")
        or data.get("markdown")
        or data.get("text")
        or ""
    )
    return bool(data.get("url")) and len(str(content).strip()) >= 200


async def crawl_article(url: str) -> dict:
    """
    Crawl one article and return metadata plus Markdown content.
    """
    from crawl4ai import AsyncWebCrawler

    source = _source_for_url(url)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    if hasattr(result, "success") and result.success is False:
        error = getattr(result, "error_message", "") or "crawler returned success=False"
        raise RuntimeError(error)

    content = extract_markdown(result)
    if len(content) < 200:
        raise ValueError(f"Crawled content is too short: {len(content)} chars")

    metadata = getattr(result, "metadata", {}) or {}
    title = metadata.get("title") if isinstance(metadata, dict) else None
    crawled_at = datetime.now().isoformat()

    return {
        "url": url,
        "source_url": url,
        "title": title or source["topic"] or "Unknown",
        "topic": source["topic"],
        "source_name": source["source_name"],
        "date_crawled": crawled_at,
        "crawled_at": crawled_at,
        "content_markdown": content,
        "content": content,
    }


async def crawl_all():
    """Crawl all configured labor-law article sources."""
    setup_directory()
    errors = []

    for i, source in enumerate(ARTICLE_SOURCES, 1):
        output_path = DATA_DIR / f"{source['slug']}.json"
        if output_path.exists() and _json_has_valid_content(output_path):
            print(f"[{i}/{len(ARTICLE_SOURCES)}] Existing valid JSON: {output_path.name}")
            continue

        print(f"[{i}/{len(ARTICLE_SOURCES)}] Crawling: {source['url']}")
        try:
            article = await crawl_article(source["url"])
            output_path.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  Saved: {output_path}")
        except Exception as exc:
            errors.append(f"{source['slug']}: {exc}")
            print(f"  [ERROR] {source['slug']}: {exc}")

    valid_json = sorted(
        path
        for path in DATA_DIR.glob("*.json")
        if path.is_file() and _json_has_valid_content(path)
    )

    if len(valid_json) < 5:
        detail = "; ".join(errors) if errors else "not enough valid JSON files"
        raise RuntimeError(f"Need at least 5 valid news JSON files ({detail}).")

    print(f"Task 2 completed: {len(valid_json)} valid JSON files.")
    return valid_json


if __name__ == "__main__":
    asyncio.run(crawl_all())
