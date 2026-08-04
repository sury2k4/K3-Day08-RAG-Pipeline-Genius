"""
Task 1 - collect legal documents for the labor-law RAG dataset.

Files are stored in data/landing/legal/. The script does not re-download
manually uploaded documents by default; call collect_legal_docs(download_missing=True)
only when you want to fetch missing files with a download_url.
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_DOCUMENTS = [
    {
        "id": "01",
        "title": "Bộ luật Lao động 2019",
        "document_number": "45/2019/QH14",
        "filename": "01-bo-luat-lao-dong-2019.pdf",
        "source_page_url": "https://vanban.chinhphu.vn/?classid=1&docid=198540&pageid=27160&typegroupid=3",
        "official_gazette_url": "https://congbao.chinhphu.vn/van-ban/nghi-quyet-so-45-2019-qh14-30232/29070.htm",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2019/12/45.signed.pdf",
    },
    {
        "id": "02",
        "title": "Nghị định hướng dẫn Bộ luật Lao động về điều kiện lao động và quan hệ lao động",
        "document_number": "145/2020/NĐ-CP",
        "filename": "02-nghi-dinh-145-2020.pdf",
        "source_page_url": "https://vanban.chinhphu.vn/?docid=201967&pageid=27160",
        "official_gazette_url": "https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-145-2020-nd-cp-32732/33806.htm",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2020/12/145.signed.pdf",
    },
    {
        "id": "03",
        "title": "Thông tư sửa đổi quy định bảo vệ việc làm của người tố cáo làm việc theo hợp đồng lao động",
        "document_number": "09/2021/TT-BLĐTBXH",
        "filename": "03-thong-tu-09-2021-bao-ve-viec-lam.doc",
        "source_page_url": (
            "https://vbpl.vn/van-ban/chi-tiet/"
            "thong-tu-so-09-2021-tt-bldtbxh-sua-doi-bo-sung-mot-so-dieu-cua-"
            "thong-tu-so-08-2020-tt-bldtbxh-ngay-15-10-2020-cua-bo-truong-"
            "bo-lao-dong-thuong-binh-va-xa-hoi-huong-dan-ve-bao-ve-viec-lam-"
            "cua-nguoi-to-cao-la-nguoi-lam-viec-theo-hop-dong-lao-dong--153706"
        ),
        "stable_source_url": "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=153706",
        "official_gazette_url": "https://congbao.chinhphu.vn/van-ban/thong-tu-so-09-2021-tt-bldtbxh-34325/36988.htm",
        "download_url": None,
        "note": "File đã được tải thủ công từ VBPL; không cần tải lại nếu đã nằm trong data/landing/legal/.",
    },
    {
        "id": "04",
        "title": "Nghị định xử phạt vi phạm hành chính trong lĩnh vực lao động",
        "document_number": "12/2022/NĐ-CP",
        "filename": "04-nghi-dinh-12-2022-xu-phat-lao-dong.pdf",
        "source_page_url": "https://vanban.chinhphu.vn/?classid=1&docid=205182&orggroupid=2&pageid=27160",
        "official_gazette_url": "https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-12-2022-nd-cp-36716.htm",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/12-2022-nd.signed.pdf",
    },
    {
        "id": "05",
        "title": "Nghị định quy định mức lương tối thiểu đối với người lao động",
        "document_number": "293/2025/NĐ-CP",
        "filename": "05-nghi-dinh-293-2025-luong-toi-thieu.pdf",
        "source_page_url": "https://vanban.chinhphu.vn/?classid=1&docid=215832&pageid=27160",
        "official_gazette_url": "https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-293-2025-nd-cp-46568/59713.htm",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/11/293-cp.signed.pdf",
    },
]


def setup_directory() -> None:
    """Create data/landing/legal/ if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _safe_log(message: str) -> None:
    """Print without crashing on Windows consoles that are not UTF-8."""
    print(message.encode("ascii", errors="replace").decode("ascii"))


def find_existing_document(doc: dict) -> Path | None:
    """Find exact filename first, then any manually uploaded file with same id prefix."""
    exact_path = DATA_DIR / doc["filename"]
    if exact_path.exists() and exact_path.stat().st_size > 1024:
        return exact_path

    valid_extensions = {".pdf", ".doc", ".docx"}
    matches = [
        path
        for path in DATA_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in valid_extensions
        and path.name.startswith(doc["id"])
        and path.stat().st_size > 1024
    ]
    return matches[0] if matches else None


def download_document(doc: dict, overwrite: bool = False) -> Path | None:
    """Download one document when download_url exists."""
    download_url = doc.get("download_url")
    if not download_url:
        return None

    filepath = DATA_DIR / doc["filename"]
    if filepath.exists() and filepath.stat().st_size > 1024 and not overwrite:
        return filepath

    response = requests.get(download_url, timeout=60)
    response.raise_for_status()
    filepath.write_bytes(response.content)
    return filepath


def collect_legal_docs(download_missing: bool = False, overwrite: bool = False) -> list[Path]:
    """Return available legal documents, optionally downloading missing ones."""
    setup_directory()
    collected = []

    for doc in LEGAL_DOCUMENTS:
        filepath = None if overwrite else find_existing_document(doc)
        if filepath is None and download_missing:
            filepath = download_document(doc, overwrite=overwrite)
        if filepath is None:
            _safe_log(f"Missing manual file: {doc['filename']} ({doc['document_number']})")
            continue

        collected.append(filepath)
        _safe_log(f"Ready: {filepath.name} - {doc['document_number']}")

    return collected


if __name__ == "__main__":
    files = collect_legal_docs(download_missing=False)
    _safe_log(f"Found {len(files)} legal documents in {DATA_DIR}")
