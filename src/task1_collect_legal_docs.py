"""
Task 1 - Collect Vietnamese labor-law source documents.

The lab corpus uses official Vietnamese labor-law documents from Government
sources. Existing user-provided files in data/landing/legal/ are kept intact.
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_DOCUMENTS = [
    {
        "document_number": "45/2019/QH14",
        "title": "Bo luat Lao dong 2019",
        "url": (
            "https://datafiles.chinhphu.vn/cpp/files/"
            "vbpq/2019/12/45.signed.pdf"
        ),
        "filename": "01_bo_luat_lao_dong_2019.pdf",
    },
    {
        "document_number": "145/2020/ND-CP",
        "title": "Nghi dinh huong dan Bo luat Lao dong",
        "url": (
            "https://datafiles.chinhphu.vn/cpp/files/"
            "vbpq/2020/12/145.signed.pdf"
        ),
        "filename": "02_nghi_dinh_145_2020_huong_dan.pdf",
    },
    {
        "document_number": "12/2022/ND-CP",
        "title": "Nghi dinh xu phat vi pham hanh chinh trong linh vuc lao dong",
        "url": (
            "https://datafiles.chinhphu.vn/cpp/files/"
            "vbpq/2022/01/12-2022-nd.signed.pdf"
        ),
        "filename": "04_nghi_dinh_12_2022_xu_phat.pdf",
    },
    {
        "document_number": "293/2025/ND-CP",
        "title": "Nghi dinh quy dinh muc luong toi thieu",
        "url": (
            "https://datafiles.chinhphu.vn/cpp/files/"
            "vbpq/2025/11/293-cp.signed.pdf"
        ),
        "filename": "05_nghi_dinh_293_2025_luong_toi_thieu.pdf",
    },
]

VALID_EXTENSIONS = {".pdf", ".docx", ".doc"}


def setup_directory():
    """Create data/landing/legal/ if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {DATA_DIR}")


def _valid_existing_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def download_file(url: str, filename: str) -> Path:
    """Download one official PDF safely and return its local path."""
    setup_directory()
    filepath = DATA_DIR / filename
    if _valid_existing_file(filepath):
        print(f"Already exists, skip download: {filepath}")
        return filepath

    part_path = filepath.with_suffix(filepath.suffix + ".part")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    content = response.content
    if filename.lower().endswith(".pdf") and not content.startswith(b"%PDF"):
        if part_path.exists():
            part_path.unlink()
        raise ValueError(f"Downloaded content is not a valid PDF: {url}")

    part_path.write_bytes(content)
    part_path.replace(filepath)
    print(f"Downloaded: {filepath}")
    return filepath


def download_all() -> list[Path]:
    """Download known legal documents and validate all local legal files."""
    setup_directory()
    errors = []

    for doc in LEGAL_DOCUMENTS:
        try:
            download_file(doc["url"], doc["filename"])
        except Exception as exc:
            errors.append(f"{doc['filename']}: {exc}")
            print(f"[ERROR] {doc['filename']}: {exc}")

    valid_files = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
        and path.stat().st_size > 0
    )

    print(f"Valid legal files found: {len(valid_files)}")
    for path in valid_files:
        print(f"  - {path.name}")

    if len(valid_files) < 3:
        detail = "; ".join(errors) if errors else "not enough local files"
        raise RuntimeError(f"Need at least 3 valid legal files ({detail}).")

    return valid_files


if __name__ == "__main__":
    files = download_all()
    print(f"Task 1 completed: {len(files)} valid legal files.")
