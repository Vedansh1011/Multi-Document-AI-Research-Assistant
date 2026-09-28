from pathlib import Path
import re

import pymupdf


def is_index_page(text: str) -> bool:
    """Detect pages that primarily contain an index or table of contents."""
    normalized_text = text.lower()

    index_markers = [
        "index",
        "table of contents",
        "contents",
        "page no.",
    ]

    marker_count = sum(
        marker in normalized_text
        for marker in index_markers
    )

    numbered_entries = len(
        re.findall(
            r"(?m)^\s*\d+\.\s+",
            text,
        )
    )

    return marker_count >= 2 or (
        marker_count >= 1 and numbered_entries >= 5
    )


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF page by page.

    Index/table-of-contents pages are excluded from the
    searchable document corpus.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    document = pymupdf.open(path)
    pages = []

    try:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            text = page.get_text("text").strip()

            if not text:
                continue

            if is_index_page(text):
                continue

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "source": path.name,
                }
            )
    finally:
        document.close()

    return pages