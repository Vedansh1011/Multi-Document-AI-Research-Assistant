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


def _extract_column_aware_text(page) -> str:
    """
    Extract page text in reading order for multi-column PDFs.

    The page header is read top-to-bottom first. After the
    abstract/body begins, two-column content is read from the
    left column top-to-bottom followed by the right column.
    """

    page_width = page.rect.width

    blocks = page.get_text(
        "blocks",
        sort=False,
    )

    text_blocks = []

    for block in blocks:
        if len(block) < 7:
            continue

        x0, y0, x1, y1, text, *_ = block

        if not text or not text.strip():
            continue

        block_type = block[6] if len(block) > 6 else 0

        if block_type != 0:
            continue

        text_blocks.append(
            {
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "text": text.strip(),
                "width": x1 - x0,
            }
        )

    if not text_blocks:
        return ""

    # Find the beginning of the main document body.
    # IEEE-style papers commonly begin the body with "Abstract".
    abstract_blocks = [
        block
        for block in text_blocks
        if re.search(
            r"\babstract\b",
            block["text"],
            flags=re.IGNORECASE,
        )
    ]

    if abstract_blocks:
        body_start_y = min(
            block["y0"]
            for block in abstract_blocks
        )
    else:
        body_start_y = min(
            block["y0"]
            for block in text_blocks
        )

    header_blocks = [
        block
        for block in text_blocks
        if block["y0"] < body_start_y
    ]

    body_blocks = [
        block
        for block in text_blocks
        if block["y0"] >= body_start_y
    ]

    # Read the page header naturally from top to bottom.
    header_blocks.sort(
        key=lambda block: (
            round(block["y0"], 1),
            round(block["x0"], 1),
        )
    )

    if len(body_blocks) < 4:
        ordered_blocks = header_blocks + sorted(
            body_blocks,
            key=lambda block: (
                round(block["y0"], 1),
                round(block["x0"], 1),
            ),
        )

        return "\n\n".join(
            block["text"]
            for block in ordered_blocks
        )

    page_midpoint = page_width / 2

    left_blocks = []
    right_blocks = []

    for block in body_blocks:
        center_x = (block["x0"] + block["x1"]) / 2

        if center_x < page_midpoint:
            left_blocks.append(block)
        else:
            right_blocks.append(block)

    left_blocks.sort(
        key=lambda block: (
            round(block["y0"], 1),
            round(block["x0"], 1),
        )
    )

    right_blocks.sort(
        key=lambda block: (
            round(block["y0"], 1),
            round(block["x0"], 1),
        )
    )

    ordered_blocks = (
        header_blocks
        + left_blocks
        + right_blocks
    )

    return "\n\n".join(
        block["text"]
        for block in ordered_blocks
    )


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF page by page.

    Uses column-aware extraction for multi-column documents
    while preserving page-level metadata.
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
            text = _extract_column_aware_text(page)

            text = text.strip()

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