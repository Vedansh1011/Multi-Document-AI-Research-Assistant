from pathlib import Path
import re


def clean_text(text: str) -> str:
    """Normalize extracted PDF text while preserving readable structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 200,
    chunk_overlap: int = 40,
) -> list[dict]:
    """
    Create word-based chunks while preserving source and page metadata.

    Chunk IDs are generated using the source filename so that
    chunk identities remain unique across multiple documents.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    chunk_counter = 1
    step = chunk_size - chunk_overlap

    for page in pages:
        text = clean_text(page["text"])

        if not text:
            continue

        words = text.split()

        if len(words) < 10:
            continue

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                continue

            source_name = Path(page["source"]).stem

            chunks.append(
                {
                    "chunk_id": (
                        f"{source_name}_chunk_{chunk_counter:04d}"
                    ),
                    "source": page["source"],
                    "page_number": page["page_number"],
                    "text": " ".join(chunk_words),
                }
            )

            chunk_counter += 1

            if start + chunk_size >= len(words):
                break

    return chunks