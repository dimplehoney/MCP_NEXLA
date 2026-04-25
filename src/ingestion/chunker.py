"""
chunker.py

Responsibility: Split each page's text into overlapping chunks small
enough to embed meaningfully (~400–600 characters each).

Strategy: fixed-size character window with overlap.
  - Simple to reason about and debug
  - Overlap prevents answers from being cut across chunk boundaries
  - Metadata (doc_name, page_num) is preserved on every chunk so the
    vector store can surface source attribution without extra lookups

Why character-based rather than token-based?
  Avoids a tokeniser dependency. At ~4 chars/token, 500 chars ≈ 125 tokens,
  well within any embedding model's limit.
"""

from typing import Iterable


# Tunable constants kept at module level so callers can override if needed.
# 1000 chars (~250 tokens) is roomy enough to keep a multi-sentence finding
# or a small table together, but still well within any embedding model's
# input limit. 150-char overlap stitches answers split across boundaries.
CHUNK_SIZE = 1000     # characters per chunk
CHUNK_OVERLAP = 150   # characters shared between consecutive chunks
MIN_CHUNK_LENGTH = 50 # discard chunks shorter than this (headers, stray lines, etc.)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping windows of `chunk_size` characters.

    Returns an empty list for blank input so callers do not need a guard.
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap  # slide forward, keeping `overlap` chars

    # Drop empty strings and chunks too short to carry meaningful content
    return [c for c in chunks if len(c) >= MIN_CHUNK_LENGTH]


def chunk_pages(pages: Iterable[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Convert a sequence of page records (from pdf_parser) into chunk records.

    Each chunk record:
        {
            "chunk_id":  str,   # "<doc_name>_p<page_num>_c<chunk_index>"
            "doc_name":  str,   # original document name
            "page_num":  int,   # 1-based page the chunk came from
            "text":      str,   # the chunk text itself
        }

    The chunk_id is deterministic, which makes it safe to re-ingest without
    creating duplicate entries if the vector store uses it as a primary key.
    """
    all_chunks: list[dict] = []

    for page in pages:
        doc_name = page["doc_name"]
        page_num = page["page_num"]
        raw_text = page["text"]

        text_chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)

        for idx, chunk_text_content in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": f"{doc_name}_p{page_num}_c{idx}",
                "doc_name": doc_name,
                "page_num": page_num,
                "text": chunk_text_content,
            })

    print(f"  Total chunks created: {len(all_chunks)}")
    return all_chunks
