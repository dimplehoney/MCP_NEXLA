"""
scripts/ingest.py

One-shot CLI script: parse PDFs → chunk → embed → store in ChromaDB.

Run once before starting the MCP server, or whenever the PDF set changes.
Re-running without --force is safe — it exits early if the index already exists.

Usage:
    python scripts/ingest.py               # skip if index exists
    python scripts/ingest.py --force       # rebuild index from scratch
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from project root (src/...)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embeddings.embedder import embed
from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_parser import parse_all_pdfs
from src.vector_store.store import add_chunks, collection_exists, reset_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PDF_DIR = Path(__file__).resolve().parents[1] / "data" / "pdfs"


def ingest_documents(pdf_dir: Path = PDF_DIR, force: bool = False) -> None:
    """
    Full ingestion pipeline: PDF → chunks → embeddings → ChromaDB.

    Args:
        pdf_dir: Directory containing the PDF files.
        force:   If True, re-ingest even if the collection already exists.
    """
    if not force and collection_exists():
        logger.info("Index already exists. Skipping ingestion. Use --force to rebuild.")
        return

    if force:
        # Drop the existing collection so stale chunks (from removed PDFs
        # or pages) don't linger after a rebuild.
        logger.info("Force flag set — clearing existing index.")
        reset_collection()

    logger.info("Starting ingestion from: %s", pdf_dir)

    # --- Step 1: Parse ---
    logger.info("Step 1/3 — Parsing PDFs...")
    pages = parse_all_pdfs(pdf_dir)

    # --- Step 2: Chunk ---
    logger.info("Step 2/3 — Chunking pages...")
    chunks = chunk_pages(pages)

    # --- Step 3: Embed + Store ---
    logger.info("Step 3/3 — Embedding and storing chunks...")
    texts = [c["text"] for c in chunks]
    vectors = embed(texts)
    add_chunks(chunks, vectors)

    logger.info("Ingestion complete. %d chunks indexed from %s.", len(chunks), pdf_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector store.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if the index already exists.",
    )
    args = parser.parse_args()

    ingest_documents(force=args.force)
