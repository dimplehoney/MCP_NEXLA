"""
store.py

Responsibility: Persist chunk vectors and their metadata in a local
ChromaDB collection, and retrieve the most semantically similar chunks
for a given query vector.

Design decisions:
  - PersistentClient writes to disk so the index survives process restarts.
    The ingest script runs once; the MCP server reads the same on-disk data.
  - chunk_id is used as the ChromaDB document ID. ChromaDB's upsert()
    replaces an existing document if the ID already exists, making
    re-ingestion safe and idempotent with no manual deduplication needed.
  - Metadata stored per chunk: doc_name, page_num. This is everything
    needed for source attribution in answers — no secondary lookup required.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Directory where ChromaDB writes its on-disk files.
# Kept outside src/ so it is not accidentally committed.
_DB_DIR = Path(__file__).resolve().parents[2] / "chroma_db"
_COLLECTION_NAME = "documents"
_UPSERT_BATCH_SIZE = 2000  # ChromaDB's hard limit is 5461; staying well below it


def _get_collection() -> chromadb.Collection:
    """Open (or create) the persistent collection. Called internally."""
    client = chromadb.PersistentClient(
        path=str(_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    # get_or_create: safe to call every time — returns existing if present.
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for text embeddings
    )


def collection_exists() -> bool:
    """
    Return True if the collection has at least one document.

    Used at server startup to decide whether to skip re-ingestion.
    """
    collection = _get_collection()
    return collection.count() > 0


def reset_collection() -> None:
    """
    Drop and recreate the collection.

    Used by `--force` re-ingestion: idempotent upserts preserve existing
    chunks but cannot remove stale ones (chunks from PDFs or pages that
    no longer exist). A clean rebuild guarantees the index matches the
    current input set exactly.
    """
    client = chromadb.PersistentClient(
        path=str(_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(name=_COLLECTION_NAME)
        logger.info("Existing collection deleted.")
    except Exception:
        # delete_collection raises if it doesn't exist — that's fine.
        logger.info("No existing collection to delete.")
    client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[dict], vectors: list[list[float]]) -> None:
    """
    Upsert chunks and their vectors into the collection.

    Args:
        chunks:  List of chunk dicts from chunker.chunk_pages().
                 Each must have: chunk_id, doc_name, page_num, text.
        vectors: Parallel list of embedding vectors (same order as chunks).

    Raises:
        ValueError if chunks and vectors have different lengths.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but {len(vectors)} vectors."
        )

    if not chunks:
        return

    collection = _get_collection()
    total = len(chunks)

    for start in range(0, total, _UPSERT_BATCH_SIZE):
        batch_chunks = chunks[start : start + _UPSERT_BATCH_SIZE]
        batch_vectors = vectors[start : start + _UPSERT_BATCH_SIZE]
        batch_num = start // _UPSERT_BATCH_SIZE + 1

        collection.upsert(
            ids=[c["chunk_id"] for c in batch_chunks],
            embeddings=batch_vectors,
            documents=[c["text"] for c in batch_chunks],
            metadatas=[
                {"doc_name": c["doc_name"], "page_num": c["page_num"]}
                for c in batch_chunks
            ],
        )
        logger.info(
            "Batch %d: upserted %d chunks (total so far: %d/%d).",
            batch_num, len(batch_chunks), min(start + _UPSERT_BATCH_SIZE, total), total,
        )


def list_document_names() -> list[str]:
    """
    Return a sorted list of unique document names currently in the collection.

    Used by the list_documents MCP tool.
    Returns an empty list if the collection is empty.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    # Fetch all metadata; we only need doc_name, not embeddings or text.
    result = collection.get(include=["metadatas"])
    names = {m["doc_name"] for m in result["metadatas"]}
    return sorted(names)


def _unpack_results(results: dict) -> list[dict]:
    """Convert ChromaDB's nested-list query response into flat result dicts."""
    ids        = results["ids"][0]
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    return [
        {
            "chunk_id": ids[i],
            "text":     documents[i],
            "doc_name": metadatas[i]["doc_name"],
            "page_num": int(metadatas[i]["page_num"]),
            "score":    round(distances[i], 4),
        }
        for i in range(len(ids))
    ]


def query(query_vector: list[float], top_k: int = 5) -> list[dict]:
    """
    Find the top_k most similar chunks to the query vector across all docs.

    Returns a list of result dicts, sorted by similarity (best first):
        {
            "chunk_id": str,
            "text":     str,
            "doc_name": str,
            "page_num": int,
            "score":    float,  # cosine distance (lower = more similar)
        }
    """
    collection = _get_collection()
    logger.info("Querying collection (top_k=%d, total_docs=%d).", top_k, collection.count())

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = _unpack_results(results)
    logger.info("Query returned %d result(s).", len(hits))
    return hits


def query_within_doc(query_vector: list[float], doc_name: str, top_k: int = 2) -> list[dict]:
    """
    Find the top_k most similar chunks within a single document.

    Used by the retriever's per-document pass — guarantees that every
    indexed document gets a chance to contribute context, even when one
    doc's embeddings dominate the global similarity ranking.
    """
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"doc_name": doc_name},
        include=["documents", "metadatas", "distances"],
    )
    return _unpack_results(results)
