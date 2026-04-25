"""
retrieval/retriever.py

Responsibility: Convert a natural language question into a query vector,
search the vector store, and return the most relevant chunks with metadata.

This module is the bridge between raw user input and the vector store.
It knows about both embeddings and the store — but not about the LLM or MCP.

Multi-document retrieval strategy:
  A naive global top-k often returns chunks from a single dominant document
  whose embeddings are uniformly closer to the query — even when the question
  explicitly spans multiple documents. (Empirically: a "compare A and B" query
  returned 100% A in the top 30, with zero B chunks.)

  To guarantee cross-document coverage we run two passes:
    1. Per-document query — top `PER_DOC_TOP_K` chunks from EACH indexed doc.
    2. Global query — top `GLOBAL_TOP_K` chunks across the whole index.
  Results are merged, deduplicated, filtered by the distance threshold,
  resorted by score, and trimmed to `top_k`.

  This costs N+1 vector queries (small, fast) and gives the LLM a balanced
  view: every document gets a fair shot, but the strongest hits still rank
  first.
"""

import logging

from src.embeddings.embedder import embed
from src.vector_store.store import list_document_names
from src.vector_store.store import query as store_query
from src.vector_store.store import query_within_doc

logger = logging.getLogger(__name__)

# How many chunks the synthesizer ultimately sees.
DEFAULT_TOP_K = 8

# Per-doc retrieval — best-in-doc chunks for each indexed document.
# Guarantees every document gets a chance to contribute context.
PER_DOC_TOP_K = 2

# Global retrieval — best chunks across the whole index, regardless of doc.
# Fills in extra coverage for the dominant document(s).
GLOBAL_TOP_K = 6

# Cosine *distance* threshold (0 = identical, 2 = opposite).
# Chunks with a distance at or above this value are dropped before LLM
# synthesis — passing low-relevance chunks gives the model noise to
# hallucinate from. 0.7 has worked well empirically: in-scope questions
# typically score 0.3–0.6, fully off-topic questions score 0.9+.
SCORE_THRESHOLD = 0.7


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for a natural language question.

    Combines a per-document pass (guarantees cross-document coverage) with
    a global pass (preserves the strongest matches). Drops chunks above the
    distance threshold and returns the merged result sorted best-first.

    Returns an empty list if the store has no documents or all chunks fall
    above the threshold.
    """
    logger.info("Retrieving chunks for question: %r (top_k=%d)", question, top_k)

    question_vector = embed([question])[0]

    # --- Per-document pass: best `PER_DOC_TOP_K` from EACH document ---
    per_doc_hits: list[dict] = []
    for doc_name in list_document_names():
        per_doc_hits.extend(
            query_within_doc(question_vector, doc_name=doc_name, top_k=PER_DOC_TOP_K)
        )

    # --- Global pass: best `GLOBAL_TOP_K` across the whole index ---
    global_hits = store_query(question_vector, top_k=GLOBAL_TOP_K)

    # Merge, dedupe by chunk_id, drop below-threshold matches.
    seen: set[str] = set()
    merged: list[dict] = []
    for c in per_doc_hits + global_hits:
        if c["chunk_id"] in seen:
            continue
        if c["score"] >= SCORE_THRESHOLD:
            continue
        seen.add(c["chunk_id"])
        merged.append(c)

    if not merged:
        logger.warning(
            "All retrieved chunks exceeded score threshold (%.2f). Returning empty.",
            SCORE_THRESHOLD,
        )
        return []

    # Sort merged pool by score (lower distance = better) and trim.
    merged.sort(key=lambda c: c["score"])
    chunks = merged[:top_k]

    logger.info(
        "Retrieved %d chunk(s) from document(s): %s",
        len(chunks),
        sorted({c["doc_name"] for c in chunks}),
    )
    return chunks
