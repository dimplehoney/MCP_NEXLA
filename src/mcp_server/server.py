"""
mcp_server/server.py

The MCP server entry point. Exposes two tools to any MCP-compatible client:
    - query_documents  → RAG-based Q&A over the indexed PDFs
    - list_documents   → list all indexed document names

This module is intentionally thin: it handles MCP protocol wiring and
startup checks only. All business logic lives in the pipeline modules.

Data flow for query_documents:
    question (str)
        → retriever.retrieve()      [embed question + vector search + rebalance]
        → synthesizer.synthesize()  [build prompt + call GPT-4o-mini]
        → JSON string response

How to run:
    python -m src.mcp_server.server

MCP clients (e.g. Claude Desktop) connect via stdio transport.
"""

import json
import logging
import sys
from pathlib import Path

# Allow running as `python -m src.mcp_server.server` from anywhere by
# adding the project root to sys.path.
# File: <project_root>/src/mcp_server/server.py
#   parents[0] = mcp_server, parents[1] = src, parents[2] = <project_root>
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

from src.llm.synthesizer import synthesize
from src.retrieval.retriever import retrieve
from src.vector_store.store import collection_exists, list_document_names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Server instance ---
mcp = FastMCP(
    name="Document Q&A Server",
    instructions=(
        "This server answers natural language questions grounded in a set of "
        "indexed PDF documents. Use query_documents to ask questions and "
        "list_documents to see what is available."
    ),
)


# ---------------------------------------------------------------------------
# Tool: query_documents
# ---------------------------------------------------------------------------

@mcp.tool()
def query_documents(question: str) -> str:
    """
    Answer a natural language question using the indexed PDF documents.

    Retrieves the most relevant passages from the document index, then
    synthesizes a grounded answer with source attribution.

    Args:
        question: A natural language question about the document content.

    Returns:
        A JSON string with:
            {
              "answer":  "<answer text>",
              "sources": [{"doc": "...", "page": N, "snippet": "..."}]
            }
        If no relevant content is found, answer will be "Not found in documents."
    """
    logger.info("Tool called: query_documents | question=%r", question)

    chunks = retrieve(question)
    result = synthesize(question, chunks)

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool: list_documents
# ---------------------------------------------------------------------------

@mcp.tool()
def list_documents() -> str:
    """
    List all document names currently indexed in the vector store.

    Useful for understanding what source material is available before
    asking questions.

    Returns:
        A JSON string with:
            {"documents": ["doc_name_1", "doc_name_2", ...]}
    """
    logger.info("Tool called: list_documents")

    names = list_document_names()
    return json.dumps({"documents": names}, indent=2)


# ---------------------------------------------------------------------------
# Startup: ensure index is built before accepting tool calls
# ---------------------------------------------------------------------------

def _ensure_index() -> None:
    """
    Check that the vector store is populated before the server starts.

    If the index is empty, attempt automatic ingestion from the default
    PDF directory. This keeps the server self-sufficient for first-run
    without requiring a manual ingest step.
    """
    if collection_exists():
        logger.info("Index found. Server ready.")
        return

    logger.warning(
        "No index found. Running ingestion automatically — "
        "this can take several minutes for large PDF sets."
    )
    logger.warning(
        "If your MCP client times out, run `python scripts/ingest.py` "
        "once before starting the server."
    )

    # Import here (not at top) to keep the ingestion dependency optional —
    # a pre-built index means ingest is never imported at all.
    from scripts.ingest import ingest_documents
    ingest_documents()


if __name__ == "__main__":
    _ensure_index()
    logger.info("Starting MCP server (stdio transport)...")
    mcp.run(transport="stdio")
