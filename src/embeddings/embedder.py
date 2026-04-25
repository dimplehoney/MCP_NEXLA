"""
embedder.py

Responsibility: Convert text strings into dense vector representations
using a local sentence-transformers model.

The model is loaded once at module import time (module-level singleton).
This avoids the ~1-2 second reload penalty on every embed() call, which
matters when the ingest script and the MCP server both call this repeatedly.
"""

try:
    import truststore
except ImportError:
    truststore = None
else:
    truststore.inject_into_ssl()

from sentence_transformers import SentenceTransformer

# Load once. All callers in this process share the same instance.
# all-MiniLM-L6-v2: small (80MB), fast, 384-dim vectors, strong semantic quality.
_MODEL_NAME = "all-MiniLM-L6-v2"
_model = SentenceTransformer(_MODEL_NAME)


def embed(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings and return a list of float vectors.

    - Batches all texts in one forward pass for efficiency.
    - Returns Python lists (not numpy arrays) so ChromaDB and JSON
      serialisation both work without extra conversion.

    Args:
        texts: Non-empty list of strings to embed.

    Returns:
        List of vectors, same length and order as `texts`.
    """
    if not texts:
        return []

    vectors = _model.encode(texts, show_progress_bar=False)
    return vectors.tolist()  # numpy ndarray → plain Python list
