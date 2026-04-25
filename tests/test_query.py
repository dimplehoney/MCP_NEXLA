"""
tests/test_query.py

Two modes:
  • Default                            → assert-based test suite (PASS/FAIL output)
  • `--question "..."`                 → CLI mode: clean user-facing answer only

Prerequisites:
    - Dependencies installed  (pip install -r requirements.txt)
    - .env file present       (OPENAI_API_KEY set)
    - Index built             (python scripts/ingest.py)

Run:
    python tests/test_query.py
    python tests/test_query.py --question "What was Salesforce's total revenue?"
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NOT_FOUND = "Not found in documents."
DIVIDER = "=" * 50


# ---------------------------------------------------------------------------
# CLI mode (--question)
# ---------------------------------------------------------------------------

def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "None"
    return "\n".join(f"[{s['doc']} — page {s['page']}]" for s in sources)


def _print_cli_result(question: str, result: dict) -> None:
    answer = result.get("answer", "").strip()
    sources = result.get("sources", [])

    if answer == NOT_FOUND:
        explanation = "No relevant information was found in the indexed documents."
    else:
        explanation = "Answer generated from retrieved document context."

    print(DIVIDER)
    print("Question:")
    print(question)
    print()
    print("Answer:")
    print(answer)
    print()
    print("Explanation:")
    print(explanation)
    print()
    print("Sources:")
    print(_format_sources(sources))
    print(DIVIDER)


def run_cli(question: str) -> None:
    """Answer one question and print only the user-facing result."""
    # Silence library logs ("All retrieved chunks exceeded threshold", etc.)
    logging.disable(logging.CRITICAL)

    from src.llm.synthesizer import synthesize
    from src.retrieval.retriever import retrieve

    chunks = retrieve(question)
    result = synthesize(question, chunks)
    _print_cli_result(question, result)


# ---------------------------------------------------------------------------
# Test suite mode (default)
# ---------------------------------------------------------------------------

def _check(label: str, condition: bool, detail: str = "") -> None:
    """Tiny assert helper that prints PASS/FAIL inline."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    assert condition, f"{label} — {detail}"


def test_index_is_built() -> None:
    from src.vector_store.store import collection_exists
    print("\n→ test_index_is_built")
    _check("collection has documents", collection_exists())


def test_in_scope_question_returns_grounded_answer() -> None:
    from src.llm.synthesizer import synthesize
    from src.retrieval.retriever import retrieve

    print("\n→ test_in_scope_question_returns_grounded_answer")
    question = "What was Salesforce's total revenue in fiscal year 2020?"
    chunks = retrieve(question)
    _check("retrieved at least one chunk", len(chunks) > 0, f"got {len(chunks)}")
    result = synthesize(question, chunks)

    _check("answer is non-empty", bool(result.get("answer")))
    _check(
        "answer is not the not-found fallback",
        result["answer"] != NOT_FOUND,
        f"answer={result['answer']!r}",
    )
    _check("answer cites at least one source", len(result.get("sources", [])) > 0)
    for s in result["sources"]:
        _check("source has doc + page", "doc" in s and "page" in s, str(s))


def test_out_of_scope_question_returns_not_found() -> None:
    from src.llm.synthesizer import synthesize
    from src.retrieval.retriever import retrieve

    print("\n→ test_out_of_scope_question_returns_not_found")
    question = "What is the recipe for sourdough bread?"
    chunks = retrieve(question)
    result = synthesize(question, chunks)

    _check(
        "answer is the not-found fallback",
        result["answer"] == NOT_FOUND,
        f"answer={result['answer']!r}",
    )
    _check("sources list is empty", result["sources"] == [])


def test_cross_document_retrieval_covers_multiple_docs() -> None:
    from src.retrieval.retriever import retrieve

    print("\n→ test_cross_document_retrieval_covers_multiple_docs")
    question = "Compare the risk factors discussed by Salesforce and Berkshire Hathaway."
    chunks = retrieve(question)

    docs = {c["doc_name"] for c in chunks}
    _check(
        "retrieval surfaced more than one document",
        len(docs) >= 2,
        f"docs={sorted(docs)}",
    )


TESTS = [
    test_index_is_built,
    test_in_scope_question_returns_grounded_answer,
    test_out_of_scope_question_returns_not_found,
    test_cross_document_retrieval_covers_multiple_docs,
]


def run_tests() -> int:
    print("=" * 72)
    print("  Document Q&A — Test Suite")
    print("=" * 72)

    failed = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {test.__name__} failed: {e}")

    print("\n" + "─" * 72)
    if failed == 0:
        print(f"  All {len(TESTS)} tests passed.")
        return 0
    print(f"  {failed} of {len(TESTS)} tests FAILED.")
    return 1


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test or query the document Q&A pipeline.")
    parser.add_argument("--question", type=str, help="Ask a single question (CLI mode).")
    args = parser.parse_args()

    if args.question:
        run_cli(args.question)
    else:
        sys.exit(run_tests())
