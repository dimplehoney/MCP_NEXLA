"""
scripts/demo.py

Run a set of realistic questions through the full retrieve → synthesize
pipeline and print copy-paste-ready output. Used to generate the example
interaction log in the README.

Usage:
    python scripts/demo.py                     # all built-in questions
    python scripts/demo.py --question "..."    # one custom question
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.synthesizer import synthesize
from src.retrieval.retriever import retrieve

QUESTIONS = [
    "What was Salesforce's total revenue in fiscal year 2020?",
    "Compare the risk factors discussed by Salesforce and Berkshire Hathaway.",
    "What was TCS's attrition rate in IT services in FY2020?",
    "What is the company's stated policy on cryptocurrency or blockchain investments?",
]


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "  (no sources)"
    lines = []
    for s in sources:
        snippet = s.get("snippet", "").strip()
        if len(snippet) > 160:
            snippet = snippet[:160] + "..."
        lines.append(f"  • [{s['doc']}  —  page {s['page']}]")
        if snippet:
            lines.append(f'    "{snippet}"')
    return "\n".join(lines)


def _print_result(index: int, question: str, result: dict) -> None:
    divider = "─" * 72
    print(f"\n{divider}")
    print(f"Q{index}. {question}")
    print(divider)
    print(f"\nAnswer:\n  {result['answer']}")
    print(f"\nSources:\n{_format_sources(result.get('sources', []))}")


def run_all() -> None:
    print("=" * 72)
    print("  Document Q&A — Example Interaction Log")
    print("=" * 72)
    for i, question in enumerate(QUESTIONS, start=1):
        chunks = retrieve(question)
        result = synthesize(question, chunks)
        _print_result(i, question, result)
    print(f"\n{'─' * 72}\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo the document Q&A pipeline.")
    parser.add_argument("--question", type=str, help="Ask a single custom question.")
    args = parser.parse_args()

    if args.question:
        chunks = retrieve(args.question)
        result = synthesize(args.question, chunks)
        _print_result(1, args.question, result)
    else:
        run_all()
