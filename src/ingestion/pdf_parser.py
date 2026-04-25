"""
pdf_parser.py

Responsibility: Open each PDF in a directory and extract raw text
on a per-page basis, along with the document name and page number.

Output is a flat list of page records — one dict per page — that the
chunker downstream can consume without knowing anything about PDFs.
"""

from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF


def parse_pdf(pdf_path: Path) -> Generator[dict, None, None]:
    """
    Yield one record per page from a single PDF file.

    Each record:
        {
            "doc_name": str,   # filename without extension, e.g. "annual_report"
            "page_num": int,   # 1-based page number
            "text": str,       # raw extracted text for that page
        }

    Pages with no extractable text (e.g. pure-image scans) are skipped.
    """
    doc_name = pdf_path.stem  # "report_2024" from "report_2024.pdf"

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:  # skip blank / image-only pages
                yield {
                    "doc_name": doc_name,
                    "page_num": page_index,
                    "text": text,
                }


def parse_all_pdfs(pdf_dir: Path) -> list[dict]:
    """
    Parse every PDF found in pdf_dir and return a flat list of page records.

    Raises FileNotFoundError if the directory does not exist.
    Raises ValueError if no PDFs are found.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    # rglob finds PDFs both at the top level and inside subdirectories
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {pdf_dir}")

    pages: list[dict] = []
    for pdf_path in pdf_files:
        print(f"  Parsing: {pdf_path.name}")
        for page_record in parse_pdf(pdf_path):
            pages.append(page_record)

    print(f"  Total pages extracted: {len(pages)} across {len(pdf_files)} document(s)")
    return pages
