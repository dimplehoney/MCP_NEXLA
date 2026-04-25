# Document Q&A MCP Server

An MCP server that answers natural language questions grounded in a set of PDF documents. Built as a take-home assignment for Nexla.

---

## Project Overview

This system lets any MCP-compatible AI client (e.g. Claude Desktop) ask questions about a collection of PDF documents and receive accurate, cited answers.

It uses a RAG (Retrieval-Augmented Generation) pipeline: PDFs are parsed, split into chunks, embedded locally, and stored in a vector database. At query time the most relevant chunks are retrieved using a hybrid per-document + global strategy, and passed to GPT-4o-mini to synthesize a grounded answer with source attribution.

---

## Architecture Overview

```
PDFs (data/pdfs/)
 │
 ▼
[PDF Parser] Extract text per page (PyMuPDF)
 │
 ▼
[Chunker] Split into overlapping 1000-char chunks
 │ Metadata attached: doc_name, page_num
 ▼
[Embedder] Local sentence-transformers model
 │ all-MiniLM-L6-v2 → 384-dim vectors
 ▼
[ChromaDB] Persistent local vector store
 │ Indexed by chunk_id (idempotent upserts)
 │
 │ ── at query time ──────────────────────────────────
 │
 ▼
[Retriever] Embed question, then run TWO retrieval passes:
 │ • per-document: top-2 from EACH doc
 │ • global: top-6 across the index
 │ Merge, dedupe, drop chunks above 0.7 distance,
 │ resort by score, trim to top-8.
 ▼
[Synthesizer] Build context prompt → call GPT-4o-mini
 │ temperature=0, response_format=json_object
 ▼
[MCP Server] Return JSON: { answer, sources }
```

**Key design choices:**
- Embeddings run fully locally — no embedding API cost, no extra network call
- The LLM is called only once per query, for final synthesis
- `chunk_id` is deterministic → re-ingestion is always safe and idempotent
- Distance threshold filters irrelevant chunks before they reach the LLM
- Per-document retrieval pass guarantees cross-document coverage even when one document's embeddings dominate the global ranking

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd MCP_NEXLA
```

### 2. Create a virtual environment with Python 3.12

> Ensure Python 3.10+ is installed (the `mcp` package does not support older versions).

Create the virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

> If PowerShell shows an execution policy error, run:
> ```bash
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

**macOS/Linux:**
```bash
cp .env.example .env
```

**Windows:**
```bash
copy .env.example .env
```

Then open `.env` and set `OPENAI_API_KEY=your_key_here`.

### 5. Add your PDF documents

Place PDFs anywhere inside `data/pdfs/` (subdirectories are supported):

```
data/
└── pdfs/
 ├── report_one.pdf
 └── subfolder/
 └── report_two.pdf
```

> **Note on the included `*_qa.jsonl` files:** the `data/pdfs/<id>/` folders ship with `*_qa.jsonl` ground-truth Q&A files alongside each PDF. They are not used at runtime — they are reference data for offline evaluation. Feel free to delete them; the ingestion pipeline ignores everything that isn't a `.pdf`.

### 6. Run ingestion

Parse, chunk, embed, and index all PDFs into ChromaDB:

```bash
python scripts/ingest.py
```

To force a clean rebuild (drops the existing collection first, so stale chunks from removed PDFs don't linger):

```bash
python scripts/ingest.py --force
```

### 7. Run the test suite or ask a question

```bash
# Assert-based test suite — verifies the contract end-to-end
python tests/test_query.py

# Ask a single question — prints clean user-facing output only
python tests/test_query.py --question "What was Salesforce's total revenue?"

# Human-readable demo run of pre-set questions (used to generate EXAMPLES.md)
python scripts/demo.py
```

### 8. Start the MCP server

```bash
python -m src.mcp_server.server
```

The server runs on stdio transport, ready for any MCP-compatible client to connect.

Once connected, clients can call:
- query_documents
- list_documents

### Connecting to an MCP Client

MCP is an open standard — this server works with any MCP-compatible client, not just Claude Desktop.

#### Claude Desktop

Config file location:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add the `document-qa` entry (replace the path with your actual clone location):

```json
{
 "mcpServers": {
 "document-qa": {
 "command": "/absolute/path/to/MCP_NEXLA/venv/bin/python",
 "args": ["-m", "src.mcp_server.server"],
 "cwd": "/absolute/path/to/MCP_NEXLA",
 "env": {
 "PYTHONPATH": "/absolute/path/to/MCP_NEXLA"
 }
 }
 }
}
```

Then **fully quit and relaunch Claude Desktop** (Cmd+Q on macOS, not just closing the window).

#### Cursor

Config file location: `~/.cursor/mcp.json` (create it if it doesn't exist)

```json
{
 "mcpServers": {
 "document-qa": {
 "command": "/absolute/path/to/MCP_NEXLA/venv/bin/python",
 "args": ["-m", "src.mcp_server.server"],
 "cwd": "/absolute/path/to/MCP_NEXLA",
 "env": {
 "PYTHONPATH": "/absolute/path/to/MCP_NEXLA"
 }
 }
 }
}
```

Restart Cursor after saving.

#### Verifying the connection

After restarting your client, open a new chat and look for the tools icon (hammer/`+` button near the input box). You should see `query_documents` and `list_documents` listed. If they don't appear, check the logs:

- **Claude Desktop logs:** `~/Library/Logs/Claude/mcp-server-document-qa.log`
- **Common fix:** make sure `PYTHONPATH` is set in the config — without it Python cannot locate the `src` package even when `cwd` is correct.

---

## MCP Tools Documentation

### `query_documents`

Ask a natural language question and receive a grounded answer from the indexed PDFs.

| | |
|---|---|
| **Input** | `question: str` — a natural language question |
| **Output** | JSON string with `answer` and `sources` |

**Output schema:**
```json
{
 "answer": "The answer derived from the documents.",
 "sources": [
 {
 "doc": "document_name",
 "page": 4,
 "snippet": "Exact quote from the source text."
 }
 ]
}
```

If no relevant content is found:
```json
{
 "answer": "Not found in documents.",
 "sources": []
}
```

**Example call:**
```
query_documents("What was Salesforce's total revenue in fiscal year 2020?")
```

---

### `list_documents`

List all document names currently indexed in the vector store.

| | |
|---|---|
| **Input** | none |
| **Output** | JSON string with a list of document names |

**Output schema:**
```json
{
 "documents": ["NYSE_CRM_2020", "NYSE_BRK-A_2021", "OTC_TCS_2020", "NYSE_TME_2021", "ASX_AJY_2020"]
}
```

---

## Example Queries and Outputs

The full example interaction log lives in [`EXAMPLES.md`](EXAMPLES.md). Three highlights below — all are real, unedited model outputs from `python scripts/demo.py`.

---

**Query 1 — Single-document factual lookup**

```
Question: What was Salesforce's total revenue in fiscal year 2020?
```

```json
{
 "answer": "Salesforce's total revenue in fiscal year 2020 was $17.1 billion.",
 "sources": [
 {
 "doc": "NYSE_CRM_2020",
 "page": 45,
 "snippet": "Total fiscal 2020 revenue was $17.1 billion, an increase of 29 percent year-over-year."
 }
 ]
}
```

---

**Query 2 — Cross-document comparison**

```
Question: Compare the risk factors discussed by Salesforce and Berkshire Hathaway.
```

```json
{
 "answer": "Berkshire Hathaway discusses the inherent risks in the insurance business... In contrast, Salesforce highlights the volatility of its common stock... while both companies recognize significant risks, Berkshire focuses on underwriting and catastrophic events in insurance, whereas Salesforce emphasizes stock price volatility and investment risks.",
 "sources": [
 { "doc": "NYSE_BRK-A_2021", "page": 139, "snippet": "Mistakes in assessing insurance risks can be huge..." },
 { "doc": "NYSE_BRK-A_2021", "page": 139, "snippet": "We will most certainly not have an underwriting profit in 16 of the next 17 years." },
 { "doc": "NYSE_CRM_2020", "page": 67, "snippet": "the financial success of our investment in any company is typically dependent on a liquidity event." },
 { "doc": "NYSE_CRM_2020", "page": 37, "snippet": "the market price of our common stock is likely to be volatile..." }
 ]
}
```

> Cross-document synthesis only works because the retriever runs a per-document pass in addition to the global one — without it, Berkshire's embeddings dominate this query so heavily that zero Salesforce chunks reach the top 30 globally.

---

**Query 3 — Out of scope (hallucination guard)**

```
Question: What is the company's stated policy on cryptocurrency investments?
```

```json
{
 "answer": "Not found in documents.",
 "sources": []
}
```

> No chunk passed the distance threshold, so the LLM was never called — the fallback was returned directly.

---

## Hallucination Control

| Layer | Mechanism |
|---|---|
| **Retrieval filtering** | Chunks with cosine *distance* ≥ 0.7 are discarded before reaching the LLM (lower distance = more similar; in-scope questions typically score 0.3–0.6, fully off-topic questions score 0.9+) |
| **Deterministic output** | `temperature=0` — no randomness, no creative filling of gaps |
| **Strict prompt rules** | The model is explicitly instructed to use only the provided sources and not infer beyond what is stated |
| **JSON-locked output** | `response_format={"type": "json_object"}` — the API guarantees valid JSON, no fence-stripping needed |
| **Partial information handling** | If only partial context is found, the model states what is known and flags what is missing — it does not guess |
| **Explicit fallback** | If no chunks pass the threshold, the system returns `"Not found in documents."` without ever calling the LLM |

---

## Known Limitations

- **Table structure is not preserved.** PyMuPDF extracts text sequentially. Multi-column financial tables (e.g. income statements) may be extracted out of order, causing values to lose their column context. PyMuPDF does have `page.find_tables()` for table-aware extraction; integrating it is a clear future-work item.
- **Image-only pages are skipped.** Scanned PDFs or pages with no text layer produce no chunks and are silently ignored. Adding OCR (e.g. Tesseract) would close this gap.
- **Chunk boundary splits.** A sentence may be cut across two chunks. The 150-character overlap mitigates this but does not eliminate it.
- **No conversational memory.** Each `query_documents` call is independent. Follow-up questions do not have access to prior answers.
- **English-only.** The embedding model (`all-MiniLM-L6-v2`) is English; multilingual queries will degrade.

---

## Testing and Validation

`tests/test_query.py` is an assert-based suite covering the four behaviours that matter most:

| Test | What it verifies |
|---|---|
| `test_index_is_built` | The vector store is populated |
| `test_in_scope_question_returns_grounded_answer` | A factual query returns a non-empty answer with sources, never the fallback |
| `test_out_of_scope_question_returns_not_found` | An off-topic query returns exactly `"Not found in documents."` with no sources |
| `test_cross_document_retrieval_covers_multiple_docs` | A comparison question retrieves chunks from at least two distinct documents |

The "not found" path and the multi-document path are the most important. Both confirm the contract: no hallucination on absence, no single-doc dominance on cross-doc questions.

---

## Design Decisions

### Technology choices

| Component | Choice | Reason |
|---|---|---|
| PDF parsing | PyMuPDF | Fast, accurate page-level extraction; preserves page numbers |
| Embeddings | `all-MiniLM-L6-v2` | Runs fully locally — no API cost, no second network call, strong semantic quality for English |
| Vector store | ChromaDB | Simple Python API, persistent local storage, cosine similarity built-in, no server needed |
| LLM | GPT-4o-mini (OpenAI) | Fast, cheap, excellent JSON instruction-following; `response_format=json_object` enforces valid JSON at the API level |
| MCP framework | FastMCP (Anthropic MCP SDK) | Official standard, minimal boilerplate, stdio transport out of the box |

### Tradeoffs

**Two-pass retrieval (per-doc + global)** — A single global cosine search returns chunks from whichever document's embeddings happen to be closer to the query. Empirically, for a "compare A and B" question the top 30 global hits were 100% A and 0% B. The per-document pass guarantees every doc gets a fair shot; the global pass keeps the strongest matches first. Cost: N+1 small ChromaDB queries instead of 1 — negligible.

**Local embeddings over API embeddings** — Slightly lower quality than `text-embedding-3-small`, but eliminates an embedding-API dependency and the per-query network cost.

**Character-based chunking over token-based** — Avoids a tokeniser dependency. At ~4 chars/token, 1000 chars ≈ 250 tokens, well within any embedding model's input limit.

**Distance threshold (0.7) as a hallucination guard** — If the best retrieved chunk is still semantically far from the question, passing it to the LLM gives it noise to reason from. Filtering at retrieval is cheaper and more reliable than asking the model to self-assess confidence.

**Batched upserts in ChromaDB** — ChromaDB enforces a hard batch limit of 5461 documents per call. With ~4500 chunks across 5 annual reports, inserts are split into batches of 2000 to stay well below the limit and remain robust to larger corpora.

---

## Vibe Coding Section

I used **Claude** (via a coding interface) as my primary AI coding assistant, with a secondary feedback loop using **ChatGPT** to validate design decisions and catch blind spots.

### How I used AI tools

I followed an iterative, two-step workflow:

**Design + Implementation (Claude)**
- Prompted Claude to design the overall RAG pipeline architecture
- Asked it to implement each module step-by-step (parser, chunker, embeddings, retriever, synthesizer, MCP server)
- Gave structured, scoped instructions to keep the system minimal and modular

**Validation + Refinement (ChatGPT + manual review)**
- After each phase, cross-checked design and code decisions
- Used ChatGPT to sanity-check architecture choices, surface potential issues (batch limits, hallucination risk, retrieval quality), and improve clarity
- Fed refined instructions back to Claude

This back-and-forth (Claude ↔ ChatGPT ↔ me) helped avoid blindly accepting generated code and surfaced issues early.

### Concrete example: the multi-document retrieval bug

The most instructive case was multi-document retrieval. Claude's first cut used a single global top-k cosine search. Locally it looked correct — single-doc questions worked, the "Not found" fallback worked. But when I tested *"Compare risk factors of Salesforce and Berkshire Hathaway,"* the system returned `"Not found in documents."`

I dug into the raw ChromaDB results and saw that the top 30 global hits were 100% Berkshire — there was simply no Salesforce content in the candidate pool, so the model couldn't compare. ChatGPT suggested re-ranking the candidate pool for diversity. I implemented that, but the test still failed, because the issue wasn't ranking — it was that Salesforce never made it into the pool at all.

I switched to a hybrid: one ChromaDB query per indexed document (top-2 each, via a `where` filter) plus one global query, then merge and trim. That made the multi-document test pass, and the assignment's *"Multi-document Awareness"* requirement actually hold up.

Lesson: AI-generated retrieval code looks correct on the happy path but quietly misses semantic-coverage failures. You only catch them by writing a test for the contract you actually care about.

### Where I leaned on AI vs intervened

**Leaned on AI for:**
- Boilerplate and module scaffolding
- Wiring components together (FastMCP tool decorators, ChromaDB client setup)
- Standard RAG-pipeline patterns

**Intervened and corrected:**
- **Multi-document retrieval** (above) — replaced naive global top-k with a per-doc + global hybrid
- **Hallucination control** — added the distance threshold, `temperature=0`, JSON-locked output, and the "soft" cross-source synthesis rule (the original hard "do not combine across sources" rule was suppressing legitimate comparison answers)
- **`--force` ingestion** — Claude's first cut just re-ran upsert; I noticed stale chunks would never be removed, and added an explicit `reset_collection()` step
- **ChromaDB batch limit** — caught the 5461 hard cap and added batched upserts of 2000
- **Lazy client init** — moved `OpenAI(api_key=...)` out of module-level so a missing `.env` produces a friendly error instead of a `KeyError` at import

### Overall view on AI in software engineering

AI accelerates scaffolding and boilerplate dramatically, but it consistently produces code that works on the happy path and silently fails on the edge cases that actually define correctness. In this project the wins were architectural recall and code generation speed; the losses were every time I trusted the output without writing a test for the contract.

The reliable pattern: use AI to draft, then write the asserts yourself. Asserts are where understanding lives — they force you to articulate the behaviour the system is supposed to have, which is exactly the part the model can't substitute for.

---

## Project Structure

```
MCP_NEXLA/
├── data/pdfs/ # Drop PDFs here (subdirectories supported; *_qa.jsonl files are ignored)
├── chroma_db/ # Auto-created by ingestion, gitignored
├── src/
│ ├── ingestion/
│ │ ├── pdf_parser.py # PDF → page records (PyMuPDF)
│ │ └── chunker.py # Page records → chunks with metadata
│ ├── embeddings/
│ │ └── embedder.py # Text → vectors (sentence-transformers)
│ ├── vector_store/
│ │ └── store.py # ChromaDB wrapper (add, query, query_within_doc, reset, list)
│ ├── retrieval/
│ │ └── retriever.py # Question → per-doc + global retrieval, filtered & merged
│ ├── llm/
│ │ └── synthesizer.py # Chunks + question → grounded answer (GPT-4o-mini)
│ └── mcp_server/
│ └── server.py # MCP tool definitions + startup check
├── scripts/
│ ├── ingest.py # Ingestion CLI (--force to clean-rebuild)
│ └── demo.py # Human-readable demo runner
├── tests/
│ └── test_query.py # Assert-based test suite; --question flag for CLI mode
├── EXAMPLES.md # Full example interaction log (real outputs)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```