"""
llm/synthesizer.py

Responsibility: Build a grounded prompt from retrieved chunks, call the
LLM (OpenAI GPT-4o-mini), and parse the response into a structured answer
with sources.

Prompt design principles:
  1. Context-first layout — chunks are shown before the question so the
     model reads evidence before seeing what it needs to answer.
  2. Hard "only use the context" instruction — prevents the model from
     answering from its own training data (reduces hallucination risk).
  3. Cross-source synthesis is allowed (and encouraged) when the question
     spans multiple documents — but inferring beyond the text is not.
  4. Explicit "not found" fallback — the model is told what to say when
     the context is insufficient, giving callers a detectable signal.
  5. Structured JSON output — parsed reliably via OpenAI's
     `response_format=json_object`, no fence-stripping needed.
  6. Snippet in sources — each source includes a short quote from the
     chunk so the caller can verify the attribution without re-querying.
"""

import json
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# gpt-4o-mini: fast and cheap for RAG synthesis — the heavy lifting is
# retrieval, not generation. Swap to gpt-4o for more nuanced answers.
_MODEL = "gpt-4o-mini"

# Hard cap on how many chunks reach the LLM. Retriever may return more;
# we trim here to keep the prompt size predictable. 8 covers cross-document
# questions while staying well within gpt-4o-mini's context window.
_MAX_CONTEXT_CHUNKS = 8

# Lazy singleton — instantiated on first use so missing API keys produce
# a clear error at call time rather than at import time.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        # max_retries handles transient 5xx / rate-limit blips automatically.
        _client = OpenAI(api_key=api_key, max_retries=2)
    return _client


def _build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Assemble the context block + question into a single user-turn prompt.

    Each chunk is labelled with its source so the model can cite it directly.
    Format:
        [Source 1] annual_report — Page 4
        <text>

        [Source 2] ...

        Question: ...
    """
    context_lines = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Source {i}] {chunk['doc_name']} — Page {chunk['page_num']}"
        context_lines.append(f"{header}\n{chunk['text']}")

    context_block = "\n\n".join(context_lines)
    return f"{context_block}\n\nQuestion: {question}"


_SYSTEM_PROMPT = """\
You are a document assistant. Answer questions strictly based on the numbered sources provided.

Rules:
- Use ONLY the information present in the provided sources. Do not use external or training-data knowledge.
- You MAY synthesize across multiple sources — for comparison, contrast, or aggregation questions, draw on every relevant source and cite each one. Do not fabricate information not present in the text.
- If only some sources are relevant to the question, use only those.
- If the answer cannot be found in any source, respond with exactly:
  {"answer": "Not found in documents.", "sources": []}
- If only partial information is found, state clearly what is known and flag what is missing — do not guess the rest.
- Always cite the sources you used.
- Be concise and factual.

Respond in valid JSON with this exact structure:
{
  "answer": "<your answer here>",
  "sources": [
    {"doc": "<doc_name>", "page": <page_num>, "snippet": "<short quote from that source>"}
  ]
}
"""


def synthesize(question: str, chunks: list[dict]) -> dict:
    """
    Generate a grounded answer from the question and retrieved chunks.

    Returns a dict with:
        "answer"  — the answer string
        "sources" — list of {"doc", "page", "snippet"}

    Falls back to a safe error dict if the LLM response cannot be parsed.
    """
    if not chunks:
        logger.warning("No chunks provided to synthesizer — returning not-found response.")
        return {"answer": "Not found in documents.", "sources": []}

    trimmed = chunks[:_MAX_CONTEXT_CHUNKS]
    logger.info(
        "Synthesizing answer from %d chunk(s) across %d doc(s).",
        len(trimmed),
        len({c["doc_name"] for c in trimmed}),
    )

    user_prompt = _build_prompt(question, trimmed)

    response = _get_client().chat.completions.create(
        model=_MODEL,
        max_tokens=1024,
        temperature=0,  # deterministic — RAG is retrieval, not creative writing
        response_format={"type": "json_object"},  # API-level JSON guarantee
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_text = response.choices[0].message.content.strip()
    logger.info("LLM response received (%d chars).", len(raw_text))
    return _parse_response(raw_text)


def _parse_response(raw_text: str) -> dict:
    """
    Parse the LLM's JSON response into a dict.

    `response_format=json_object` guarantees valid JSON, so this is mostly
    a safety net. If parsing fails, return the raw text as the answer
    rather than swallowing the error silently.
    """
    try:
        parsed = json.loads(raw_text)
        parsed.setdefault("answer", "No answer returned.")
        parsed.setdefault("sources", [])
        return parsed
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON. Raw: %r", raw_text)
        return {"answer": raw_text, "sources": []}
