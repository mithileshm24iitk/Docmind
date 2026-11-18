"""RAG chain: retrieve chunks, check relevance, ask Gemini, return or stream answer + sources.

Flow for answer(question, session_id) / stream_answer(question, session_id):
  1. If session_id has history, condense the follow-up question into a standalone
     search query (skipped when history is empty).
  2. Relevance gate: one vector search for the single closest chunk to the search
     query. If the collection is empty or that chunk is too far away (cosine distance
     above RELEVANCE_MAX_DISTANCE), return/stream a fixed "not found" answer.
  3. Hybrid retrieval (vector + BM25 fused with RRF) for the top k chunks.
  4. Build a prompt with each chunk labelled [filename, p.N] (plus chat history)
     and call or stream the LLM, then save the turn in session memory.
"""

from collections.abc import Iterator
import json
import logging
import time
from functools import lru_cache

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.rag.memory import get_memory_store
from app.rag.prompts import (
    CHUNK_TEMPLATE,
    CONDENSE_SYSTEM_PROMPT,
    NOT_FOUND_ANSWER,
    SYSTEM_PROMPT,
    build_condense_prompt,
    build_generation_prompt,
)
from app.retrieval import vectorstore
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vectorstore import DEFAULT_COLLECTION

logger = logging.getLogger(__name__)

LLM_TIMEOUT_S = 60
SNIPPET_CHARS = 200


class LLMError(Exception):
    """A clean, user-safe LLM failure: an HTTP status code plus a short message.

    The message never contains the original exception text, so no stack
    traces, request details or key material leak to the client.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@lru_cache
def get_llm() -> BaseChatModel:
    """Return the Gemini chat model (typed generically so tests can use a fake)."""
    settings = get_settings()
    if not settings.google_api_key:
        raise LLMError(503, "The AI service is not configured (missing API key).")
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.google_api_key,
        timeout=LLM_TIMEOUT_S,
        max_retries=2,
    )


# ---------- small helpers ----------

def best_distance(collection: str, question: str) -> float | None:
    """Cosine distance of the single closest chunk, or None if the collection is empty."""
    hits = vectorstore.similarity_search(collection, question, k=1)
    return float(hits[0][1]) if hits else None


def format_context(chunks: list[dict]) -> str:
    """Join chunks into one string, each labelled with the citation to use."""
    return "\n\n".join(
        CHUNK_TEMPLATE.format(
            filename=c["metadata"]["source"],
            page=c["metadata"]["page"],
            text=c["text"],
        )
        for c in chunks
    )


def build_messages(question: str, chunks: list[dict], history: list | None = None) -> list:
    """System prompt + user message (labelled context, optional history, and question)."""
    user = build_generation_prompt(format_context(chunks), question, history)
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]


def to_sources(chunks: list[dict]) -> list[dict]:
    """The chunks we actually sent to the LLM, in a frontend-friendly shape."""
    return [
        {
            "filename": c["metadata"]["source"],
            "page": c["metadata"]["page"],
            "chunk_id": c["metadata"]["chunk_id"],
            "snippet": c["text"][:SNIPPET_CHARS],
        }
        for c in chunks
    ]


def _status_code(exc: BaseException) -> int | None:
    """Find an HTTP status code on the exception or anything it was raised from.

    langchain-google-genai re-raises google-genai's APIError (which has `.code`)
    as its own error types, so we walk the __cause__/__context__ chain.
    """
    seen: BaseException | None = exc
    while seen is not None:
        code = getattr(seen, "code", None)
        if isinstance(code, int):
            return code
        seen = seen.__cause__ or seen.__context__
    return None


def _is_timeout(exc: BaseException) -> bool:
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, (TimeoutError, httpx.TimeoutException)):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def to_llm_error(exc: Exception) -> LLMError:
    """Map any LLM exception to a fixed, safe message. Never echoes str(exc)."""
    if isinstance(exc, LLMError):
        return exc
    code = _status_code(exc)
    if _is_timeout(exc):
        return LLMError(504, "The AI service timed out. Please try again.")
    if code == 429:
        return LLMError(429, "The AI service quota or rate limit was reached. Try again later.")
    if code in (401, 403):
        return LLMError(502, "The AI service rejected the server's credentials.")
    if code == 404:
        return LLMError(502, "The configured AI model was not found.")
    return LLMError(502, "The AI service failed to answer. Please try again.")


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _generate(messages: list) -> tuple[str, dict | None]:
    """Call the LLM; return (answer text, usage metadata or None)."""
    try:
        response = get_llm().invoke(messages)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a clean LLMError
        error = to_llm_error(exc)
        # Log only the type and status, never the message (it could echo request data).
        logger.warning(
            "LLM call failed: %s -> HTTP %s", type(exc).__name__, error.status_code
        )
        raise error from None
    return response.text, getattr(response, "usage_metadata", None)


def _stream_generate(messages: list) -> Iterator[tuple[str, dict | None]]:
    """Stream (token_text, usage_metadata) pairs from the LLM."""
    try:
        for chunk in get_llm().stream(messages):
            token = chunk.text if hasattr(chunk, "text") else str(chunk.content)
            usage = getattr(chunk, "usage_metadata", None)
            if token or usage:
                yield token, usage
    except Exception as exc:  # noqa: BLE001 - every failure becomes a clean LLMError
        error = to_llm_error(exc)
        logger.warning(
            "LLM stream failed: %s -> HTTP %s", type(exc).__name__, error.status_code
        )
        raise error from None


def _sse(payload: dict) -> str:
    """Format a payload dictionary as a Server-Sent Event frame."""
    return f"data: {json.dumps(payload)}\n\n"


def rewrite_question(question: str, history: list) -> str:
    """Rewrite a follow-up question into a standalone search query using chat history.

    If history is empty, returns the question unchanged without calling the LLM.
    """
    if not history:
        return question
    messages = [
        SystemMessage(content=CONDENSE_SYSTEM_PROMPT),
        HumanMessage(content=build_condense_prompt(question, history)),
    ]
    rewritten, _ = _generate(messages)
    return rewritten.strip() or question


# ---------- main entry points ----------

def answer(
    question: str,
    session_id: str | None = None,
    k: int | None = None,
    collection: str = DEFAULT_COLLECTION,
) -> dict:
    """Answer a question from the documents, using conversation memory when session_id is set.

    Returns: answer, sources, best_distance, grounded, stats.
    Raises LLMError if the LLM call fails.
    """
    settings = get_settings()
    k = k or settings.top_k
    store = get_memory_store()
    history = store.get_history(session_id) if session_id else []
    search_query = rewrite_question(question, history) if session_id else question

    stats: dict = {
        "retrieval_ms": None,
        "generation_ms": None,
        "input_tokens": None,
        "output_tokens": None,
    }

    start = time.perf_counter()
    distance = best_distance(collection, search_query)
    passed_gate = distance is not None and distance <= settings.relevance_max_distance
    chunks = hybrid_search(collection, search_query, k) if passed_gate else []
    stats["retrieval_ms"] = _ms_since(start)

    if not chunks:
        if session_id:
            store.add_turn(session_id, question, NOT_FOUND_ANSWER)
        logger.info("RAG gate blocked: best_distance=%s stats=%s", distance, stats)
        return {
            "answer": NOT_FOUND_ANSWER,
            "sources": [],
            "best_distance": distance,
            "grounded": False,
            "stats": stats,
        }

    start = time.perf_counter()
    text, usage = _generate(build_messages(question, chunks, history))
    stats["generation_ms"] = _ms_since(start)
    if usage:
        stats["input_tokens"] = usage.get("input_tokens")
        stats["output_tokens"] = usage.get("output_tokens")

    if session_id:
        store.add_turn(session_id, question, text)

    logger.info("RAG answered: best_distance=%.3f stats=%s", distance, stats)
    return {
        "answer": text,
        "sources": to_sources(chunks),
        "best_distance": distance,
        "grounded": True,
        "stats": stats,
    }


def stream_answer(
    question: str,
    session_id: str,
    k: int | None = None,
    collection: str = DEFAULT_COLLECTION,
) -> Iterator[str]:
    """Stream an answer as SSE events (`token` chunks followed by a final `done` event)."""
    settings = get_settings()
    k = k or settings.top_k
    store = get_memory_store()
    history = store.get_history(session_id) if session_id else []
    rewritten = rewrite_question(question, history)

    stats: dict = {
        "retrieval_ms": None,
        "generation_ms": None,
        "input_tokens": None,
        "output_tokens": None,
    }

    start = time.perf_counter()
    distance = best_distance(collection, rewritten)
    passed_gate = distance is not None and distance <= settings.relevance_max_distance
    chunks = hybrid_search(collection, rewritten, k) if passed_gate else []
    stats["retrieval_ms"] = _ms_since(start)

    if not chunks:
        if session_id:
            store.add_turn(session_id, question, NOT_FOUND_ANSWER)
        logger.info("RAG stream gate blocked: best_distance=%s stats=%s", distance, stats)
        yield _sse({"type": "token", "content": NOT_FOUND_ANSWER})
        yield _sse(
            {
                "type": "done",
                "sources": [],
                "best_distance": distance,
                "grounded": False,
                "rewritten_question": rewritten,
                "stats": stats,
            }
        )
        return

    start = time.perf_counter()
    collected: list[str] = []
    usage: dict | None = None
    for token, chunk_usage in _stream_generate(build_messages(question, chunks, history)):
        if chunk_usage:
            usage = chunk_usage
        if token:
            collected.append(token)
            yield _sse({"type": "token", "content": token})
    stats["generation_ms"] = _ms_since(start)
    if usage:
        stats["input_tokens"] = usage.get("input_tokens")
        stats["output_tokens"] = usage.get("output_tokens")

    full_answer = "".join(collected)
    if session_id:
        store.add_turn(session_id, question, full_answer)

    logger.info("RAG stream answered: best_distance=%.3f stats=%s", distance, stats)
    yield _sse(
        {
            "type": "done",
            "sources": to_sources(chunks),
            "best_distance": distance,
            "grounded": True,
            "rewritten_question": rewritten,
            "stats": stats,
        }
    )
