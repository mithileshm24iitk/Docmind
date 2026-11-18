"""RAG chain tests. Fake embeddings + fake LLM, so no real API calls are made."""

import httpx
import pytest
from google.genai.errors import ClientError
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.rag import chain
from app.rag.chain import LLMError, answer
from app.rag.prompts import NOT_FOUND_ANSWER, SYSTEM_PROMPT
from app.retrieval import vectorstore

COL = "default"
GROUNDED_Q = "how does photosynthesis turn sunlight into energy"
FAR_Q = "quantum chromodynamics gluon confinement"


def add_corpus() -> None:
    chunks = [
        Document(
            page_content="photosynthesis lets plants turn sunlight into chemical energy",
            metadata={"source": "biology.pdf", "page": 2, "chunk_id": 0},
        ),
        Document(
            page_content="the eiffel tower was completed in paris in 1889",
            metadata={"source": "history.pdf", "page": 7, "chunk_id": 3},
        ),
    ]
    vectorstore.add_chunks(COL, chunks)


# ---------- grounded path ----------

def test_grounded_answer_and_sources(fake_embeddings, fake_llm):
    add_corpus()
    result = answer(GROUNDED_Q, k=2)

    assert set(result) == {"answer", "sources", "best_distance", "grounded", "stats"}
    assert result["grounded"] is True
    assert result["answer"] == fake_llm.reply
    assert fake_llm.call_count == 1
    assert result["best_distance"] <= 0.6

    # Sources are exactly the chunks hybrid retrieval returned (best first).
    first = result["sources"][0]
    assert first == {
        "filename": "biology.pdf",
        "page": 2,
        "chunk_id": 0,
        "snippet": "photosynthesis lets plants turn sunlight into chemical energy",
    }
    assert {s["filename"] for s in result["sources"]} <= {"biology.pdf", "history.pdf"}


def test_prompt_contains_labelled_context_and_question(fake_embeddings, fake_llm):
    add_corpus()
    answer(GROUNDED_Q, k=2)

    system, user = fake_llm.calls[0]
    assert isinstance(system, SystemMessage) and system.content == SYSTEM_PROMPT
    assert isinstance(user, HumanMessage)
    assert "[biology.pdf, p.2]" in user.content
    assert "photosynthesis lets plants turn sunlight" in user.content
    assert GROUNDED_Q in user.content
    assert "<context>" in user.content


def test_stats_fields(fake_embeddings, fake_llm):
    add_corpus()
    stats = answer(GROUNDED_Q)["stats"]
    assert set(stats) == {"retrieval_ms", "generation_ms", "input_tokens", "output_tokens"}
    assert stats["retrieval_ms"] >= 0 and stats["generation_ms"] >= 0
    assert stats["input_tokens"] == 120 and stats["output_tokens"] == 15


def test_stats_tokens_null_when_api_gives_none(fake_embeddings, fake_llm, monkeypatch):
    add_corpus()
    monkeypatch.setattr(chain, "_generate", lambda messages: ("ok", None))
    stats = answer(GROUNDED_Q)["stats"]
    assert stats["input_tokens"] is None and stats["output_tokens"] is None


# ---------- relevance gate ----------

def assert_blocked(result: dict) -> None:
    assert result["grounded"] is False
    assert result["answer"] == NOT_FOUND_ANSWER
    assert result["sources"] == []


def test_gate_empty_collection_skips_llm(fake_embeddings, fake_llm):
    result = answer(GROUNDED_Q)
    assert_blocked(result)
    assert result["best_distance"] is None
    assert result["stats"]["generation_ms"] is None
    assert fake_llm.call_count == 0


def test_gate_far_away_query_skips_llm(fake_embeddings, fake_llm):
    add_corpus()
    result = answer(FAR_Q)
    assert_blocked(result)
    assert result["best_distance"] > 0.6
    assert fake_llm.call_count == 0


def test_gate_threshold_comes_from_config(fake_embeddings, fake_llm, monkeypatch):
    add_corpus()
    monkeypatch.setenv("RELEVANCE_MAX_DISTANCE", "0.0001")
    chain.get_settings.cache_clear()
    assert_blocked(answer(GROUNDED_Q))
    assert fake_llm.call_count == 0


# ---------- LLM errors become clean LLMErrors ----------

@pytest.mark.parametrize(
    "error, status",
    [
        (ClientError(429, {"error": {"code": 429, "message": "quota key=AIzaSECRET"}}), 429),
        (ClientError(401, {"error": {"code": 401, "message": "bad key AIzaSECRET"}}), 502),
        (httpx.ReadTimeout("timed out"), 504),
        (RuntimeError("something odd AIzaSECRET"), 502),
    ],
)
def test_llm_errors_are_wrapped(fake_embeddings, fake_llm, error, status):
    add_corpus()
    fake_llm.error = error
    with pytest.raises(LLMError) as info:
        answer(GROUNDED_Q)
    assert info.value.status_code == status
    assert "AIza" not in info.value.message
    assert "Traceback" not in info.value.message


def test_missing_api_key_is_clean_error(fake_embeddings):
    # No fake_llm: the real get_llm runs, and conftest blanks GOOGLE_API_KEY.
    add_corpus()
    with pytest.raises(LLMError) as info:
        answer(GROUNDED_Q)
    assert info.value.status_code == 503
