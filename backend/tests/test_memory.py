"""Tests for conversation memory, truncation, and follow-up question rewriting."""

from langchain_core.documents import Document

from app.config import get_settings
from app.rag.chain import answer, rewrite_question
from app.rag.memory import BaseMemoryStore, InMemorySessionStore, get_memory_store
from app.retrieval import vectorstore


def test_store_implements_base_interface():
    store = InMemorySessionStore()
    assert isinstance(store, BaseMemoryStore)


def test_memory_truncation_keeps_last_max_turns(monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_TURNS", "5")
    get_settings.cache_clear()
    store = InMemorySessionStore()

    for i in range(8):
        store.add_turn("s1", f"question {i}", f"answer {i}")

    history = store.get_history("s1")
    assert len(history) == 5
    assert [turn["user"] for turn in history] == [
        "question 3",
        "question 4",
        "question 5",
        "question 6",
        "question 7",
    ]


def test_sessions_are_isolated_and_clearable():
    store = InMemorySessionStore()
    store.add_turn("s1", "q1", "a1")
    store.add_turn("s2", "q2", "a2")

    store.clear("s1")
    assert store.get_history("s1") == []
    assert len(store.get_history("s2")) == 1


def test_rewrite_question_skipped_when_history_empty(fake_llm):
    q = "how does photosynthesis turn sunlight into energy"
    assert rewrite_question(q, []) == q
    assert fake_llm.call_count == 0


def test_rewrite_question_called_when_history_exists(fake_llm):
    fake_llm.reply = "how does photosynthesis turn sunlight into chemical energy"
    history = [{"user": "tell me about photosynthesis", "assistant": "Plants use sunlight."}]

    rewritten = rewrite_question("how does it turn sunlight into energy?", history)
    assert rewritten == "how does photosynthesis turn sunlight into chemical energy"
    assert fake_llm.call_count == 1
    user_msg = fake_llm.calls[0][1].content
    assert "tell me about photosynthesis" in user_msg
    assert "how does it turn sunlight into energy?" in user_msg


def test_answer_with_session_id_rewrites_and_stores_turns(fake_embeddings, fake_llm):
    vectorstore.add_chunks(
        "default",
        [
            Document(
                page_content="photosynthesis lets plants turn sunlight into chemical energy",
                metadata={"source": "biology.pdf", "page": 2, "chunk_id": 0},
            )
        ],
    )
    store = get_memory_store()

    # Turn 1: no prior history -> only 1 LLM call (generation, no rewrite).
    fake_llm.replies = ["Photosynthesis uses sunlight [biology.pdf, p.2]."]
    r1 = answer("how does photosynthesis turn sunlight into energy", session_id="sess-1")
    assert r1["grounded"] is True
    assert fake_llm.call_count == 1
    assert len(store.get_history("sess-1")) == 1

    # Turn 2: history exists -> 2 LLM calls (1 rewrite + 1 generation with history).
    fake_llm.replies = [
        "photosynthesis sunlight chemical energy",
        "It stores chemical energy [biology.pdf, p.2].",
    ]
    r2 = answer("what kind of energy does it create?", session_id="sess-1")
    assert r2["grounded"] is True
    assert r2["answer"] == "It stores chemical energy [biology.pdf, p.2]."
    assert fake_llm.call_count == 3
    assert "<chat_history>" in fake_llm.calls[2][1].content
    assert len(store.get_history("sess-1")) == 2
