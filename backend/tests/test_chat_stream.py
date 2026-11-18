"""Tests for POST /chat/stream SSE endpoint and DELETE /chat/sessions/{session_id}."""

import json

from fastapi.testclient import TestClient

from app.main import app
from app.rag.memory import get_memory_store
from app.rag.prompts import NOT_FOUND_ANSWER

client = TestClient(app)

TEXT = "Photosynthesis lets plants turn sunlight into chemical energy."


def upload() -> None:
    r = client.post(
        "/documents/upload", files={"file": ("notes.txt", TEXT.encode(), "text/plain")}
    )
    assert r.status_code == 200


def parse_sse(raw_body: str) -> list[dict]:
    """Parse SSE `data: <json>\\n\\n` frames into dicts."""
    events: list[dict] = []
    for block in raw_body.split("\n\n"):
        line = block.strip()
        if line.startswith("data: "):
            events.append(json.loads(line.removeprefix("data: ")))
    return events


def test_stream_emits_tokens_and_done_and_saves_memory(fake_embeddings, fake_llm):
    upload()
    fake_llm.reply = "Plants use sunlight [notes.txt, p.1]."

    r = client.post(
        "/chat/stream",
        json={
            "session_id": "s-stream",
            "question": "how do plants turn sunlight into energy",
            "k": 3,
        },
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(r.text)
    token_events = [e for e in events if e["type"] == "token"]
    done_events = [e for e in events if e["type"] == "done"]

    assert len(token_events) >= 2
    assert "".join(e["content"] for e in token_events) == fake_llm.reply
    assert len(done_events) == 1

    done = done_events[0]
    assert done["grounded"] is True
    assert done["sources"][0]["filename"] == "notes.txt"
    assert done["sources"][0]["page"] == 1
    assert done["rewritten_question"] == "how do plants turn sunlight into energy"
    assert set(done["stats"]) == {
        "retrieval_ms",
        "generation_ms",
        "input_tokens",
        "output_tokens",
    }

    history = get_memory_store().get_history("s-stream")
    assert history == [
        {
            "user": "how do plants turn sunlight into energy",
            "assistant": fake_llm.reply,
        }
    ]


def test_stream_follow_up_rewrites_question(fake_embeddings, fake_llm):
    upload()
    get_memory_store().add_turn("s-follow", "tell me about plants", "Plants use sunlight.")
    fake_llm.replies = [
        "how do plants turn sunlight into chemical energy",
        "They use photosynthesis [notes.txt, p.1].",
    ]

    r = client.post(
        "/chat/stream",
        json={"session_id": "s-follow", "question": "how do they do it?"},
    )
    assert r.status_code == 200
    events = parse_sse(r.text)
    done = [e for e in events if e["type"] == "done"][0]

    assert done["grounded"] is True
    assert done["rewritten_question"] == "how do plants turn sunlight into chemical energy"
    assert fake_llm.call_count == 2  # 1 condense invoke + 1 generation stream


def test_stream_relevance_gate_blocks_and_skips_llm(fake_embeddings, fake_llm):
    upload()
    r = client.post(
        "/chat/stream",
        json={
            "session_id": "s-gate",
            "question": "quantum chromodynamics gluon confinement",
        },
    )
    assert r.status_code == 200
    events = parse_sse(r.text)

    assert events[0] == {"type": "token", "content": NOT_FOUND_ANSWER}
    assert events[1]["type"] == "done"
    assert events[1]["grounded"] is False
    assert events[1]["sources"] == []
    assert fake_llm.call_count == 0


def test_delete_session_clears_history(fake_embeddings, fake_llm):
    store = get_memory_store()
    store.add_turn("s-del", "hello", "world")
    assert len(store.get_history("s-del")) == 1

    r = client.delete("/chat/sessions/s-del")
    assert r.status_code == 200
    assert r.json() == {"session_id": "s-del", "cleared": True}
    assert store.get_history("s-del") == []
