"""Tests for POST /chat/stream SSE endpoint and DELETE /chat/sessions/{session_id}."""

import json

from fastapi.testclient import TestClient

from app.main import app
from app.rag.memory import get_memory_store
from tests.conftest import TEST_USER_ID
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

    history = get_memory_store().get_history("s-stream", user_id=TEST_USER_ID)
    assert history == [
        {
            "user": "how do plants turn sunlight into energy",
            "assistant": fake_llm.reply,
        }
    ]


def test_stream_follow_up_rewrites_question(fake_embeddings, fake_llm):
    upload()
    get_memory_store().add_turn(
        "s-follow", "tell me about plants", "Plants use sunlight.", user_id=TEST_USER_ID
    )
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
    store.add_turn("s-del", "hello", "world", user_id=TEST_USER_ID)
    assert len(store.get_history("s-del", user_id=TEST_USER_ID)) == 1

    r = client.delete("/chat/sessions/s-del")
    assert r.status_code == 200
    assert r.json() == {"session_id": "s-del", "cleared": True}
    assert store.get_history("s-del", user_id=TEST_USER_ID) == []


def test_stream_llm_error_yields_error_sse_event(fake_embeddings, fake_llm):
    from google.genai.errors import ClientError

    upload()
    fake_llm.error = ClientError(503, {"error": {"code": 503, "message": "key AIzaSECRET"}})

    r = client.post(
        "/chat/stream",
        json={
            "session_id": "s-err",
            "question": "how do plants turn sunlight into energy",
        },
    )
    assert r.status_code == 200
    events = parse_sse(r.text)
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["status_code"] == 503
    assert "AIza" not in error_events[0]["detail"]
    assert error_events[0]["detail"]


def test_stream_fallback_on_503_succeeds(fake_embeddings, monkeypatch, caplog):
    from google.genai.errors import ClientError
    from langchain_core.messages import AIMessage
    from app.rag import chain

    upload()
    models_called: list[str] = []

    class StreamConditionalLLM:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def stream(self, messages: list):
            models_called.append(self.model_name)
            if self.model_name == "gemini-3.8-flash":
                raise ClientError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}})
            yield AIMessage(
                content="Fallback stream [notes.txt, p.1].",
                usage_metadata={"input_tokens": 30, "output_tokens": 5, "total_tokens": 35},
            )

    monkeypatch.setattr(
        chain, "get_llm", lambda model=None: StreamConditionalLLM(model or "gemini-3.8-flash")
    )

    r = client.post(
        "/chat/stream",
        json={
            "session_id": "s-fallback",
            "question": "how do plants turn sunlight into energy",
        },
    )
    assert r.status_code == 200
    events = parse_sse(r.text)
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == "Fallback stream [notes.txt, p.1]."
    assert models_called == ["gemini-3.8-flash", "gemini-2.5-flash"]
    assert any("gemini-2.5-flash" in rec.message for rec in caplog.records)



# ---------- chat memory is saved under the signed-in user's own id ----------

def _fk_violation():
    from postgrest.exceptions import APIError

    return APIError(
        {
            "code": "23503",
            "message": 'violates foreign key constraint "chat_messages_user_id_fkey"',
            "details": None,
            "hint": None,
        }
    )


def test_stream_saves_turn_under_authenticated_user_id(fake_embeddings, fake_llm, fake_supabase):
    """Real token check, no override: the row must carry the token's user id, never "anonymous"."""
    from app.api.auth import create_access_token

    upload()
    app.dependency_overrides.clear()  # use the real get_current_user from here on
    token = create_access_token({"id": "user-carol", "email": "carol@example.com", "name": "Carol"})
    headers = {"Authorization": f"Bearer {token}"}

    r = client.post(
        "/chat/stream",
        json={"session_id": "s-carol", "question": "how do plants turn sunlight into energy"},
        headers=headers,
    )
    assert r.status_code == 200
    assert parse_sse(r.text)[-1]["type"] == "done"

    rows = fake_supabase.tables["chat_messages"]
    assert [row["user_id"] for row in rows] == ["user-carol"]
    assert get_memory_store().get_history("s-carol", user_id="user-carol")[0]["assistant"] == fake_llm.reply

    # /chat/ask uses the same id
    r = client.post("/chat/ask", json={"question": "plants sunlight", "session_id": "s-carol"}, headers=headers)
    assert r.status_code == 200
    assert {row["user_id"] for row in fake_supabase.tables["chat_messages"]} == {"user-carol"}


def test_stream_without_valid_token_is_401_and_saves_nothing(fake_embeddings, fake_llm, fake_supabase):
    upload()
    app.dependency_overrides.clear()
    body = {"session_id": "s-x", "question": "plants sunlight"}
    assert client.post("/chat/stream", json=body).status_code == 401
    assert client.post("/chat/stream", json=body, headers={"Authorization": "Bearer junk"}).status_code == 401
    assert client.post("/chat/ask", json=body).status_code == 401
    assert fake_supabase.tables.get("chat_messages", []) == []
    assert fake_llm.call_count == 0


def test_failing_memory_save_does_not_break_the_stream(fake_embeddings, fake_llm, monkeypatch, caplog):
    upload()
    store = get_memory_store()

    def broken_add_turn(*args, **kwargs):
        raise _fk_violation()

    monkeypatch.setattr(store, "add_turn", broken_add_turn)
    fake_llm.reply = "Plants use sunlight [notes.txt, p.1]."

    r = client.post(
        "/chat/stream",
        json={"session_id": "s-broken", "question": "how do plants turn sunlight into energy"},
    )
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert "".join(e["content"] for e in events if e["type"] == "token") == fake_llm.reply
    assert events[-1]["type"] == "done"
    assert events[-1]["grounded"] is True
    assert not [e for e in events if e["type"] == "error"]

    # Logged with type and code only, not the database message.
    assert "Saving chat memory failed: APIError (code=23503)" in caplog.text
    assert "foreign key" not in caplog.text


def test_failing_memory_save_on_gate_block_still_sends_done(fake_embeddings, fake_llm, monkeypatch):
    upload()
    monkeypatch.setattr(get_memory_store(), "add_turn", lambda *a, **k: (_ for _ in ()).throw(_fk_violation()))
    r = client.post(
        "/chat/stream",
        json={"session_id": "s-gate2", "question": "quantum chromodynamics gluon confinement"},
    )
    events = parse_sse(r.text)
    assert [e["type"] for e in events] == ["token", "done"]
    assert events[-1]["grounded"] is False


def test_failing_memory_save_does_not_break_ask(fake_embeddings, fake_llm, monkeypatch):
    upload()

    def broken_add_turn(*args, **kwargs):
        raise _fk_violation()

    monkeypatch.setattr(get_memory_store(), "add_turn", broken_add_turn)
    r = client.post("/chat/ask", json={"question": "plants sunlight", "session_id": "s-ask"})
    assert r.status_code == 200
    assert r.json()["grounded"] is True
