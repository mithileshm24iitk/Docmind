from fastapi.testclient import TestClient
from google.genai.errors import ClientError

from app.main import app

client = TestClient(app)

TEXT = "Photosynthesis lets plants turn sunlight into energy."


def upload() -> None:
    r = client.post(
        "/documents/upload", files={"file": ("notes.txt", TEXT.encode(), "text/plain")}
    )
    assert r.status_code == 200


def ask(body: dict):
    return client.post("/chat/ask", json=body)


def test_ask_grounded(fake_embeddings, fake_llm):
    upload()
    r = ask({"question": "how do plants turn sunlight into energy", "k": 3})
    assert r.status_code == 200
    data = r.json()
    assert data["grounded"] is True
    assert data["sources"][0]["filename"] == "notes.txt"
    assert data["sources"][0]["page"] == 1
    assert "retrieval_ms" in data["stats"]


def test_ask_empty_collection_not_grounded(fake_embeddings, fake_llm):
    r = ask({"question": "anything"})
    assert r.status_code == 200
    assert r.json()["grounded"] is False
    assert fake_llm.call_count == 0


def test_empty_question_422(fake_embeddings, fake_llm):
    assert ask({"question": ""}).status_code == 422
    assert ask({"question": "   "}).status_code == 422
    assert ask({}).status_code == 422


def test_too_long_question_422(fake_embeddings, fake_llm):
    assert ask({"question": "a" * 2001}).status_code == 422
    assert ask({"question": "hi", "k": 0}).status_code == 422


def test_llm_error_is_clean_http_error(fake_embeddings, fake_llm):
    upload()
    fake_llm.error = ClientError(429, {"error": {"code": 429, "message": "key AIzaSECRET"}})
    r = ask({"question": "how do plants turn sunlight into energy"})
    assert r.status_code == 429
    assert set(r.json()) == {"detail"}
    assert "AIza" not in r.text
