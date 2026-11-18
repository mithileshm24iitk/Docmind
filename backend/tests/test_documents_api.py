import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

TEXT = (
    "Photosynthesis lets plants turn sunlight into energy.\n\n"
    "The Eiffel Tower is in Paris, France."
)


def upload(name: str = "notes.txt", body: str = TEXT):
    return client.post(
        "/documents/upload", files={"file": (name, body.encode(), "text/plain")}
    )


def test_preview_still_works():
    r = client.post(
        "/documents/preview", files={"file": ("notes.txt", TEXT.encode(), "text/plain")}
    )
    assert r.status_code == 200
    assert r.json()["num_chunks"] >= 1


def test_upload_list_search(fake_embeddings):
    r = upload()
    assert r.status_code == 200
    assert r.json() == {"filename": "notes.txt", "pages": 1, "num_chunks": 1}

    r = client.get("/documents")
    assert r.json() == {"documents": [{"filename": "notes.txt", "num_chunks": 1}]}

    r = client.post("/documents/search", json={"query": "Eiffel Tower", "k": 3})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert results[0]["metadata"] == {"source": "notes.txt", "page": 1, "chunk_id": 0}
    assert "score" in results[0]


def test_upload_rejects_bad_type(fake_embeddings):
    r = upload(name="image.png")
    assert r.status_code == 400


def test_search_validates_input(fake_embeddings):
    assert client.post("/documents/search", json={"query": "", "k": 3}).status_code == 422
    assert client.post("/documents/search", json={"query": "x", "k": 0}).status_code == 422


def test_search_default_mode_is_hybrid(fake_embeddings):
    upload()
    r = client.post("/documents/search", json={"query": "Eiffel Tower", "k": 3})
    assert r.json()["mode"] == "hybrid"


def test_search_hybrid_mode_shows_ranks(fake_embeddings):
    upload()
    r = client.post(
        "/documents/search", json={"query": "Eiffel", "k": 3, "mode": "hybrid"}
    )
    assert r.status_code == 200
    result = r.json()["results"][0]
    assert result["vector_rank"] == 1
    assert result["bm25_rank"] == 1
    assert result["score"] == pytest.approx(2 / 61)


def test_search_bm25_mode(fake_embeddings):
    upload()
    r = client.post("/documents/search", json={"query": "Eiffel", "mode": "bm25"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1 and results[0]["bm25_rank"] == 1


def test_search_vector_mode_keeps_old_shape(fake_embeddings):
    upload()
    r = client.post("/documents/search", json={"query": "Eiffel", "mode": "vector"})
    result = r.json()["results"][0]
    assert set(result) == {"text", "score", "metadata"}


def test_search_empty_collection_all_modes(fake_embeddings):
    for mode in ["vector", "bm25", "hybrid"]:
        r = client.post("/documents/search", json={"query": "x", "mode": mode})
        assert r.status_code == 200
        assert r.json()["results"] == []


def test_search_rejects_unknown_mode(fake_embeddings):
    r = client.post("/documents/search", json={"query": "x", "mode": "magic"})
    assert r.status_code == 422
