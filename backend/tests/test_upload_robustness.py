"""Phase 10: batched embedding with backoff, clean upload errors, limits, and DELETE /documents."""

import io

import pytest
from fastapi.testclient import TestClient
from google.genai.errors import ClientError, ServerError
from langchain_core.documents import Document
from pypdf import PdfWriter

from app.api import documents
from app.main import app
from app.retrieval import bm25, vectorstore
from app.retrieval.vectorstore import EmbeddingError
from tests.conftest import FakeEmbeddings

client = TestClient(app)
COL = "default"


def gemini_error(code: int) -> Exception:
    """The same exception types google-genai raises for a real 429 / 503."""
    body = {"error": {"code": code, "status": "X", "message": "fake"}}
    return ClientError(code, body) if code < 500 else ServerError(code, body)


class FlakyEmbeddings(FakeEmbeddings):
    """FakeEmbeddings whose embed_documents fails the first `fail_times` calls."""

    def __init__(self, fail_times: int = 0, error: Exception | None = None) -> None:
        super().__init__()
        self.fail_times = fail_times
        self.error = error
        self.batch_sizes: list[int] = []  # size of every embed_documents call, failed or not

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        return super().embed_documents(texts)


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Record backoff delays instead of really sleeping."""
    calls: list[float] = []
    monkeypatch.setattr(vectorstore, "_sleep", calls.append)
    return calls


def use_embeddings(monkeypatch, emb: FakeEmbeddings) -> FakeEmbeddings:
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: emb)
    return emb


def make_chunks(n: int, source: str = "big.txt") -> list[Document]:
    return [
        Document(page_content=f"chunk number {i} text", metadata={"source": source, "page": 1, "chunk_id": i})
        for i in range(n)
    ]


def stored_count() -> int:
    return sum(d["num_chunks"] for d in vectorstore.list_documents(COL))


def pdf_bytes(pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------- batching + backoff (vectorstore.add_chunks) ----------

def test_chunks_are_embedded_in_batches_of_50(monkeypatch, sleeps):
    emb = use_embeddings(monkeypatch, FlakyEmbeddings())
    assert vectorstore.add_chunks(COL, make_chunks(120)) == 120
    assert emb.batch_sizes == [50, 50, 20]
    assert stored_count() == 120
    assert sleeps == []


@pytest.mark.parametrize("code", [429, 503])
def test_transient_error_is_retried_with_exponential_backoff(monkeypatch, sleeps, code):
    emb = use_embeddings(monkeypatch, FlakyEmbeddings(fail_times=2, error=gemini_error(code)))
    assert vectorstore.add_chunks(COL, make_chunks(3)) == 3
    assert sleeps == [1.0, 2.0]
    assert len(emb.batch_sizes) == 3  # 2 failures + 1 success
    assert stored_count() == 3


@pytest.mark.parametrize("code", [429, 503])
def test_gives_up_after_5_tries_with_clean_error(monkeypatch, sleeps, code):
    emb = use_embeddings(monkeypatch, FlakyEmbeddings(fail_times=99, error=gemini_error(code)))
    with pytest.raises(EmbeddingError) as info:
        vectorstore.add_chunks(COL, make_chunks(3))
    assert info.value.status_code == code
    assert "fake" not in info.value.message  # the raw Gemini message never leaks
    assert len(emb.batch_sizes) == 5
    assert sleeps == [1.0, 2.0, 4.0, 8.0]


def test_non_transient_error_is_not_retried(monkeypatch, sleeps):
    emb = use_embeddings(monkeypatch, FlakyEmbeddings(fail_times=99, error=gemini_error(400)))
    with pytest.raises(ClientError):
        vectorstore.add_chunks(COL, make_chunks(3))
    assert len(emb.batch_sizes) == 1
    assert sleeps == []


def test_failed_upload_rolls_back_earlier_batches(monkeypatch, sleeps):
    emb = FlakyEmbeddings()
    use_embeddings(monkeypatch, emb)
    original = emb.embed_documents
    calls = {"n": 0}

    def first_batch_ok_then_429(texts):
        calls["n"] += 1
        if calls["n"] == 1:
            return original(texts)
        raise gemini_error(429)

    monkeypatch.setattr(emb, "embed_documents", first_batch_ok_then_429)
    with pytest.raises(EmbeddingError):
        vectorstore.add_chunks(COL, make_chunks(80))
    assert stored_count() == 0  # batch 1 was removed again


# ---------- /documents/upload: clean errors and limits ----------

@pytest.mark.parametrize("code", [429, 503])
def test_upload_returns_clean_json_error_on_final_failure(monkeypatch, sleeps, code):
    use_embeddings(monkeypatch, FlakyEmbeddings(fail_times=99, error=gemini_error(code)))
    r = client.post("/documents/upload", files={"file": ("n.txt", b"Plants use sunlight.", "text/plain")})
    assert r.status_code == code
    assert set(r.json()) == {"detail"}
    assert "Traceback" not in r.text and "fake" not in r.text


def test_upload_unexpected_embedding_error_is_503_not_500(monkeypatch):
    def broken():
        raise RuntimeError("GOOGLE_API_KEY is not set")

    monkeypatch.setattr(vectorstore, "get_embeddings", broken)
    r = client.post("/documents/upload", files={"file": ("n.txt", b"Plants use sunlight.", "text/plain")})
    assert r.status_code == 503
    assert "GOOGLE_API_KEY" not in r.text


def test_upload_rejects_too_many_pages(monkeypatch, fake_embeddings):
    monkeypatch.setattr(documents, "MAX_PAGES", 2)
    r = client.post("/documents/upload", files={"file": ("long.pdf", pdf_bytes(3), "application/pdf")})
    assert r.status_code == 413
    assert r.json()["detail"] == "Document has 3 pages; the limit is 2 pages."


def test_default_page_limit_is_300():
    assert documents.MAX_PAGES == 300
    assert documents.MAX_MB == 20


def test_upload_rejects_too_large_file(monkeypatch, fake_embeddings):
    monkeypatch.setattr(documents, "MAX_MB", 1)
    r = client.post("/documents/upload", files={"file": ("big.txt", b"a" * (1024 * 1024 + 1), "text/plain")})
    assert r.status_code == 413


def test_upload_corrupt_pdf_is_422_not_500(fake_embeddings):
    r = client.post("/documents/upload", files={"file": ("bad.pdf", b"not really a pdf", "application/pdf")})
    assert r.status_code == 422


# ---------- DELETE /documents/{filename} ----------

def upload(name: str, body: bytes):
    r = client.post("/documents/upload", files={"file": (name, body, "text/plain")})
    assert r.status_code == 200
    return r


def test_delete_document_removes_chunks_and_refreshes_bm25(fake_embeddings):
    upload("plants.txt", b"Photosynthesis lets plants turn sunlight into energy.")
    upload("paris.txt", b"The Eiffel Tower is in Paris.")
    assert bm25.bm25_search(COL, "Eiffel")  # warms the BM25 cache with both files

    r = client.delete("/documents/paris.txt")
    assert r.status_code == 200
    assert r.json() == {"filename": "paris.txt", "deleted_chunks": 1}

    assert client.get("/documents").json() == {
        "documents": [{"filename": "plants.txt", "num_chunks": 1}]
    }
    assert bm25.bm25_search(COL, "Eiffel") == []  # stale cache was not used
    r = client.post("/documents/search", json={"query": "Eiffel Tower", "mode": "hybrid"})
    assert all(res["metadata"]["source"] != "paris.txt" for res in r.json()["results"])


def test_delete_unknown_document_is_404(fake_embeddings):
    upload("plants.txt", b"Photosynthesis.")
    r = client.delete("/documents/nope.txt")
    assert r.status_code == 404
    assert client.delete("/documents/plants.txt").status_code == 200
    assert client.delete("/documents/plants.txt").status_code == 404  # already gone


def test_delete_requires_auth(fake_embeddings):
    app.dependency_overrides.clear()  # use the real get_current_user
    assert client.delete("/documents/plants.txt").status_code == 401
