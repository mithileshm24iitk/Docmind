"""Shared test setup: temp Chroma dir, no real API key, fake embeddings."""

import hashlib
import math

import pytest
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.retrieval import bm25, vectorstore


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings, so no network calls are made.

    Each word is hashed into one of `dim` buckets. Texts sharing words get
    similar vectors, so similarity search still behaves sensibly in tests.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in text.lower().split():
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point Chroma at a temp folder and blank the API key (env beats .env).

    Also empty the in-memory BM25 cache, since every test gets a fresh Chroma dir.
    """
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    get_settings.cache_clear()
    bm25.clear_cache()
    yield
    get_settings.cache_clear()
    bm25.clear_cache()


@pytest.fixture
def fake_embeddings(monkeypatch) -> FakeEmbeddings:
    """Make the vector store use FakeEmbeddings instead of Gemini."""
    fake = FakeEmbeddings()
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: fake)
    return fake
