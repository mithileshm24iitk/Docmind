"""Shared test setup: temp Chroma dir, no real API key, fake embeddings and LLM."""

import hashlib
import math

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from app.config import get_settings
from app.rag import chain, memory
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
    memory.get_memory_store().clear_all()
    yield
    get_settings.cache_clear()
    bm25.clear_cache()
    memory.get_memory_store().clear_all()


@pytest.fixture
def fake_embeddings(monkeypatch) -> FakeEmbeddings:
    """Make the vector store use FakeEmbeddings instead of Gemini."""
    fake = FakeEmbeddings()
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: fake)
    return fake


class FakeLLM:
    """Stands in for ChatGoogleGenerativeAI. Records every call, never hits the network.

    Set `error` to make invoke() raise, to test error handling.
    """

    def __init__(self, reply: str = "Plants use sunlight [notes.txt, p.1].") -> None:
        self.reply = reply
        self.replies: list[str] = []
        self.calls: list[list] = []  # the messages passed to each invoke()/stream()
        self.error: Exception | None = None

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def _next_reply(self) -> str:
        return self.replies.pop(0) if self.replies else self.reply

    def invoke(self, messages: list) -> AIMessage:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return AIMessage(
            content=self._next_reply(),
            usage_metadata={"input_tokens": 120, "output_tokens": 15, "total_tokens": 135},
        )

    def stream(self, messages: list):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        words = self._next_reply().split(" ")
        for i, word in enumerate(words):
            piece = word if i == len(words) - 1 else word + " "
            usage = (
                {"input_tokens": 120, "output_tokens": 15, "total_tokens": 135}
                if i == len(words) - 1
                else None
            )
            yield AIMessage(content=piece, usage_metadata=usage)


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    """Make the RAG chain use FakeLLM instead of Gemini."""
    fake = FakeLLM()
    monkeypatch.setattr(chain, "get_llm", lambda: fake)
    return fake
