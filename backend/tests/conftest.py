"""Shared test setup: temp Chroma dir, no real API keys, fake embeddings, LLM, and Supabase."""

from datetime import datetime, timezone
import hashlib
import math
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from app import db
from app.api.auth import get_current_user
from app.config import get_settings
from app.main import app
from app.rag import chain, memory, tracing
from app.retrieval import bm25, vectorstore

# Fake 48-char signing key for tests only (never a real secret).
TEST_AUTH_SECRET = "test-auth-secret-not-real-0123456789abcdefghijkl"


class FakeSupabaseTable:
    """In-memory stand-in for a Supabase PostgREST table query builder."""

    def __init__(self, client: "FakeSupabaseClient", table_name: str) -> None:
        self._client = client
        self._table_name = table_name
        self._op: str = "select"
        self._payload: Any = None
        self._filters: list[tuple[str, Any]] = []
        self._order_col: str | None = None
        self._order_desc: bool = False
        self._limit_count: int | None = None

    @property
    def _rows(self) -> list[dict[str, Any]]:
        return self._client.tables.setdefault(self._table_name, [])

    def select(self, _columns: str = "*") -> "FakeSupabaseTable":
        self._op = "select"
        return self

    def insert(self, payload: dict[str, Any] | list[dict[str, Any]]) -> "FakeSupabaseTable":
        self._op = "insert"
        self._payload = payload
        return self

    def upsert(self, payload: dict[str, Any] | list[dict[str, Any]]) -> "FakeSupabaseTable":
        self._op = "upsert"
        self._payload = payload
        return self

    def delete(self) -> "FakeSupabaseTable":
        self._op = "delete"
        return self

    def eq(self, column: str, value: Any) -> "FakeSupabaseTable":
        self._filters.append((column, value))
        return self

    def order(self, column: str, *, desc: bool = False) -> "FakeSupabaseTable":
        self._order_col = column
        self._order_desc = desc
        return self

    def limit(self, count: int) -> "FakeSupabaseTable":
        self._limit_count = count
        return self

    def _matches(self, row: dict[str, Any]) -> bool:
        return all(row.get(col) == val for col, val in self._filters)

    def execute(self) -> SimpleNamespace:
        rows = self._rows

        if self._op == "insert":
            items = self._payload if isinstance(self._payload, list) else [self._payload]
            inserted: list[dict[str, Any]] = []
            for item in items:
                record = dict(item)
                record.setdefault("id", self._client.next_id())
                record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                rows.append(record)
                inserted.append(dict(record))
            return SimpleNamespace(data=inserted)

        if self._op == "upsert":
            items = self._payload if isinstance(self._payload, list) else [self._payload]
            upserted: list[dict[str, Any]] = []
            for item in items:
                record = dict(item)
                key_id = record.get("id")
                existing = next((r for r in rows if key_id is not None and r.get("id") == key_id), None)
                if existing is not None:
                    existing.update(record)
                    upserted.append(dict(existing))
                else:
                    record.setdefault("id", self._client.next_id())
                    record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
                    rows.append(record)
                    upserted.append(dict(record))
            return SimpleNamespace(data=upserted)

        if self._op == "delete":
            kept: list[dict[str, Any]] = []
            deleted: list[dict[str, Any]] = []
            for row in rows:
                if self._matches(row):
                    deleted.append(dict(row))
                else:
                    kept.append(row)
            self._client.tables[self._table_name] = kept
            return SimpleNamespace(data=deleted)

        # Default: select
        matched = [dict(row) for row in rows if self._matches(row)]
        if self._order_col:
            matched.sort(key=lambda r: r.get(self._order_col, ""), reverse=self._order_desc)
        if self._limit_count is not None:
            matched = matched[: self._limit_count]
        return SimpleNamespace(data=matched)


class FakeSupabaseClient:
    """In-memory mock of `supabase.Client` for offline testing."""

    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {}
        self._counter = 0

    def next_id(self) -> int:
        self._counter += 1
        return self._counter

    def table(self, name: str) -> FakeSupabaseTable:
        return FakeSupabaseTable(self, name)

    def reset(self) -> None:
        self.tables.clear()
        self._counter = 0


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


@pytest.fixture
def fake_supabase() -> FakeSupabaseClient:
    """Return the active in-memory FakeSupabaseClient."""
    return db.get_supabase()  # type: ignore[return-value]


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point Chroma at a temp folder, blank secrets, and mock Supabase offline."""
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(tracing, "DEFAULT_TRACE_PATH", tmp_path / "traces.jsonl")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.8-flash")
    # Pin it so a local backend/.env value cannot change test results.
    monkeypatch.setenv("LLM_FALLBACK_MODELS", '["gemini-2.5-flash"]')
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_KEY", "")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "")
    monkeypatch.setenv("AUTH_SECRET", TEST_AUTH_SECRET)

    mock_sb = FakeSupabaseClient()
    monkeypatch.setattr(db, "get_supabase_client", lambda: mock_sb)
    monkeypatch.setattr(db, "get_supabase", lambda: mock_sb)

    def _fake_current_user() -> dict[str, str]:
        user = {"id": "test-user", "email": "test@example.com", "name": "Test User"}
        memory.set_current_user_id(user["id"])
        return user

    app.dependency_overrides[get_current_user] = _fake_current_user
    memory.set_current_user_id("test-user")

    get_settings.cache_clear()
    bm25.clear_cache()
    memory.get_memory_store().clear_all()
    yield
    app.dependency_overrides.clear()
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
