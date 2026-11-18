import math

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.retrieval import vectorstore
from app.retrieval.hybrid import hybrid_search, rrf_scores

COL = "test"


class KeywordBlindEmbeddings(Embeddings):
    """A fake 'semantic' model that only understands two topic words.

    Every other word (like the made-up keyword 'zorblax') is invisible to it,
    which mimics a real embedding model missing an exact rare term/code.
    Vector = [bias, count('rocket'), count('cat')], normalised.
    """

    def _embed(self, text: str) -> list[float]:
        words = text.lower().split()
        vec = [1.0, float(words.count("rocket")), float(words.count("cat"))]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def blind_embeddings(monkeypatch) -> KeywordBlindEmbeddings:
    fake = KeywordBlindEmbeddings()
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: fake)
    return fake


def add_corpus() -> None:
    """12 plain notes + 1 rocket-heavy chunk that contains the keyword."""
    plain = [f"plain note number {i}" for i in range(12)]
    target = "rocket rocket rocket zorblax"
    texts = plain + [target]
    chunks = [
        Document(page_content=t, metadata={"source": "c.txt", "page": 1, "chunk_id": i})
        for i, t in enumerate(texts)
    ]
    vectorstore.add_chunks(COL, chunks)


# ---------- RRF on hand-made lists ----------

def test_rrf_scores_match_formula():
    scores = rrf_scores([["a", "b", "c"], ["c", "a"]], c=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_ordering():
    # 'a' is decent in both lists, 'x' is #1 in only one list.
    vector = ["x", "a", "b"]
    keyword = ["a", "b", "y"]
    scores = rrf_scores([vector, keyword])
    order = sorted(scores, key=lambda k: scores[k], reverse=True)
    # a: 1/62+1/61, b: 1/63+1/62, x: 1/61, y: 1/63
    assert order == ["a", "b", "x", "y"]


def test_rrf_empty_lists():
    assert rrf_scores([[], []]) == {}


# ---------- hybrid_search end-to-end with fake embeddings ----------

def test_keyword_found_by_bm25_but_missed_by_vector(blind_embeddings):
    add_corpus()
    k = 4
    results = hybrid_search(COL, "zorblax", k=k)

    # Sanity check: the vector search alone (3*k candidates) misses the target.
    vector_only = vectorstore.similarity_search(COL, "zorblax", 3 * k)
    assert all("zorblax" not in doc.page_content for doc, _ in vector_only)

    # ...but hybrid still returns it, thanks to its BM25 rank.
    target = [r for r in results if "zorblax" in r["text"]]
    assert len(target) == 1
    assert target[0]["vector_rank"] is None
    assert target[0]["bm25_rank"] == 1
    assert target[0]["score"] == pytest.approx(1 / 61)
    assert len(results) == k


def test_result_fields(blind_embeddings):
    add_corpus()
    result = hybrid_search(COL, "plain note", k=2)[0]
    assert set(result) == {"text", "metadata", "vector_rank", "bm25_rank", "score"}
    assert result["metadata"]["source"] == "c.txt"


def test_empty_collection_returns_empty_list(blind_embeddings):
    assert hybrid_search(COL, "anything", k=4) == []
