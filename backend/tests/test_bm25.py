from langchain_core.documents import Document

from app.retrieval import bm25, vectorstore
from app.retrieval.bm25 import bm25_search, get_index, tokenize

COL = "test"


def make_chunks(source: str, texts: list[str]) -> list[Document]:
    return [
        Document(page_content=t, metadata={"source": source, "page": 1, "chunk_id": i})
        for i, t in enumerate(texts)
    ]


def test_tokenize_lowercases_and_splits_on_non_alphanumeric():
    assert tokenize("Hello, World! GPT-4o costs $20.") == [
        "hello", "world", "gpt", "4o", "costs", "20"
    ]


def test_empty_collection_returns_empty_list(fake_embeddings):
    assert bm25_search(COL, "anything", k=5) == []


def test_finds_keyword_and_returns_metadata(fake_embeddings):
    vectorstore.add_chunks(
        COL, make_chunks("a.txt", ["cats purr and sleep", "rockets fly to space",
                                   "dogs bark loudly"])
    )
    hits = bm25_search(COL, "Rockets!", k=5)
    assert len(hits) == 1  # only chunks that share a query word are returned
    doc, score = hits[0]
    assert doc.page_content == "rockets fly to space"
    assert doc.metadata == {"source": "a.txt", "page": 1, "chunk_id": 1}
    assert score > 0


def test_index_is_cached_until_collection_changes(fake_embeddings):
    vectorstore.add_chunks(COL, make_chunks("a.txt", ["one", "two"]))
    first = get_index(COL)
    assert get_index(COL) is first  # no change -> same cached object


def test_index_refreshes_after_new_upload(fake_embeddings):
    vectorstore.add_chunks(COL, make_chunks("a.txt", ["cats purr", "dogs bark"]))
    assert bm25_search(COL, "zorblax") == []

    vectorstore.add_chunks(COL, make_chunks("b.txt", ["the zorblax engine"]))
    hits = bm25_search(COL, "zorblax")
    assert [doc.metadata["source"] for doc, _ in hits] == ["b.txt"]


def test_index_refreshes_after_delete(fake_embeddings):
    vectorstore.add_chunks(COL, make_chunks("a.txt", ["cats purr", "dogs bark"]))
    assert len(bm25_search(COL, "cats")) == 1
    vectorstore.delete_collection(COL)
    assert bm25_search(COL, "cats") == []


def test_clear_cache():
    bm25._cache["x"] = "stale"  # type: ignore[assignment]
    bm25.clear_cache()
    assert bm25._cache == {}
