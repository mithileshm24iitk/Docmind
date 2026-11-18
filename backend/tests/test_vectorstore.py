from langchain_core.documents import Document

from app.retrieval import vectorstore

COL = "test"


def make_chunks(source: str, texts: list[str]) -> list[Document]:
    return [
        Document(page_content=t, metadata={"source": source, "page": 1, "chunk_id": i})
        for i, t in enumerate(texts)
    ]


def test_add_and_search_returns_metadata(fake_embeddings):
    chunks = make_chunks("a.txt", ["cats purr and sleep", "rockets fly to space"])
    assert vectorstore.add_chunks(COL, chunks) == 2

    results = vectorstore.similarity_search(COL, "rockets space", k=1)
    doc, score = results[0]
    assert doc.page_content == "rockets fly to space"
    assert doc.metadata == {"source": "a.txt", "page": 1, "chunk_id": 1}
    assert isinstance(score, float)


def test_list_documents_counts_chunks(fake_embeddings):
    vectorstore.add_chunks(COL, make_chunks("a.txt", ["one", "two"]))
    vectorstore.add_chunks(COL, make_chunks("b.txt", ["three"]))
    assert vectorstore.list_documents(COL) == [
        {"filename": "a.txt", "num_chunks": 2},
        {"filename": "b.txt", "num_chunks": 1},
    ]


def test_reupload_same_file_does_not_duplicate(fake_embeddings):
    vectorstore.add_chunks(COL, make_chunks("a.txt", ["one", "two"]))
    vectorstore.add_chunks(COL, make_chunks("a.txt", ["one", "two"]))
    assert vectorstore.list_documents(COL) == [{"filename": "a.txt", "num_chunks": 2}]


def test_delete_collection(fake_embeddings):
    vectorstore.add_chunks(COL, make_chunks("a.txt", ["one"]))
    vectorstore.delete_collection(COL)
    assert vectorstore.list_documents(COL) == []


def test_add_empty_list_is_noop(fake_embeddings):
    assert vectorstore.add_chunks(COL, []) == 0
