"""ChromaDB persistent vector store (via langchain-chroma)."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.retrieval.embeddings import get_embeddings

DEFAULT_COLLECTION = "default"

# collection name -> number of writes (uploads/deletes) since the app started.
# Other modules (e.g. the BM25 cache) compare this number to know if their
# cached copy of the collection is stale.
_versions: dict[str, int] = {}


def collection_version(collection: str) -> int:
    """Return a counter that increases every time the collection changes."""
    return _versions.get(collection, 0)


def _mark_changed(collection: str) -> None:
    _versions[collection] = collection_version(collection) + 1


def get_store(collection: str, embeddings: Embeddings | None = None) -> Chroma:
    """Open (or create) a collection stored on disk at CHROMA_DIR."""
    settings = get_settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        # Cosine distance is the usual choice for text embeddings.
        collection_metadata={"hnsw:space": "cosine"},
    )


def _chunk_id(chunk: Document) -> str:
    """Stable ID like 'report.pdf:p3:c12', so re-uploading a file overwrites its chunks."""
    m = chunk.metadata
    return f"{m['source']}:p{m['page']}:c{m['chunk_id']}"


def add_chunks(
    collection: str, chunks: list[Document], embeddings: Embeddings | None = None
) -> int:
    """Embed chunks and store them. Keeps only source, page, chunk_id as metadata."""
    if not chunks:
        return 0
    docs = [
        Document(
            page_content=c.page_content,
            metadata={
                "source": c.metadata["source"],
                "page": int(c.metadata["page"]),
                "chunk_id": int(c.metadata["chunk_id"]),
            },
        )
        for c in chunks
    ]
    store = get_store(collection, embeddings)
    store.add_documents(docs, ids=[_chunk_id(d) for d in docs])
    _mark_changed(collection)
    return len(docs)


def similarity_search(
    collection: str, query: str, k: int = 4, embeddings: Embeddings | None = None
) -> list[tuple[Document, float]]:
    """Return the top-k (chunk, score) pairs. Score is cosine distance: lower = closer."""
    store = get_store(collection, embeddings)
    return store.similarity_search_with_score(query, k=k)


def delete_collection(collection: str, embeddings: Embeddings | None = None) -> None:
    """Delete a whole collection (all its chunks)."""
    get_store(collection, embeddings).delete_collection()
    _mark_changed(collection)


def get_all_chunks(
    collection: str, embeddings: Embeddings | None = None
) -> list[Document]:
    """Read every stored chunk (text + metadata) back out of Chroma."""
    store = get_store(collection, embeddings)
    data = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


def list_documents(
    collection: str, embeddings: Embeddings | None = None
) -> list[dict]:
    """List unique source files in a collection with their chunk counts."""
    store = get_store(collection, embeddings)
    metadatas = store.get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for m in metadatas:
        counts[m["source"]] = counts.get(m["source"], 0) + 1
    return [{"filename": name, "num_chunks": n} for name, n in sorted(counts.items())]
