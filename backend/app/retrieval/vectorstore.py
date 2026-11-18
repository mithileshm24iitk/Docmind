"""ChromaDB persistent vector store (via langchain-chroma)."""

import logging
import time

import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.retrieval.embeddings import get_embeddings

DEFAULT_COLLECTION = "default"

logger = logging.getLogger(__name__)

# Uploads are embedded in batches, and each batch is retried with exponential
# backoff (1s, 2s, 4s, 8s) when Gemini says "slow down" (429) or "busy" (503).
EMBED_BATCH_SIZE = 50
EMBED_MAX_TRIES = 5
EMBED_BASE_DELAY_S = 1.0
RETRYABLE_STATUS = {429, 503}
_sleep = time.sleep  # tests replace this so they do not actually wait


class EmbeddingError(Exception):
    """Embedding failed for good. Carries a clean status (429 or 503) and message."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message

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


def _status_code(exc: BaseException) -> int | None:
    """Find an HTTP status on the exception or its causes.

    langchain-google-genai wraps google-genai's APIError (which has an int
    `.code`) in its own error type, so we walk the __cause__/__context__ chain.
    Timeouts count as 503 (service busy).
    """
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, (TimeoutError, httpx.TimeoutException)):
            return 503
        code = getattr(seen, "code", None)
        if isinstance(code, int):
            return code
        seen = seen.__cause__ or seen.__context__
    return None


def _with_backoff(fn, *args, **kwargs):
    """Call fn, retrying on 429/503 up to EMBED_MAX_TRIES times with exponential backoff.

    Raises EmbeddingError(429 or 503) when the last try still fails with 429/503.
    Any other error is re-raised immediately (retrying would not help).
    """
    for attempt in range(1, EMBED_MAX_TRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            code = _status_code(exc)
            if code not in RETRYABLE_STATUS:
                raise
            if attempt == EMBED_MAX_TRIES:
                logger.warning("Embedding failed after %d tries (HTTP %s)", attempt, code)
                if code == 429:
                    raise EmbeddingError(
                        429, "The embedding rate limit was reached. Wait a minute and try again."
                    ) from None
                raise EmbeddingError(
                    503, "The embedding service is temporarily unavailable. Please try again."
                ) from None
            delay = EMBED_BASE_DELAY_S * 2 ** (attempt - 1)
            logger.warning(
                "Embedding got HTTP %s (try %d/%d); retrying in %.0fs",
                code, attempt, EMBED_MAX_TRIES, delay,
            )
            _sleep(delay)


def add_chunks(
    collection: str, chunks: list[Document], embeddings: Embeddings | None = None
) -> int:
    """Embed chunks in batches and store them. Keeps only source, page, chunk_id as metadata.

    If a batch fails for good, the batches already stored for this upload are
    removed again, so a half-indexed file never shows up in search.
    """
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
    added_ids: list[str] = []
    try:
        for start in range(0, len(docs), EMBED_BATCH_SIZE):
            batch = docs[start : start + EMBED_BATCH_SIZE]
            ids = [_chunk_id(d) for d in batch]
            _with_backoff(store.add_documents, batch, ids=ids)
            added_ids.extend(ids)
    except Exception:
        if added_ids:
            store.delete(ids=added_ids)  # roll back the partial upload
        raise
    finally:
        if added_ids:
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


def delete_document(
    collection: str, filename: str, embeddings: Embeddings | None = None
) -> int:
    """Delete every chunk whose source is `filename`. Returns how many were deleted (0 = unknown)."""
    store = get_store(collection, embeddings)
    ids = store.get(where={"source": filename}, include=[])["ids"]
    if not ids:
        return 0
    store.delete(ids=ids)
    _mark_changed(collection)  # BM25 sees the new version and rebuilds its index
    return len(ids)


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
