"""BM25 keyword search over the chunks stored in a Chroma collection.

The index is built from the texts already in Chroma (so Chroma stays the single
source of truth), cached in memory, and rebuilt lazily the next time someone
searches after the collection has changed (upload or delete).
"""

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from app.retrieval import vectorstore


def tokenize(text: str) -> list[str]:
    """Lowercase and split on anything that is not a letter or digit."""
    return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]


@dataclass
class BM25Index:
    version: int  # vectorstore.collection_version() when this index was built
    chunks: list[Document]
    tokens: list[set[str]]  # unique tokens per chunk, used to skip non-matches
    bm25: BM25Okapi | None  # None when the collection is empty


# collection name -> cached index
_cache: dict[str, BM25Index] = {}


def clear_cache() -> None:
    """Drop every cached index (used by tests)."""
    _cache.clear()


def _build_index(collection: str, embeddings: Embeddings | None) -> BM25Index:
    version = vectorstore.collection_version(collection)
    chunks = vectorstore.get_all_chunks(collection, embeddings)
    tokenized = [tokenize(c.page_content) for c in chunks]
    # BM25Okapi divides by the number of docs, so it crashes on an empty corpus.
    bm25 = BM25Okapi(tokenized) if chunks else None
    return BM25Index(version, chunks, [set(t) for t in tokenized], bm25)


def get_index(collection: str, embeddings: Embeddings | None = None) -> BM25Index:
    """Return the cached index, rebuilding it only if the collection changed."""
    cached = _cache.get(collection)
    if cached is None or cached.version != vectorstore.collection_version(collection):
        cached = _build_index(collection, embeddings)
        _cache[collection] = cached
    return cached


def bm25_search(
    collection: str, query: str, k: int = 4, embeddings: Embeddings | None = None
) -> list[tuple[Document, float]]:
    """Return up to k (chunk, bm25_score) pairs, best first. Higher score = better.

    Only chunks that contain at least one query word are returned, so a chunk
    with no keyword overlap never gets a BM25 rank.
    """
    index = get_index(collection, embeddings)
    query_tokens = tokenize(query)
    if index.bm25 is None or not query_tokens:
        return []

    scores = index.bm25.get_scores(query_tokens)
    wanted = set(query_tokens)
    matches = [
        (index.chunks[i], float(scores[i]))
        for i in range(len(index.chunks))
        if wanted & index.tokens[i]
    ]
    matches.sort(key=lambda pair: pair[1], reverse=True)
    return matches[:k]
