"""Hybrid retrieval: vector search + BM25, merged with Reciprocal Rank Fusion."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.retrieval import vectorstore
from app.retrieval.bm25 import bm25_search

RRF_K = 60  # standard RRF constant; dampens the advantage of being ranked #1
CANDIDATE_MULTIPLIER = 3  # fetch 3*k candidates from each retriever


def chunk_key(doc: Document) -> str:
    """Identify a chunk by source/page/chunk_id (same parts as its Chroma ID)."""
    m = doc.metadata
    return f"{m['source']}:p{m['page']}:c{m['chunk_id']}"


def rrf_scores(ranked_lists: list[list[str]], c: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (c + rank of d).

    Ranks start at 1. A key missing from a list simply gets nothing from it.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank)
    return scores


def hybrid_search(
    collection: str, query: str, k: int = 4, embeddings: Embeddings | None = None
) -> list[dict]:
    """Run vector + BM25 search separately, fuse with RRF, return the top k.

    Each result: text, metadata, vector_rank, bm25_rank (None if the chunk was
    not in that retriever's candidates) and score (the fused RRF score).
    """
    n = k * CANDIDATE_MULTIPLIER
    vector_hits = vectorstore.similarity_search(collection, query, n, embeddings)
    bm25_hits = bm25_search(collection, query, n, embeddings)

    vector_keys = [chunk_key(doc) for doc, _ in vector_hits]
    bm25_keys = [chunk_key(doc) for doc, _ in bm25_hits]
    docs = {chunk_key(doc): doc for doc, _ in vector_hits + bm25_hits}

    vector_rank = {key: i for i, key in enumerate(vector_keys, start=1)}
    bm25_rank = {key: i for i, key in enumerate(bm25_keys, start=1)}

    fused = rrf_scores([vector_keys, bm25_keys])
    best = sorted(fused, key=lambda key: fused[key], reverse=True)[:k]
    return [
        {
            "text": docs[key].page_content,
            "metadata": docs[key].metadata,
            "vector_rank": vector_rank.get(key),
            "bm25_rank": bm25_rank.get(key),
            "score": fused[key],
        }
        for key in best
    ]
