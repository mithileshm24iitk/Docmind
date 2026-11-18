import logging
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.api.auth import get_current_user
from app.config import get_settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import LOADERS, count_pages, load_document
from app.retrieval import vectorstore
from app.retrieval.bm25 import bm25_search
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vectorstore import DEFAULT_COLLECTION

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(get_current_user)],
)

logger = logging.getLogger(__name__)

MAX_MB = 20
MAX_PAGES = 300


async def _read_and_load(file: UploadFile) -> tuple[str, list[Document]]:
    """Validate an upload, save it to a temp file, and load it into page Documents."""
    filename = Path(file.filename or "upload").name
    ext = Path(filename).suffix.lower()
    if ext not in LOADERS:
        raise HTTPException(400, f"Unsupported type. Allowed: {', '.join(LOADERS)}")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_MB} MB")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_bytes(data)
        try:
            pages = count_pages(path)
            if pages > MAX_PAGES:
                raise HTTPException(
                    413, f"Document has {pages} pages; the limit is {MAX_PAGES} pages."
                )
            docs = load_document(path)
        except HTTPException:
            raise
        except Exception as exc:  # corrupt or encrypted file: a clean 422, not a 500
            logger.warning("Could not read upload: %s", type(exc).__name__)
            raise HTTPException(422, "Could not read this file (corrupt or encrypted?)") from None

    if not docs:
        raise HTTPException(422, "No extractable text (scanned PDF?)")
    return filename, docs


@router.post("/preview")
async def preview_chunks(
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    if chunk_overlap >= chunk_size:
        raise HTTPException(400, "chunk_overlap must be smaller than chunk_size")
    filename, docs = await _read_and_load(file)

    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    return {
        "filename": filename,
        "pages": len(docs),
        "num_chunks": len(chunks),
        "chunks": [
            {"text": c.page_content, "metadata": c.metadata} for c in chunks[:5]
        ],
    }


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Load -> chunk -> embed -> store in the default Chroma collection."""
    filename, docs = await _read_and_load(file)
    settings = get_settings()
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    try:
        num_chunks = vectorstore.add_chunks(DEFAULT_COLLECTION, chunks)
    except vectorstore.EmbeddingError as e:
        raise HTTPException(e.status_code, e.message) from None
    except Exception as exc:  # e.g. missing/invalid API key: never a 500 or a stack trace
        logger.warning("Embedding failed: %s", type(exc).__name__)
        raise HTTPException(
            503, "Could not embed the document right now. Please try again later."
        ) from None
    return {"filename": filename, "pages": len(docs), "num_chunks": num_chunks}


@router.get("")
def list_uploaded_documents():
    return {"documents": vectorstore.list_documents(DEFAULT_COLLECTION)}


@router.delete("/{filename}")
def delete_uploaded_document(filename: str):
    """Remove one file's chunks from Chroma (BM25 rebuilds on the next search). 404 if unknown."""
    deleted = vectorstore.delete_document(DEFAULT_COLLECTION, filename)
    if deleted == 0:
        raise HTTPException(404, f"Document not found: {filename}")
    return {"filename": filename, "deleted_chunks": deleted}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(4, ge=1, le=20)
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid"


@router.post("/search")
def search_documents(req: SearchRequest):
    """Return top-k chunks.

    vector: score = cosine distance (lower means more similar).
    bm25:   score = BM25 keyword score (higher is better).
    hybrid: score = fused RRF score (higher is better), plus vector_rank/bm25_rank.
    """
    if req.mode == "hybrid":
        results = hybrid_search(DEFAULT_COLLECTION, req.query, req.k)
    elif req.mode == "bm25":
        hits = bm25_search(DEFAULT_COLLECTION, req.query, req.k)
        results = [
            {
                "text": doc.page_content,
                "score": score,
                "metadata": doc.metadata,
                "bm25_rank": rank,
            }
            for rank, (doc, score) in enumerate(hits, start=1)
        ]
    else:
        hits = vectorstore.similarity_search(DEFAULT_COLLECTION, req.query, req.k)
        results = [
            {"text": doc.page_content, "score": score, "metadata": doc.metadata}
            for doc, score in hits
        ]
    return {"query": req.query, "mode": req.mode, "results": results}
