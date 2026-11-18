import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import LOADERS, load_document
from app.retrieval import vectorstore
from app.retrieval.vectorstore import DEFAULT_COLLECTION

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_MB = 20


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
        docs = load_document(path)

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
    num_chunks = vectorstore.add_chunks(DEFAULT_COLLECTION, chunks)
    return {"filename": filename, "pages": len(docs), "num_chunks": num_chunks}


@router.get("")
def list_uploaded_documents():
    return {"documents": vectorstore.list_documents(DEFAULT_COLLECTION)}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(4, ge=1, le=20)


@router.post("/search")
def search_documents(req: SearchRequest):
    """Return top-k chunks. score = cosine distance (lower means more similar)."""
    results = vectorstore.similarity_search(DEFAULT_COLLECTION, req.query, req.k)
    return {
        "query": req.query,
        "results": [
            {"text": doc.page_content, "score": score, "metadata": doc.metadata}
            for doc, score in results
        ],
    }
