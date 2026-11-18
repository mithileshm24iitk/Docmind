import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ingestion.chunking import chunk_documents
from app.ingestion.loaders import LOADERS, load_document

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_MB = 20


@router.post("/preview")
async def preview_chunks(
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
):
    filename = Path(file.filename or "upload").name
    ext = Path(filename).suffix.lower()
    if ext not in LOADERS:
        raise HTTPException(400, f"Unsupported type. Allowed: {', '.join(LOADERS)}")
    if chunk_overlap >= chunk_size:
        raise HTTPException(400, "chunk_overlap must be smaller than chunk_size")

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

    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    return {
        "filename": filename,
        "pages": len(docs),
        "num_chunks": len(chunks),
        "chunks": [
            {"text": c.page_content, "metadata": c.metadata} for c in chunks[:5]
        ],
    }
