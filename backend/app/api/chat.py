from typing import Annotated

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StringConstraints

from app.rag.chain import LLMError, answer, stream_answer
from app.rag.memory import get_memory_store

router = APIRouter(prefix="/chat", tags=["chat"])

MAX_QUESTION_CHARS = 2000
MAX_SESSION_ID_CHARS = 128

QuestionStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS),
]
SessionIdStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_SESSION_ID_CHARS),
]


class AskRequest(BaseModel):
    # strip_whitespace makes "   " count as empty, so it is rejected too.
    question: QuestionStr
    session_id: SessionIdStr | None = None
    k: int | None = Field(None, ge=1, le=20)


class StreamRequest(BaseModel):
    session_id: SessionIdStr
    question: QuestionStr
    k: int | None = Field(None, ge=1, le=20)


@router.post("/ask")
def ask(req: AskRequest) -> dict:
    """Answer a question from the uploaded documents, with citations and sources."""
    try:
        return answer(req.question, session_id=req.session_id, k=req.k)
    except LLMError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from None


@router.post("/stream")
def stream(req: StreamRequest) -> StreamingResponse:
    """Stream an answer as Server-Sent Events (token chunks + final done event)."""
    return StreamingResponse(
        stream_answer(req.question, session_id=req.session_id, k=req.k),
        media_type="text/event-stream",
    )


@router.delete("/sessions/{session_id}")
def clear_session(session_id: str) -> dict:
    """Clear conversation memory for a session."""
    get_memory_store().clear(session_id)
    return {"session_id": session_id, "cleared": True}
