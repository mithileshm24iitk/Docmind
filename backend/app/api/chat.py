from typing import Annotated

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from app.rag.chain import LLMError, answer

router = APIRouter(prefix="/chat", tags=["chat"])

MAX_QUESTION_CHARS = 2000


class AskRequest(BaseModel):
    # strip_whitespace makes "   " count as empty, so it is rejected too.
    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS),
    ]
    k: int | None = Field(None, ge=1, le=20)


@router.post("/ask")
def ask(req: AskRequest) -> dict:
    """Answer a question from the uploaded documents, with citations and sources."""
    try:
        return answer(req.question, k=req.k)
    except LLMError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message) from None
