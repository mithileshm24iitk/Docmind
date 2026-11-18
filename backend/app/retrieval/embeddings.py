"""Gemini embeddings via langchain-google-genai."""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> Embeddings:
    """Return a Gemini embeddings object (model name comes from config).

    Typed as the generic LangChain `Embeddings` so tests can swap in a fake.
    """
    settings = get_settings()
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in backend/.env")
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
