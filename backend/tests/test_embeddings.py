import pytest
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.retrieval.embeddings import get_embeddings


@pytest.fixture(autouse=True)
def clear_cache():
    get_embeddings.cache_clear()
    yield
    get_embeddings.cache_clear()


def test_missing_key_raises():
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        get_embeddings()


def test_builds_gemini_embeddings_with_configured_model(monkeypatch):
    # Creating the object does not call the API; only embed_* calls do.
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-tests")
    get_settings.cache_clear()
    emb = get_embeddings()
    assert isinstance(emb, GoogleGenerativeAIEmbeddings)
    assert get_settings().embedding_model in emb.model
