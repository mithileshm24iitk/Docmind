"""App settings loaded from environment variables and backend/.env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ folder (this file lives in backend/app/config.py)
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All config in one place. Values in backend/.env override these defaults."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str = ""
    chroma_dir: Path = BACKEND_DIR / "data" / "chroma"
    # Stable text embedding model from the Gemini API docs (ai.google.dev).
    embedding_model: str = "gemini-embedding-001"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- RAG chain (phase 4) ---
    # Stable Gemini text model from the Gemini API models page (ai.google.dev).
    llm_model: str = "gemini-3.8-flash"
    llm_fallback_models: list[str] = ["gemini-2.5-flash"]
    llm_temperature: float = 0.2
    top_k: int = 5  # chunks sent to the LLM as context
    # Cosine distance cutoff for the relevance gate (0 = identical, larger = less
    # similar). If even the closest chunk is farther than this, we skip the LLM.
    relevance_max_distance: float = 0.6

    # --- Conversation memory (phase 5) ---
    memory_max_turns: int = 5  # user/assistant message pairs kept per session

    # --- Auth & Supabase (phase 7) ---
    supabase_url: str = ""
    supabase_key: str = ""
    google_client_id: str = ""
    google_client_secret: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return one shared Settings instance (read .env only once)."""
    return Settings()
