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


@lru_cache
def get_settings() -> Settings:
    """Return one shared Settings instance (read .env only once)."""
    return Settings()
