from pathlib import Path

from app.config import Settings, get_settings


def test_defaults_without_env_file(monkeypatch):
    monkeypatch.delenv("CHROMA_DIR", raising=False)
    s = Settings(_env_file=None)
    assert s.chunk_size == 1000
    assert s.chunk_overlap == 200
    assert s.embedding_model == "gemini-embedding-001"
    assert s.chroma_dir.parts[-3:] == ("backend", "data", "chroma")


def test_env_var_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path))
    monkeypatch.setenv("CHUNK_SIZE", "500")
    get_settings.cache_clear()
    s = get_settings()
    assert s.chroma_dir == Path(tmp_path)
    assert s.chunk_size == 500
