"""Tests for Supabase client initialization in app/db.py."""

from types import SimpleNamespace

import pytest

from app import db
from app.config import get_settings

_REAL_GET_SUPABASE_CLIENT = db.get_supabase_client


def test_get_supabase_client_raises_when_unconfigured(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_KEY", "")
    get_settings.cache_clear()
    _REAL_GET_SUPABASE_CLIENT.cache_clear()

    with pytest.raises(RuntimeError, match="SUPABASE_URL and SUPABASE_KEY"):
        _REAL_GET_SUPABASE_CLIENT()


def test_get_supabase_client_calls_create_client(monkeypatch):
    captured: dict[str, str] = {}
    fake_client = SimpleNamespace(name="fake-sb")

    def _fake_create_client(url: str, key: str):
        captured["url"] = url
        captured["key"] = key
        return fake_client

    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-service-key")
    get_settings.cache_clear()
    _REAL_GET_SUPABASE_CLIENT.cache_clear()
    monkeypatch.setattr(db, "create_client", _fake_create_client)

    client = _REAL_GET_SUPABASE_CLIENT()
    assert client is fake_client
    assert captured == {
        "url": "https://project.supabase.co",
        "key": "test-service-key",
    }
    _REAL_GET_SUPABASE_CLIENT.cache_clear()
