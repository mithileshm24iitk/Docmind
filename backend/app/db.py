"""Supabase client initialization."""

from functools import lru_cache

from supabase import Client, create_client

from app.config import get_settings


@lru_cache
def get_supabase_client() -> Client:
    """Create and return a cached Supabase client using settings from backend/.env."""
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be configured.")
    return create_client(settings.supabase_url, settings.supabase_key)


def get_supabase() -> Client:
    """Return the shared Supabase client (alias for get_supabase_client so callers/tests can patch either)."""
    return get_supabase_client()
