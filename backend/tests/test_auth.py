"""Tests for Google OAuth login, JWT verification, and route protection (`get_current_user`)."""

from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from app.api import auth
from app.api.auth import create_access_token
from app.config import get_settings
from app.main import app

client = TestClient(app)


def test_google_login_url_returns_503_when_missing_client_id():
    r = client.get("/auth/login")
    assert r.status_code == 503


def test_google_login_url_returns_oauth_consent_url(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "my-google-client-id.apps.googleusercontent.com")
    get_settings.cache_clear()

    r = client.get("/auth/google", params={"redirect_uri": "http://localhost:3000"})
    assert r.status_code == 200
    auth_url = r.json()["auth_url"]
    assert auth_url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "my-google-client-id.apps.googleusercontent.com" in auth_url
    assert "response_type=code" in auth_url


def test_google_code_exchange_upserts_user_and_returns_jwt(monkeypatch, fake_supabase):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "csecret-that-is-at-least-32-bytes-long")
    get_settings.cache_clear()

    def _fake_post(url: str, data: dict, timeout: float):
        assert url == auth.GOOGLE_TOKEN_URL
        assert data["code"] == "google-auth-code-123"
        return SimpleNamespace(status_code=200, json=lambda: {"access_token": "ya29.fake"})

    def _fake_get(url: str, headers: dict, timeout: float):
        assert url == auth.GOOGLE_USERINFO_URL
        assert headers["Authorization"] == "Bearer ya29.fake"
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "sub": "google-uid-42",
                "email": "student@example.com",
                "name": "DocuMind Student",
            },
        )

    monkeypatch.setattr(httpx, "post", _fake_post)
    monkeypatch.setattr(httpx, "get", _fake_get)

    r = client.post("/auth/google", json={"code": "google-auth-code-123"})
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["user"] == {
        "id": "google-uid-42",
        "email": "student@example.com",
        "name": "DocuMind Student",
    }
    assert body["access_token"]

    # User was persisted to Supabase `users` table
    assert fake_supabase.tables["users"][0]["id"] == "google-uid-42"
    assert fake_supabase.tables["users"][0]["email"] == "student@example.com"

    # Returned JWT passes /auth/verify when real get_current_user runs
    app.dependency_overrides.clear()
    verify_res = client.get(
        "/auth/verify",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert verify_res.status_code == 200
    assert verify_res.json() == {"valid": True, "user": body["user"]}


def test_protected_routes_reject_unauthenticated_or_bad_token():
    app.dependency_overrides.clear()

    # Without Authorization header -> 401
    assert client.get("/documents").status_code == 401
    assert client.post("/chat/ask", json={"question": "hello"}).status_code == 401
    assert (
        client.post("/chat/stream", json={"session_id": "s1", "question": "hello"}).status_code
        == 401
    )
    assert client.delete("/chat/sessions/s1").status_code == 401
    assert client.get("/auth/verify").status_code == 401

    # With invalid JWT -> 401
    bad_headers = {"Authorization": "Bearer not-a-valid-jwt"}
    assert client.get("/documents", headers=bad_headers).status_code == 401
    assert client.post("/chat/ask", json={"question": "hello"}, headers=bad_headers).status_code == 401


def test_protected_routes_succeed_with_valid_bearer_token_and_isolate_users(
    fake_embeddings, fake_llm, fake_supabase
):
    app.dependency_overrides.clear()

    alice_token = create_access_token(
        {"id": "user-alice", "email": "alice@example.com", "name": "Alice"}
    )
    bob_token = create_access_token(
        {"id": "user-bob", "email": "bob@example.com", "name": "Bob"}
    )
    alice_headers = {"Authorization": f"Bearer {alice_token}"}
    bob_headers = {"Authorization": f"Bearer {bob_token}"}

    # Upload document as Alice
    up = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"Photosynthesis lets plants turn sunlight into energy.", "text/plain")},
        headers=alice_headers,
    )
    assert up.status_code == 200

    # Ask with session_id="shared" as Alice
    r_alice = client.post(
        "/chat/ask",
        json={"question": "how do plants turn sunlight into energy", "session_id": "shared"},
        headers=alice_headers,
    )
    assert r_alice.status_code == 200

    # Ask with same session_id="shared" as Bob -> no prior history for Bob, so no rewrite call
    r_bob = client.post(
        "/chat/ask",
        json={"question": "how do plants turn sunlight into energy", "session_id": "shared"},
        headers=bob_headers,
    )
    assert r_bob.status_code == 200

    rows = fake_supabase.tables["chat_messages"]
    assert len(rows) == 2
    assert {row["user_id"] for row in rows} == {"user-alice", "user-bob"}
