"""Google OAuth login, JWT token verification, and the `get_current_user` dependency."""

import time
from urllib.parse import urlencode

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app import db
from app.config import get_settings
from app.rag.memory import set_current_user_id

router = APIRouter(prefix="/auth", tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
DEFAULT_REDIRECT_URI = "http://localhost:3000"

JWT_ALGORITHM = "HS256"
JWT_TTL_SECONDS = 60 * 60 * 24  # 24 hours
AUTH_SECRET_MIN_LENGTH = 32  # HS256 keys should be at least 256 bits

_bearer_scheme = HTTPBearer(auto_error=False)


def _jwt_secret() -> str:
    """Return AUTH_SECRET, the only key used to sign and verify app JWTs.

    There is deliberately no fallback (no default, no other secret), so a
    misconfigured server refuses to issue or accept tokens instead of
    silently using a guessable or shared key.
    """
    secret = get_settings().auth_secret
    if len(secret) < AUTH_SECRET_MIN_LENGTH:
        raise HTTPException(
            status_code=503,
            detail=(
                "Auth is not configured: AUTH_SECRET must be set "
                f"(at least {AUTH_SECRET_MIN_LENGTH} characters)."
            ),
        )
    return secret


def create_access_token(user: dict[str, str], ttl_seconds: int = JWT_TTL_SECONDS) -> str:
    """Create a signed JWT containing the user's id (`sub`), email, and name."""
    now = int(time.time())
    payload = {
        "sub": user["id"],
        "email": user.get("email", ""),
        "name": user.get("name", ""),
        "iat": now,
        "exp": now + ttl_seconds,
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)


def verify_access_token(token: str) -> dict[str, str]:
    """Decode and validate a signed JWT, returning the user dict or raising 401."""
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from None

    user_id = payload.get("sub")
    if not user_id or not isinstance(user_id, str):
        raise HTTPException(status_code=401, detail="Invalid token payload")

    return {
        "id": user_id,
        "email": str(payload.get("email", "")),
        "name": str(payload.get("name", "")),
    }


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict[str, str]:
    """FastAPI dependency that validates the Bearer token and sets the active user ID."""
    if credentials is None or credentials.scheme.lower() != "bearer" or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    user = verify_access_token(credentials.credentials)
    set_current_user_id(user["id"])
    return user


def _build_google_auth_url(redirect_uri: str) -> str:
    """Construct the Google OAuth 2.0 consent screen URL."""
    settings = get_settings()
    if not settings.google_client_id:
        raise HTTPException(status_code=503, detail="Google OAuth is not configured.")

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def _exchange_google_code(code: str, redirect_uri: str) -> dict[str, str]:
    """Exchange a Google authorization code for user profile details (id, email, name)."""
    settings = get_settings()
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(status_code=503, detail="Google OAuth is not configured.")

    token_resp = httpx.post(
        GOOGLE_TOKEN_URL,
        data={
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=15.0,
    )
    if token_resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Failed to exchange Google OAuth code.")

    google_access_token = token_resp.json().get("access_token")
    if not google_access_token:
        raise HTTPException(status_code=401, detail="Google token response missing access_token.")

    userinfo_resp = httpx.get(
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {google_access_token}"},
        timeout=15.0,
    )
    if userinfo_resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Failed to fetch Google user profile.")

    profile = userinfo_resp.json()
    user_id = profile.get("sub") or profile.get("id")
    email = profile.get("email", "")
    if not user_id or not email:
        raise HTTPException(status_code=401, detail="Google profile missing user ID or email.")

    return {
        "id": str(user_id),
        "email": str(email),
        "name": str(profile.get("name", email)),
    }


def _upsert_user(user: dict[str, str]) -> None:
    """Save or update the authenticated user in the Supabase `users` table."""
    db.get_supabase().table("users").upsert(
        {
            "id": user["id"],
            "email": user["email"],
            "name": user.get("name", ""),
        }
    ).execute()


class GoogleCodeRequest(BaseModel):
    code: str = Field(..., min_length=1)
    redirect_uri: str = DEFAULT_REDIRECT_URI


@router.get("/login")
@router.get("/google")
def google_login_url(redirect_uri: str = Query(DEFAULT_REDIRECT_URI)) -> dict[str, str]:
    """Return the Google OAuth 2.0 authorization URL for the frontend login button."""
    return {"auth_url": _build_google_auth_url(redirect_uri)}


@router.post("/google")
def google_oauth_exchange(req: GoogleCodeRequest) -> dict:
    """Exchange a Google authorization code, upsert user in Supabase, and return a JWT."""
    user = _exchange_google_code(req.code, req.redirect_uri)
    _upsert_user(user)
    token = create_access_token(user)
    return {"access_token": token, "token_type": "bearer", "user": user}


@router.get("/callback")
def google_oauth_callback(
    code: str = Query(..., min_length=1),
    redirect_uri: str = Query(DEFAULT_REDIRECT_URI),
) -> dict:
    """Handle Google OAuth callback via GET query parameter."""
    user = _exchange_google_code(code, redirect_uri)
    _upsert_user(user)
    token = create_access_token(user)
    return {"access_token": token, "token_type": "bearer", "user": user}


@router.get("/verify")
@router.get("/me")
def verify_token_endpoint(user: dict[str, str] = Depends(get_current_user)) -> dict:
    """Verify the Bearer JWT token and return the authenticated user's profile."""
    return {"valid": True, "user": user}
