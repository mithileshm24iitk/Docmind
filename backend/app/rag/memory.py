"""Session conversation memory store backed by the Supabase `chat_messages` table."""

from abc import ABC, abstractmethod
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from app import db
from app.config import get_settings

Turn = dict[str, str]

DEFAULT_USER_ID = "anonymous"
_current_user_id: ContextVar[str] = ContextVar("current_user_id", default=DEFAULT_USER_ID)


def set_current_user_id(user_id: str) -> None:
    """Set the authenticated user's ID for the active request context."""
    _current_user_id.set(user_id or DEFAULT_USER_ID)


def get_current_user_id() -> str:
    """Return the authenticated user's ID for the active request context."""
    return _current_user_id.get()


class BaseMemoryStore(ABC):
    """Interface for storing per-session user/assistant turns."""

    @abstractmethod
    def get_history(self, session_id: str, user_id: str | None = None) -> list[Turn]:
        """Return stored turns for a session, oldest first."""

    @abstractmethod
    def add_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        user_id: str | None = None,
    ) -> None:
        """Append one (user, assistant) turn and truncate to the max turn limit."""

    @abstractmethod
    def clear(self, session_id: str, user_id: str | None = None) -> None:
        """Delete all turns for a session."""


class SupabaseSessionStore(BaseMemoryStore):
    """Supabase-backed session store reading/writing `chat_messages` filtered by user_id."""

    TABLE = "chat_messages"

    def __init__(
        self,
        client: Any | None = None,
        max_turns: int | None = None,
        user_id: str | None = None,
    ) -> None:
        self._client = client
        self._max_turns = max_turns
        self._user_id = user_id

    def _db(self) -> Any:
        if self._client is not None:
            return self._client
        return db.get_supabase()

    def _limit(self) -> int:
        if self._max_turns is not None:
            return self._max_turns
        return get_settings().memory_max_turns

    def _resolve_user_id(self, user_id: str | None = None) -> str:
        return user_id or self._user_id or get_current_user_id()

    def get_history(self, session_id: str, user_id: str | None = None) -> list[Turn]:
        """Fetch conversation turns for (user_id, session_id), oldest first."""
        limit = self._limit()
        if limit <= 0:
            return []

        uid = self._resolve_user_id(user_id)
        response = (
            self._db()
            .table(self.TABLE)
            .select("id, user_message, assistant_message, created_at")
            .eq("user_id", uid)
            .eq("session_id", session_id)
            .order("created_at")
            .execute()
        )
        rows = response.data or []
        turns: list[Turn] = [
            {
                "user": row.get("user_message", row.get("user", "")),
                "assistant": row.get("assistant_message", row.get("assistant", "")),
            }
            for row in rows
        ]
        return turns[-limit:]

    def add_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        user_id: str | None = None,
    ) -> None:
        """Insert a turn into `chat_messages` for the authenticated user and prune old turns."""
        uid = self._resolve_user_id(user_id)
        limit = self._limit()
        if limit <= 0:
            self.clear(session_id, user_id=uid)
            return

        client = self._db()
        client.table(self.TABLE).insert(
            {
                "user_id": uid,
                "session_id": session_id,
                "user_message": user_msg,
                "assistant_message": assistant_msg,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ).execute()

        # Keep only the most recent `limit` turns for this user and session.
        existing = (
            client.table(self.TABLE)
            .select("id, created_at")
            .eq("user_id", uid)
            .eq("session_id", session_id)
            .order("created_at")
            .execute()
        )
        rows = existing.data or []
        if len(rows) > limit:
            for stale_row in rows[:-limit]:
                row_id = stale_row.get("id")
                if row_id is not None:
                    client.table(self.TABLE).delete().eq("id", row_id).execute()

    def clear(self, session_id: str, user_id: str | None = None) -> None:
        """Delete all turns for (user_id, session_id) from `chat_messages`."""
        uid = self._resolve_user_id(user_id)
        (
            self._db()
            .table(self.TABLE)
            .delete()
            .eq("user_id", uid)
            .eq("session_id", session_id)
            .execute()
        )

    def clear_all(self) -> None:
        """Clear all stored turns when backed by a test mock client."""
        try:
            client = self._db()
        except RuntimeError:
            return
        if hasattr(client, "reset"):
            client.reset()


_store = SupabaseSessionStore()


def get_memory_store() -> SupabaseSessionStore:
    """Return the shared Supabase session memory store."""
    return _store
