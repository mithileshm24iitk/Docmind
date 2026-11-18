"""Session conversation memory store backed by the Supabase `chat_messages` table.

Every read and write takes the authenticated user's id explicitly. There is no
"current user" global or "anonymous" default: /chat/stream runs its generator
in worker threads, and each step gets a fresh copy of the request context, so a
context variable set inside it was lost and turns were saved as "anonymous"
(a foreign-key violation on chat_messages.user_id).
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from app import db
from app.config import get_settings

Turn = dict[str, str]


def _require_user_id(user_id: str) -> str:
    """Refuse to touch chat_messages without a real user id."""
    if not user_id:
        raise ValueError("user_id is required for chat memory")
    return user_id


class BaseMemoryStore(ABC):
    """Interface for storing per-session user/assistant turns."""

    @abstractmethod
    def get_history(self, session_id: str, *, user_id: str) -> list[Turn]:
        """Return stored turns for a session, oldest first."""

    @abstractmethod
    def add_turn(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        *,
        user_id: str,
    ) -> None:
        """Append one (user, assistant) turn and truncate to the max turn limit."""

    @abstractmethod
    def clear(self, session_id: str, *, user_id: str) -> None:
        """Delete all turns for a session."""


class SupabaseSessionStore(BaseMemoryStore):
    """Supabase-backed session store reading/writing `chat_messages` filtered by user_id."""

    TABLE = "chat_messages"

    def __init__(self, client: Any | None = None, max_turns: int | None = None) -> None:
        self._client = client
        self._max_turns = max_turns

    def _db(self) -> Any:
        if self._client is not None:
            return self._client
        return db.get_supabase()

    def _limit(self) -> int:
        if self._max_turns is not None:
            return self._max_turns
        return get_settings().memory_max_turns

    def get_history(self, session_id: str, *, user_id: str) -> list[Turn]:
        """Fetch conversation turns for (user_id, session_id), oldest first."""
        limit = self._limit()
        if limit <= 0:
            return []

        uid = _require_user_id(user_id)
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
        *,
        user_id: str,
    ) -> None:
        """Insert a turn into `chat_messages` for the authenticated user and prune old turns."""
        uid = _require_user_id(user_id)
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

    def clear(self, session_id: str, *, user_id: str) -> None:
        """Delete all turns for (user_id, session_id) from `chat_messages`."""
        uid = _require_user_id(user_id)
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
