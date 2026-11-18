"""Session conversation memory store (in-memory for now, replaceable by Supabase in Phase 7)."""

from abc import ABC, abstractmethod

from app.config import get_settings

Turn = dict[str, str]


class BaseMemoryStore(ABC):
    """Interface for storing per-session user/assistant turns."""

    @abstractmethod
    def get_history(self, session_id: str) -> list[Turn]:
        """Return stored turns for a session, oldest first."""

    @abstractmethod
    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Append one (user, assistant) turn and truncate to the max turn limit."""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Delete all turns for a session."""


class InMemorySessionStore(BaseMemoryStore):
    """In-process dict session store keeping only the last MEMORY_MAX_TURNS turns."""

    def __init__(self, max_turns: int | None = None) -> None:
        self._max_turns = max_turns
        self._sessions: dict[str, list[Turn]] = {}

    def _limit(self) -> int:
        if self._max_turns is not None:
            return self._max_turns
        return get_settings().memory_max_turns

    def get_history(self, session_id: str) -> list[Turn]:
        return list(self._sessions.get(session_id, []))

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        turns = self._sessions.setdefault(session_id, [])
        turns.append({"user": user_msg, "assistant": assistant_msg})
        limit = self._limit()
        if limit <= 0:
            self._sessions[session_id] = []
        elif len(turns) > limit:
            self._sessions[session_id] = turns[-limit:]

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear_all(self) -> None:
        """Clear every session (used to isolate tests)."""
        self._sessions.clear()


_store = InMemorySessionStore()


def get_memory_store() -> InMemorySessionStore:
    """Return the shared session memory store."""
    return _store
