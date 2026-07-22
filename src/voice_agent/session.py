"""In-memory session tracking for the voice-agent demo."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Union
from uuid import UUID, uuid4

DEFAULT_SESSION_TTL = timedelta(minutes=30)
DEMO_SESSION_FALLBACK_MAX_AGE = timedelta(minutes=2)

ConversationId = Union[str, UUID]
SessionMessage = dict[str, Any]


@dataclass
class SessionInfo:
    """State kept for one local voice-agent conversation."""

    conversation_id: str
    perception_session_id: str
    created_at: datetime
    last_seen: datetime
    message_history: list[SessionMessage] = field(default_factory=list)
    english_mode: bool = False
    english_offered: bool = False
    high_hesitation_streak: int = 0

    def touch(self, when: datetime | None = None) -> None:
        """Mark the session as active at ``when`` or now."""
        self.last_seen = _coerce_utc(when or _utc_now())

    def is_expired(
        self,
        *,
        now: datetime | None = None,
        expire_after: timedelta = DEFAULT_SESSION_TTL,
    ) -> bool:
        """Return true if the session has been inactive past ``expire_after``."""
        return _coerce_utc(now or _utc_now()) - self.last_seen >= expire_after


class SessionStore:
    """Small, process-local session store for a single FastAPI worker."""

    def __init__(
        self,
        *,
        expire_after: timedelta = DEFAULT_SESSION_TTL,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if expire_after < timedelta(0):
            raise ValueError("expire_after must not be negative")
        self._expire_after = expire_after
        self._clock = clock or _utc_now
        self._lock = RLock()
        self._sessions: dict[str, SessionInfo] = {}

    @property
    def expire_after(self) -> timedelta:
        """Inactivity window before a session expires."""
        return self._expire_after

    def create(
        self,
        perception_session_id: str,
        *,
        conversation_id: ConversationId | None = None,
        message_history: Iterable[Mapping[str, Any]] | None = None,
    ) -> SessionInfo:
        """Create and store a session keyed by a local conversation UUID."""
        if not perception_session_id.strip():
            raise ValueError("perception_session_id must not be empty")

        resolved_conversation_id = (
            str(uuid4())
            if conversation_id is None
            else _normalize_conversation_id(conversation_id)
        )
        now = self._now()
        session = SessionInfo(
            conversation_id=resolved_conversation_id,
            perception_session_id=perception_session_id.strip(),
            created_at=now,
            last_seen=now,
            message_history=_copy_messages(
                () if message_history is None else message_history
            ),
        )

        with self._lock:
            self.cleanup_expired(now=now)
            if resolved_conversation_id in self._sessions:
                raise ValueError(
                    f"session already exists for conversation_id "
                    f"{resolved_conversation_id}"
                )
            self._sessions[resolved_conversation_id] = session
            return session

    def get(
        self,
        conversation_id: ConversationId,
        *,
        touch: bool = True,
    ) -> SessionInfo | None:
        """Return an active session by conversation ID, or None if missing."""
        now = self._now()
        with self._lock:
            session = self._get_active_locked(conversation_id, now=now)
            if session is not None and touch:
                session.touch(now)
            return session

    def touch(self, conversation_id: ConversationId) -> SessionInfo | None:
        """Refresh ``last_seen`` for an active session."""
        now = self._now()
        with self._lock:
            session = self._get_active_locked(conversation_id, now=now)
            if session is None:
                return None
            session.touch(now)
            return session

    def add_message(
        self,
        conversation_id: ConversationId,
        message: Mapping[str, Any],
    ) -> SessionInfo | None:
        """Append one chat message and refresh the session activity time."""
        now = self._now()
        with self._lock:
            session = self._get_active_locked(conversation_id, now=now)
            if session is None:
                return None
            session.message_history.append(deepcopy(dict(message)))
            session.touch(now)
            return session

    def extend_history(
        self,
        conversation_id: ConversationId,
        messages: Iterable[Mapping[str, Any]],
    ) -> SessionInfo | None:
        """Append multiple chat messages and refresh the session activity time."""
        now = self._now()
        with self._lock:
            session = self._get_active_locked(conversation_id, now=now)
            if session is None:
                return None
            session.message_history.extend(_copy_messages(messages))
            session.touch(now)
            return session

    def update_history(
        self,
        conversation_id: ConversationId,
        message_history: Iterable[Mapping[str, Any]],
    ) -> SessionInfo | None:
        """Replace the stored chat history and refresh the session activity time."""
        now = self._now()
        with self._lock:
            session = self._get_active_locked(conversation_id, now=now)
            if session is None:
                return None
            session.message_history = _copy_messages(message_history)
            session.touch(now)
            return session

    def find_by_perception_id(
        self,
        perception_session_id: str,
        *,
        touch: bool = True,
    ) -> SessionInfo | None:
        """Find the active local session associated with a perception session."""
        if not perception_session_id.strip():
            return None

        now = self._now()
        with self._lock:
            self.cleanup_expired(now=now)
            for session in self._sessions.values():
                if session.perception_session_id == perception_session_id.strip():
                    if touch:
                        session.touch(now)
                    return session
            return None

    def end(self, conversation_id: ConversationId) -> SessionInfo | None:
        """Remove a session and return it if it existed."""
        try:
            normalized_id = _normalize_conversation_id(conversation_id)
        except ValueError:
            return None
        with self._lock:
            return self._sessions.pop(normalized_id, None)

    def delete(self, conversation_id: ConversationId) -> bool:
        """Remove a session and report whether anything was deleted."""
        return self.end(conversation_id) is not None

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        """Delete expired sessions and return the number removed."""
        resolved_now = _coerce_utc(now or self._now())
        with self._lock:
            expired_ids = [
                conversation_id
                for conversation_id, session in self._sessions.items()
                if session.is_expired(
                    now=resolved_now,
                    expire_after=self._expire_after,
                )
            ]
            for conversation_id in expired_ids:
                del self._sessions[conversation_id]
            return len(expired_ids)

    def active_count(self) -> int:
        """Return the number of currently active sessions after cleanup."""
        with self._lock:
            self.cleanup_expired()
            return len(self._sessions)

    def get_unambiguous_active(self, *, touch: bool = True) -> SessionInfo | None:
        """Return the only active session, or None when missing or ambiguous."""
        now = self._now()
        with self._lock:
            self.cleanup_expired(now=now)
            if len(self._sessions) != 1:
                return None
            session = next(iter(self._sessions.values()))
            if touch:
                session.touch(now)
            return session

    def get_latest_active(
        self,
        *,
        touch: bool = True,
        max_age: timedelta | None = None,
    ) -> SessionInfo | None:
        """Return the freshest active session, optionally bounded by age."""
        if max_age is not None and max_age < timedelta(0):
            raise ValueError("max_age must not be negative")

        now = self._now()
        with self._lock:
            self.cleanup_expired(now=now)
            if not self._sessions:
                return None
            session = max(
                self._sessions.values(),
                key=lambda candidate: (candidate.last_seen, candidate.created_at),
            )
            if max_age is not None and now - session.last_seen > max_age:
                return None
            if touch:
                session.touch(now)
            return session

    def clear(self) -> None:
        """Remove all sessions. Intended for tests and local demos."""
        with self._lock:
            self._sessions.clear()

    def _get_active_locked(
        self,
        conversation_id: ConversationId,
        *,
        now: datetime,
    ) -> SessionInfo | None:
        try:
            normalized_id = _normalize_conversation_id(conversation_id)
        except ValueError:
            return None
        session = self._sessions.get(normalized_id)
        if session is None:
            return None
        if session.is_expired(now=now, expire_after=self._expire_after):
            del self._sessions[normalized_id]
            return None
        return session

    def _now(self) -> datetime:
        return _coerce_utc(self._clock())


_ENGLISH_REQUEST_PHRASES = (
    # Bare tokens (catch most phrasings) - kept intentionally.
    "english",
    "englisch",
    # English phrasings.
    "in english",
    "speak english",
    "switch to english",
    "can we speak english",
    "can you speak english",
    "could you speak english",
    "let's speak english",
    "speak in english",
    "talk in english",
    "continue in english",
    "english please",
    "in english please",
    "let us continue in english",
    # German phrasings.
    "auf englisch",
    "sprechen sie englisch",
    "sprechen sie english",
    "koennen wir englisch",
    "können wir englisch",
    "auf englisch bitte",
    "englisch bitte",
    "reden wir englisch",
    "lass uns englisch",
    "wechseln wir zu englisch",
    "auf englisch wechseln",
    "auf englisch umschalten",
    "auf englisch weitermachen",
)
_ENGLISH_ACCEPT_WORDS = (
    "yes",
    "ja",
    "please",
    "bitte",
    "okay",
    "ok",
    "sure",
    "gerne",
)


def update_language_state(
    session: SessionInfo,
    perception_state: dict,
    last_user_message: str,
) -> dict:
    """Advance the sticky English-mode state machine for one turn.

    Mutates ``session`` in place (streak counter and mode booleans) and returns
    a small dict describing the language mode the webhook should build a prompt
    block from.
    """
    hesitation = _coerce_hesitation(perception_state.get("hesitation_score"))
    if hesitation > 0.75:
        session.high_hesitation_streak += 1
    else:
        session.high_hesitation_streak = 0

    lowered = last_user_message.lower()

    if any(phrase in lowered for phrase in _ENGLISH_REQUEST_PHRASES):
        session.english_mode = True
        session.english_offered = True
        return {"mode": "english_locked", "just_switched": True}

    if session.english_offered and not session.english_mode:
        if any(word in lowered for word in _ENGLISH_ACCEPT_WORDS):
            session.english_mode = True
            return {"mode": "english_locked", "just_switched": True}
        return {"mode": "german_locked", "just_switched": False}

    should_offer = (
        not session.english_offered
        and not session.english_mode
        and (
            session.high_hesitation_streak >= 2
            or hesitation > 0.9
        )
    )
    if should_offer:
        session.english_offered = True
        return {"mode": "offer_english", "just_switched": False}

    if session.english_mode:
        return {"mode": "english", "just_switched": False}
    return {"mode": "german", "just_switched": False}


def _coerce_hesitation(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _copy_messages(messages: Iterable[Mapping[str, Any]]) -> list[SessionMessage]:
    return [deepcopy(dict(message)) for message in messages]


def _normalize_conversation_id(conversation_id: ConversationId) -> str:
    try:
        return str(UUID(str(conversation_id)))
    except ValueError as exc:
        raise ValueError("conversation_id must be a UUID") from exc


def _coerce_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


session_store = SessionStore()
