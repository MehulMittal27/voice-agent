"""Async HTTP client for the companion voice-perception service."""

from __future__ import annotations

import copy
import logging
from typing import Any
from urllib.parse import quote

import httpx

from voice_agent.config import get_settings

logger = logging.getLogger(__name__)

STATE_TIMEOUT_SECONDS = 0.1
REQUEST_TIMEOUT_SECONDS = 5.0

DEFAULT_NEUTRAL_STATE: dict[str, Any] = {
    "emotion": "NEUTRAL",
    "emotion_confidence": 0.0,
    "stability": "stable",
    "audio_events": [],
    "hesitation_score": 0.0,
}


class PerceptionClientError(RuntimeError):
    """Raised when a required perception service operation fails."""


class PerceptionClient:
    """Small async client for voice-perception HTTP endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        request_timeout: float = REQUEST_TIMEOUT_SECONDS,
        state_timeout: float = STATE_TIMEOUT_SECONDS,
    ) -> None:
        resolved_base_url = base_url or get_settings().voice_perception_url
        self.base_url = resolved_base_url.rstrip("/")
        self.request_timeout = request_timeout
        self.state_timeout = state_timeout

    async def start_session(self, language: str) -> str:
        """Start a voice-perception session and return its session ID."""
        normalized_language = language.strip().lower()
        if not normalized_language:
            raise ValueError("language is required")

        try:
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.post(
                    self._url("/session/start"),
                    json={"language": normalized_language},
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise PerceptionClientError(
                f"Could not start perception session at {self.base_url}"
            ) from exc
        except ValueError as exc:
            raise PerceptionClientError(
                "Perception service returned invalid JSON for session start"
            ) from exc

        session_id = self._extract_session_id(payload)
        if session_id is None:
            raise PerceptionClientError(
                "Perception service session start response did not include a session ID"
            )

        logger.info("Started perception session")
        return session_id

    async def get_state(self, session_id: str) -> dict[str, Any]:
        """Return perception state, falling back to neutral state on any error."""
        if not session_id.strip():
            logger.warning("Cannot fetch perception state without a session ID")
            return neutral_state()

        try:
            async with httpx.AsyncClient(timeout=self.state_timeout) as client:
                response = await client.get(
                    self._url(f"/state/{quote(session_id, safe='')}")
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.TimeoutException:
            logger.warning(
                "Timed out fetching perception state for session %s after %.0fms",
                session_id,
                self.state_timeout * 1000,
            )
            return neutral_state()
        except httpx.HTTPError as exc:
            logger.warning(
                "Could not fetch perception state for session %s: %s",
                session_id,
                exc,
            )
            return neutral_state()
        except ValueError as exc:
            logger.warning(
                "Perception service returned invalid JSON for session %s: %s",
                session_id,
                exc,
            )
            return neutral_state()
        except Exception as exc:
            logger.warning(
                "Unexpected error fetching perception state for session %s: %s",
                session_id,
                exc,
            )
            return neutral_state()

        if not isinstance(payload, dict):
            logger.warning(
                "Perception service returned non-object state for session %s",
                session_id,
            )
            return neutral_state()

        return _merge_with_neutral_state(payload)

    async def end_session(self, session_id: str) -> None:
        """End a voice-perception session, logging and swallowing failures."""
        if not session_id.strip():
            logger.warning("Cannot end perception session without a session ID")
            return

        try:
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.post(
                    self._url(f"/session/{quote(session_id, safe='')}/end")
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "Could not end perception session %s: %s",
                session_id,
                exc,
            )
            return
        except Exception as exc:
            logger.warning(
                "Unexpected error ending perception session %s: %s",
                session_id,
                exc,
            )
            return

        logger.info("Ended perception session")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _extract_session_id(payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None
        for key in ("perception_session_id", "session_id", "id"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None


def neutral_state() -> dict[str, Any]:
    """Return a fresh copy of the default neutral perception state."""
    return copy.deepcopy(DEFAULT_NEUTRAL_STATE)


def _merge_with_neutral_state(payload: dict[str, Any]) -> dict[str, Any]:
    raw_state = payload.get("state")
    if isinstance(raw_state, dict):
        state = raw_state
    else:
        state = payload

    merged = neutral_state()
    merged.update(state)
    return merged


async def start_session(language: str) -> str:
    """Start a voice-perception session using configured settings."""
    return await PerceptionClient().start_session(language)


async def get_state(session_id: str) -> dict[str, Any]:
    """Fetch perception state using configured settings."""
    return await PerceptionClient().get_state(session_id)


async def end_session(session_id: str) -> None:
    """End a voice-perception session using configured settings."""
    await PerceptionClient().end_session(session_id)


__all__ = [
    "DEFAULT_NEUTRAL_STATE",
    "PerceptionClient",
    "PerceptionClientError",
    "end_session",
    "get_state",
    "neutral_state",
    "start_session",
]
