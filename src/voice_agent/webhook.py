"""OpenAI-compatible ElevenLabs webhook routes."""

from __future__ import annotations

import inspect
import json
import logging
import re
from datetime import datetime, timezone
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from voice_agent import clerk
from voice_agent.logging_config import REDACTION_TEXT, redact_secrets
from voice_agent.perception_client import PerceptionClient, neutral_state
from voice_agent.session import (
    DEMO_SESSION_FALLBACK_MAX_AGE,
    SessionInfo,
    session_store,
    update_language_state,
)
from voice_agent.streaming import openai_to_openai_sse

logger = logging.getLogger(__name__)

router = APIRouter()

SSE_HEADERS = {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

_SESSION_ID_KEYS = (
    "perception_session_id",
    "perceptionSessionId",
    "voice_perception_session_id",
    "voicePerceptionSessionId",
)
_DYNAMIC_VARIABLE_CONTAINER_KEYS = (
    "conversation_initiation_client_data",
    "conversationInitiationClientData",
    "conversation_initiation_clientData",
)
_EXTRA_BODY_KEYS = (
    "elevenlabs_extra_body",
    "extra_body",
    "extraBody",
)
_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "client_secret",
    "password",
    "secret",
    "token",
    "x-api-key",
    "xi-api-key",
)
_SYSTEM_MESSAGE_SESSION_ID_RE = re.compile(
    r"\bperception_session_id\s*[:=]\s*([^\s,;\"'`<>{}]+)",
    re.IGNORECASE,
)

# id_source strings returned by _resolve_perception_session_id when the session
# was guessed from local state rather than carried by the request.
_DEMO_FALLBACK_ID_SOURCES = frozenset(
    {
        "single active local session fallback",
        "latest active demo session fallback",
    }
)


def _transient_session() -> SessionInfo:
    """Build an unstored session so the language state machine still runs.

    Used only when no local session was resolved for this webhook turn. It is
    never persisted, so streak state cannot carry across turns in that path.
    """
    now = datetime.now(timezone.utc)
    return SessionInfo(
        conversation_id="",
        perception_session_id="",
        created_at=now,
        last_seen=now,
    )


def _correlation_mode(id_source: str) -> str:
    """Map an id_source string to a human-readable LLM-turn correlation mode.

    Direct dynamic-variable / explicit placements (top-level, extra-body,
    dynamic-variable containers, the system-message marker, and user-message
    metadata) all mean the request itself carried the id. The demo fallbacks
    mean we guessed from local session state. Unknown sources are named
    verbatim rather than silently mislabelled.
    """
    if id_source in _DEMO_FALLBACK_ID_SOURCES:
        return "latest active demo fallback"
    if id_source == "missing":
        return "none resolved"
    return "direct dynamic variable"


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> StreamingResponse:
    """Handle ElevenLabs Custom LLM calls and stream OpenAI-format SSE."""
    logger.info("LLM turn started: POST /v1/chat/completions")
    body = await _read_json_body(request)
    messages = _extract_messages(body)
    perception_session_id, id_source = _resolve_perception_session_id(body)

    if perception_session_id:
        logger.info("Resolved perception_session_id from %s", id_source)
    else:
        logger.warning("No perception_session_id found in webhook request body")

    session = _touch_or_create_session(perception_session_id)
    _log_first_session_body(session, body, id_source)
    _store_latest_messages(session, messages)

    perception_state = await _fetch_perception_state(request, perception_session_id)
    logger.info(
        "Perception state for session %s: %s",
        perception_session_id or "missing",
        redact_secrets(perception_state),
    )

    _log_injected_perception(perception_session_id, id_source, perception_state)

    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_msg = m.get("content", "") or ""
            break
    if not isinstance(last_user_msg, str):
        last_user_msg = ""

    language_session = session if session is not None else _transient_session()
    language_state = update_language_state(
        session=language_session,
        perception_state=perception_state,
        last_user_message=last_user_msg,
    )
    logger.info(
        "language_state session=%s mode=%s just_switched=%s streak=%d",
        perception_session_id or "missing",
        language_state["mode"],
        language_state["just_switched"],
        language_session.high_hesitation_streak,
    )

    text_stream = clerk.run_turn(messages, perception_state, language_state)
    sse_stream = openai_to_openai_sse(_ensure_async_text_stream(text_stream))
    return StreamingResponse(
        sse_stream,
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


def _log_injected_perception(
    perception_session_id: str | None,
    id_source: str,
    perception_state: Mapping[str, Any],
) -> None:
    """Log exactly what perception state is injected into this LLM turn."""
    if perception_session_id:
        logger.info(
            "LLM turn perception correlation: %s, session=%s",
            _correlation_mode(id_source),
            perception_session_id,
        )
    else:
        logger.warning(
            "LLM turn injected neutral fallback because no perception session "
            "was resolved (id_source=%s)",
            id_source,
        )

    logger.info(
        "LLM turn injected perception state: emotion=%s, confidence=%s, "
        "stability=%s, hesitation=%s, events=%s",
        perception_state.get("emotion"),
        perception_state.get("emotion_confidence"),
        perception_state.get("stability"),
        perception_state.get("hesitation_score"),
        perception_state.get("audio_events"),
    )


async def _read_json_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return payload


def _extract_messages(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list")

    messages: list[dict[str, Any]] = []
    for index, message in enumerate(raw_messages):
        if not isinstance(message, Mapping):
            raise HTTPException(
                status_code=400,
                detail=f"messages[{index}] must be a JSON object",
            )
        messages.append(dict(message))
    return messages


def extract_perception_session_id(body: Mapping[str, Any]) -> tuple[str | None, str]:
    """Extract the perception session ID from known ElevenLabs placements."""
    direct_value = _session_id_from_mapping(body)
    if direct_value is not None:
        return direct_value, "top-level perception_session_id"

    for extra_body_key in _EXTRA_BODY_KEYS:
        extra_body = _as_mapping(body.get(extra_body_key))
        if extra_body is None:
            continue
        extra_value, extra_source = _session_id_from_mapping_with_source(
            extra_body,
            extra_body_key,
        )
        if extra_value is not None:
            return extra_value, extra_source

    for container_key in _DYNAMIC_VARIABLE_CONTAINER_KEYS:
        container = _as_mapping(body.get(container_key))
        if container is None:
            continue
        container_value, container_source = _session_id_from_mapping_with_source(
            container,
            container_key,
        )
        if container_value is not None:
            return container_value, container_source

    system_message_value, system_message_source = _session_id_from_system_messages(
        body.get("messages")
    )
    if system_message_value is not None:
        return system_message_value, system_message_source

    last_user_metadata = _last_user_message_metadata(body.get("messages"))
    if last_user_metadata is not None:
        metadata_value, metadata_source = _session_id_from_mapping_with_source(
            last_user_metadata,
            "last user message metadata",
        )
        if metadata_value is not None:
            return metadata_value, metadata_source

    return None, "missing"


def _resolve_perception_session_id(body: Mapping[str, Any]) -> tuple[str | None, str]:
    perception_session_id, id_source = extract_perception_session_id(body)
    if perception_session_id is not None:
        return perception_session_id, id_source

    fallback_session = session_store.get_unambiguous_active()
    if fallback_session is not None:
        logger.warning(
            "No perception_session_id in webhook body; using only active local "
            "session %s with perception session %s as demo fallback",
            fallback_session.conversation_id,
            fallback_session.perception_session_id,
        )
        return fallback_session.perception_session_id, "single active local session fallback"

    active_count = session_store.active_count()
    if active_count > 1:
        fallback_session = session_store.get_latest_active(
            max_age=DEMO_SESSION_FALLBACK_MAX_AGE,
        )
        if fallback_session is not None:
            logger.warning(
                "No perception_session_id in webhook body; using latest active "
                "demo session %s",
                fallback_session.perception_session_id,
            )
            return (
                fallback_session.perception_session_id,
                "latest active demo session fallback",
            )
        logger.warning(
            "No perception_session_id in webhook body and %d active local sessions, "
            "but none touched in the last %.0f seconds; not guessing",
            active_count,
            DEMO_SESSION_FALLBACK_MAX_AGE.total_seconds(),
        )
    return None, "missing"


def _session_id_from_mapping(mapping: Mapping[str, Any]) -> str | None:
    value, _source = _session_id_from_mapping_with_source(mapping, "")
    return value


def _session_id_from_mapping_with_source(
    mapping: Mapping[str, Any],
    source_prefix: str,
) -> tuple[str | None, str]:
    for key in _SESSION_ID_KEYS:
        value = _nonempty_string(mapping.get(key))
        if value is not None:
            source = f"{source_prefix}.{key}" if source_prefix else key
            return value, source

    for nested_key in ("dynamic_variables", "dynamicVariables"):
        nested = _as_mapping(mapping.get(nested_key))
        if nested is None:
            continue
        for key in _SESSION_ID_KEYS:
            value = _nonempty_string(nested.get(key))
            if value is not None:
                prefix = f"{source_prefix}.{nested_key}" if source_prefix else nested_key
                return value, f"{prefix}.{key}"

    return None, "missing"


def _session_id_from_system_messages(raw_messages: Any) -> tuple[str | None, str]:
    if not isinstance(raw_messages, Sequence) or isinstance(
        raw_messages,
        (str, bytes, bytearray),
    ):
        return None, "missing"

    for index, message in enumerate(raw_messages):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role", "")).strip().lower() != "system":
            continue
        content_text = _message_content_to_text(message.get("content"))
        match = _SYSTEM_MESSAGE_SESSION_ID_RE.search(content_text)
        if match is None:
            continue
        value = match.group(1).strip().rstrip(".)]")
        if value:
            return value, f"messages[{index}].system content perception_session_id marker"
    return None, "missing"


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(
        content,
        (str, bytes, bytearray),
    ):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        for key in ("text", "content"):
            value = item.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
                break
    return "\n".join(parts)


def _last_user_message_metadata(raw_messages: Any) -> Mapping[str, Any] | None:
    if not isinstance(raw_messages, Sequence) or isinstance(
        raw_messages,
        (str, bytes, bytearray),
    ):
        return None

    for message in reversed(raw_messages):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role", "")).strip().lower() != "user":
            continue
        for metadata_key in ("metadata", "extra_body", "extraBody"):
            metadata = _as_mapping(message.get(metadata_key))
            if metadata is not None:
                return metadata
        return None
    return None


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        if isinstance(parsed, Mapping):
            return parsed
    return None


def _nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _touch_or_create_session(perception_session_id: str | None) -> SessionInfo | None:
    if not perception_session_id:
        return None

    session = session_store.find_by_perception_id(perception_session_id)
    if session is not None:
        return session

    try:
        return session_store.create(perception_session_id)
    except ValueError:
        logger.debug("Could not create local session for webhook request", exc_info=True)
        return session_store.find_by_perception_id(perception_session_id)


def _log_first_session_body(
    session: SessionInfo | None,
    body: Mapping[str, Any],
    id_source: str,
) -> None:
    if session is None:
        logger.info(
            "Incoming webhook body without a local session, id_source=%s: %s",
            id_source,
            _json_for_log(body),
        )
        return

    if session.message_history:
        return

    logger.info(
        "First webhook body for perception session %s, id_source=%s: %s",
        session.perception_session_id,
        id_source,
        _json_for_log(body),
    )


def _store_latest_messages(
    session: SessionInfo | None,
    messages: Sequence[Mapping[str, Any]],
) -> None:
    if session is None:
        return
    session_store.update_history(session.conversation_id, messages)


async def _fetch_perception_state(
    request: Request,
    perception_session_id: str | None,
) -> dict[str, Any]:
    if not perception_session_id:
        return neutral_state()

    client = _get_perception_client(request)
    try:
        state_result = client.get_state(perception_session_id)
        if inspect.isawaitable(state_result):
            state = await state_result
        else:
            state = state_result
    except Exception as exc:
        logger.warning(
            "Perception state lookup failed for session %s: %s",
            perception_session_id,
            exc,
        )
        return neutral_state()

    if isinstance(state, dict):
        return state

    logger.warning(
        "Perception state lookup returned non-object state for session %s",
        perception_session_id,
    )
    return neutral_state()


def _get_perception_client(request: Request) -> PerceptionClient:
    client = getattr(request.app.state, "perception_client", None)
    if client is not None:
        return client
    return PerceptionClient()


async def _ensure_async_text_stream(stream: Any) -> AsyncIterator[str]:
    if inspect.isawaitable(stream):
        stream = await stream

    if hasattr(stream, "__aiter__"):
        async for text_delta in stream:
            if text_delta:
                yield str(text_delta)
        return

    if isinstance(stream, str):
        if stream:
            yield stream
        return

    if isinstance(stream, Sequence):
        for text_delta in stream:
            if text_delta:
                yield str(text_delta)
        return

    raise TypeError("clerk.run_turn must return an async text stream")


def _json_for_log(value: Any) -> str:
    return json.dumps(
        _redact_for_log(value),
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )


def _redact_for_log(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_secret_key(key_text):
                redacted[key_text] = REDACTION_TEXT
            else:
                redacted[key_text] = _redact_for_log(item)
        return redact_secrets(redacted)

    if isinstance(value, list):
        return [(_redact_for_log(item)) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_for_log(item) for item in value)
    return redact_secrets(value)


def _is_secret_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(part.replace("-", "_") in normalized for part in _SECRET_KEY_PARTS)


__all__ = [
    "SSE_HEADERS",
    "chat_completions",
    "extract_perception_session_id",
    "router",
]
