#!/usr/bin/env python3
"""Smoke test the local OpenAI-compatible ElevenLabs webhook."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voice_agent.config import load_settings

DEFAULT_BASE_URL = "http://127.0.0.1:8001"
BASE_URL_ENV_VARS = ("VOICE_AGENT_URL", "VOICE_AGENT_BASE_URL")
DEFAULT_USER_MESSAGE = (
    "I trained as a nurse in India and I live in Nuremberg. "
    "What should I do first for recognition in Germany?"
)
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]+"),
    re.compile(r"xi-[A-Za-z0-9_\-]+"),
)


class WebhookSmokeTestError(RuntimeError):
    """Raised for expected smoke test failures with user-facing messages."""


async def run(args: argparse.Namespace) -> int:
    settings = load_settings(PROJECT_ROOT / ".env")
    if _missing_openai_key(settings.openai_api_key):
        raise WebhookSmokeTestError(
            "OPENAI_API_KEY is not configured. Set it in .env or the environment "
            "before running this live webhook smoke test. ELEVENLABS_AGENT_ID is "
            "not required because this script calls the webhook directly."
        )

    base_url = _normalize_base_url(args.base_url)
    timeout = httpx.Timeout(args.timeout, connect=min(5.0, args.timeout))

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        health = await _probe_health(client, base_url)
        _print_json("health", _safe_health_payload(health))

        session = await _start_session(client, base_url)
        _print_json("session", _safe_session_payload(session))

        perception_session_id = _require_string(
            session,
            "perception_session_id",
            "Session start response did not include perception_session_id.",
        )
        print(
            "session_start_note: /session/start used the server-configured "
            "PERCEPTION_LANGUAGE; no caller language was sent by this script."
        )

        request_payload = _build_webhook_payload(
            perception_session_id=perception_session_id,
            user_message=args.message,
        )
        print(f"webhook_url: {_safe_url(urljoin(base_url, '/v1/chat/completions'))}")
        print("streaming_sse:")
        german_text = await _stream_webhook(client, request_payload, base_url)

    if not german_text.strip():
        raise WebhookSmokeTestError(
            "Webhook completed but produced no text deltas. Check the voice-agent "
            "server logs for OpenAI or streaming errors."
        )

    print("\nreassembled_german_text:")
    print(german_text)
    return 0


async def _probe_health(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    try:
        response = await client.get("/health")
    except httpx.HTTPError as exc:
        raise _server_unreachable_error(base_url, exc) from exc

    if response.status_code != httpx.codes.OK:
        raise WebhookSmokeTestError(
            "voice-agent /health returned "
            f"HTTP {response.status_code}: {_safe_text(response.text)}"
        )
    return _json_object(response, "/health")


async def _start_session(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    try:
        response = await client.post("/session/start", json={})
    except httpx.HTTPError as exc:
        raise _server_unreachable_error(base_url, exc) from exc

    if response.status_code != httpx.codes.OK:
        raise WebhookSmokeTestError(
            "voice-agent /session/start returned "
            f"HTTP {response.status_code}: {_safe_text(response.text)}"
        )
    return _json_object(response, "/session/start")


async def _stream_webhook(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    base_url: str,
) -> str:
    text_parts: list[str] = []
    saw_done = False

    try:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code != httpx.codes.OK:
                body = (await response.aread()).decode(errors="replace")
                raise WebhookSmokeTestError(
                    "voice-agent /v1/chat/completions returned "
                    f"HTTP {response.status_code}: {_safe_text(body)}"
                )

            async for event_data in _iter_sse_data(response):
                if event_data == "[DONE]":
                    saw_done = True
                    continue
                text_parts.extend(_content_deltas(event_data))
    except httpx.HTTPError as exc:
        raise _server_unreachable_error(base_url, exc) from exc

    if not saw_done:
        raise WebhookSmokeTestError(
            "Webhook stream ended before data: [DONE]. Check the voice-agent "
            "server logs. If the stream failed early, confirm OPENAI_API_KEY is "
            "set in the server environment and OPENAI_MODEL is valid."
        )

    return "".join(text_parts)


async def _iter_sse_data(response: httpx.Response) -> AsyncIterator[str]:
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            print(flush=True)
            continue

        print(line, flush=True)
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").lstrip(" "))

    if data_lines:
        yield "\n".join(data_lines)


def _content_deltas(event_data: str) -> list[str]:
    try:
        payload = json.loads(event_data)
    except json.JSONDecodeError as exc:
        raise WebhookSmokeTestError(
            f"Received non-JSON SSE data chunk: {_safe_text(event_data)}"
        ) from exc

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return []

    deltas: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            deltas.append(content)
    return deltas


def _build_webhook_payload(
    *,
    perception_session_id: str,
    user_message: str,
) -> dict[str, Any]:
    return {
        "model": "custom",
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": "You are the ElevenLabs Custom LLM webhook test caller.",
            },
            {"role": "user", "content": user_message},
        ],
        "perception_session_id": perception_session_id,
    }


def _normalize_base_url(raw_base_url: str) -> str:
    trimmed = raw_base_url.strip().rstrip("/")
    if not trimmed:
        trimmed = DEFAULT_BASE_URL
    if "://" not in trimmed:
        trimmed = f"http://{trimmed}"

    parsed = urlparse(trimmed)
    if not parsed.scheme or not parsed.netloc:
        raise WebhookSmokeTestError(f"Invalid voice-agent base URL: {raw_base_url!r}")
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))


def _server_unreachable_error(base_url: str, exc: httpx.HTTPError) -> WebhookSmokeTestError:
    return WebhookSmokeTestError(
        f"voice-agent server is not reachable at {_safe_url(base_url)}: {_safe_text(str(exc))}. "
        "Start the server first, for example: "
        "PYTHONPATH=src uvicorn voice_agent.main:app --host 127.0.0.1 --port 8001"
    )


def _safe_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.hostname:
        return _safe_text(url)

    host = parsed.hostname
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    if parsed.username or parsed.password:
        host = f"[REDACTED]@{host}"
    return urlunparse((parsed.scheme, host, parsed.path, "", "", ""))


def _json_object(response: httpx.Response, endpoint: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise WebhookSmokeTestError(
            f"voice-agent {endpoint} returned non-JSON: {_safe_text(response.text)}"
        ) from exc

    if not isinstance(payload, dict):
        raise WebhookSmokeTestError(f"voice-agent {endpoint} returned a non-object JSON body")
    return payload


def _safe_health_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "status",
        "perception_reachable",
        "voice_perception_url",
        "perception_language",
        "openai_model",
    }
    return {key: payload[key] for key in allowed_keys if key in payload}


def _safe_session_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "conversation_id",
        "perception_session_id",
        "elevenlabs_agent_id",
        "voice_perception_url",
        "perception_state_url",
        "perception_audio_ws_url",
        "perception_reachable",
        "perception_language",
        "warning",
    }
    return {key: payload[key] for key in allowed_keys if key in payload}


def _require_string(payload: dict[str, Any], key: str, error_message: str) -> str:
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise WebhookSmokeTestError(error_message)


def _print_json(label: str, payload: dict[str, Any]) -> None:
    print(f"{label}: {json.dumps(payload, ensure_ascii=False, sort_keys=True)}")


def _safe_text(text: str, limit: int = 1000) -> str:
    sanitized = text.replace("\n", " ").replace("\r", " ").strip()
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    if len(sanitized) > limit:
        return f"{sanitized[:limit]}..."
    return sanitized


def _missing_openai_key(value: str) -> bool:
    stripped = value.strip()
    return not stripped or stripped == "sk-your-openai-api-key" or stripped.startswith("sk-your-")


def _env_default_base_url() -> str:
    for name in BASE_URL_ENV_VARS:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return DEFAULT_BASE_URL


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start a voice-agent session and stream a fake ElevenLabs Custom LLM "
            "request through /v1/chat/completions."
        )
    )
    parser.add_argument(
        "base_url",
        nargs="?",
        default=_env_default_base_url(),
        help=(
            "voice-agent base URL. Defaults to VOICE_AGENT_URL, then "
            "VOICE_AGENT_BASE_URL, then http://127.0.0.1:8001."
        ),
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_USER_MESSAGE,
        help="Sample English user message to send to the webhook.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="HTTP timeout in seconds for server and OpenAI-backed streaming calls.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return asyncio.run(run(args))
    except WebhookSmokeTestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
