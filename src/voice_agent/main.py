"""FastAPI application entrypoint for the voice-agent service."""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from voice_agent.config import Settings, get_settings
from voice_agent.logging_config import configure_logging
from voice_agent.perception_client import (
    PerceptionClient,
    PerceptionClientError,
    neutral_state,
)
from voice_agent.session import SessionInfo, session_store
from voice_agent.webhook import router as webhook_router

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_INDEX_PATH = PROJECT_ROOT / "static" / "index.html"

CORS_ORIGIN_REGEX = "|".join(
    (
        r"^https?://localhost(:\d+)?$",
        r"^https?://127\.0\.0\.1(:\d+)?$",
        r"^https?://\[::1\](:\d+)?$",
        r"^https://[a-zA-Z0-9-]+\.ngrok-free\.app$",
        r"^https://[a-zA-Z0-9-]+\.ngrok\.io$",
        r"^https://[a-zA-Z0-9-]+\.trycloudflare\.com$",
    )
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI app."""
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Voice Agent ready on port %s. Perception at %s, language=%s. "
            "OpenAI model=%s.",
            resolved_settings.port,
            resolved_settings.voice_perception_url,
            resolved_settings.perception_language,
            resolved_settings.openai_model,
        )
        yield

    app = FastAPI(
        title="Voice Agent Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.perception_client = PerceptionClient(resolved_settings.voice_perception_url)

    _configure_cors(app)
    _register_routes(app)
    app.include_router(webhook_router)
    return app


def _configure_cors(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_origin_regex=CORS_ORIGIN_REGEX,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _register_routes(app: FastAPI) -> None:
    @app.post("/session/start")
    async def start_session(request: Request) -> dict[str, Any]:
        settings = _settings(request)
        client = _perception_client(request)

        perception_reachable = True
        warning: str | None = None
        try:
            perception_session_id = await _maybe_await(
                client.start_session(settings.perception_language)
            )
        except (PerceptionClientError, ValueError) as exc:
            perception_reachable = False
            perception_session_id = str(uuid4())
            warning = "Perception service unavailable; using neutral fallback state."
            logger.warning(
                "Could not start perception session, using fallback session ID: %s",
                exc,
            )

        session = _create_or_reuse_local_session(perception_session_id)
        response = _session_start_payload(
            settings=settings,
            session=session,
            perception_reachable=perception_reachable,
            warning=warning,
        )
        return response

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        settings = _settings(request)
        perception_reachable = await _perception_reachable(_perception_client(request))
        return {
            "status": "ok",
            "perception_reachable": perception_reachable,
            "voice_perception_url": settings.voice_perception_url,
            "perception_language": settings.perception_language,
            "openai_model": settings.openai_model,
        }

    @app.get("/perception/state/{perception_session_id}")
    async def perception_state(
        perception_session_id: str,
        request: Request,
    ) -> dict[str, Any]:
        state = await _maybe_await(
            _perception_client(request).get_state(perception_session_id)
        )
        if not isinstance(state, dict):
            logger.warning(
                "Perception state proxy returned non-object state for session %s",
                perception_session_id,
            )
            state = neutral_state()
        return {"state": state}

    @app.websocket("/perception/audio/{perception_session_id}")
    async def perception_audio(
        perception_session_id: str,
        websocket: WebSocket,
    ) -> None:
        await websocket.accept()
        target_url = _perception_audio_url(
            _settings(websocket).voice_perception_url,
            perception_session_id,
        )
        try:
            await _proxy_audio_websocket(websocket, target_url)
        except Exception as exc:
            logger.warning(
                "Perception audio proxy failed for session %s: %s",
                perception_session_id,
                exc,
            )
            with suppress(Exception):
                await websocket.close(
                    code=1011,
                    reason="Perception audio proxy failed",
                )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_INDEX_PATH)


def _settings(request: Request | WebSocket) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return get_settings()


def _perception_client(request: Request) -> PerceptionClient:
    client = getattr(request.app.state, "perception_client", None)
    if client is not None:
        return client
    return PerceptionClient(_settings(request).voice_perception_url)


def _create_or_reuse_local_session(perception_session_id: str) -> SessionInfo:
    existing_session = session_store.find_by_perception_id(perception_session_id)
    if existing_session is not None:
        return existing_session

    try:
        return session_store.create(perception_session_id)
    except ValueError:
        existing_session = session_store.find_by_perception_id(perception_session_id)
        if existing_session is not None:
            return existing_session
        raise


def _session_start_payload(
    *,
    settings: Settings,
    session: SessionInfo,
    perception_reachable: bool,
    warning: str | None,
) -> dict[str, Any]:
    perception_session_id = session.perception_session_id
    encoded_session_id = quote(perception_session_id, safe="")
    perception_base_url = settings.voice_perception_url.rstrip("/")
    perception_state_url = f"/perception/state/{encoded_session_id}"
    perception_audio_ws_url = f"/perception/audio/{encoded_session_id}"

    fallback_enabled = _perception_session_fallback_enabled(session)
    payload: dict[str, Any] = {
        "conversation_id": session.conversation_id,
        "perception_session_id": perception_session_id,
        "elevenlabs_agent_id": settings.elevenlabs_agent_id,
        "voice_perception_url": perception_base_url,
        "voice_perception_ws_url": _to_websocket_url(perception_base_url),
        "perception_state_url": perception_state_url,
        "perception_audio_ws_url": perception_audio_ws_url,
        "perception_reachable": perception_reachable,
        "perception_language": settings.perception_language,
        "perception_correlation_mode": (
            "dynamic_variable_with_server_fallback"
            if fallback_enabled
            else "dynamic_variable_only"
        ),
        "perception_fallback_enabled": fallback_enabled,
    }
    if warning is not None:
        payload["warning"] = warning
    return payload


def _perception_session_fallback_enabled(session: SessionInfo) -> bool:
    fallback_session = session_store.get_unambiguous_active(touch=False)
    return (
        fallback_session is not None
        and fallback_session.conversation_id == session.conversation_id
    )


def _to_websocket_url(url: str) -> str:
    if url.startswith("https://"):
        return f"wss://{url[len('https://') :]}"
    if url.startswith("http://"):
        return f"ws://{url[len('http://') :]}"
    return url


def _perception_audio_url(base_url: str, perception_session_id: str) -> str:
    encoded_session_id = quote(perception_session_id, safe="")
    return _to_websocket_url(f"{base_url.rstrip('/')}/audio/{encoded_session_id}")


async def _proxy_audio_websocket(websocket: WebSocket, target_url: str) -> None:
    # uvicorn[standard] provides websockets in normal installs.
    try:
        import websockets
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("websockets package is required for audio proxying") from exc

    async with websockets.connect(target_url) as upstream:
        client_task = asyncio.create_task(
            _client_audio_to_upstream(websocket, upstream)
        )
        upstream_task = asyncio.create_task(
            _upstream_audio_to_client(websocket, upstream)
        )
        done, pending = await asyncio.wait(
            {client_task, upstream_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
        for task in pending:
            with suppress(asyncio.CancelledError):
                await task
        for task in done:
            task.result()


async def _client_audio_to_upstream(websocket: WebSocket, upstream: Any) -> None:
    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                return

            data = message.get("bytes")
            if data is not None:
                await upstream.send(data)
                continue

            text = message.get("text")
            if text is not None:
                await upstream.send(text)
    except WebSocketDisconnect:
        return


async def _upstream_audio_to_client(websocket: WebSocket, upstream: Any) -> None:
    async for message in upstream:
        if isinstance(message, bytes):
            await websocket.send_bytes(message)
        else:
            await websocket.send_text(str(message))


async def _perception_reachable(client: PerceptionClient) -> bool:
    checker = getattr(client, "is_reachable", None)
    if checker is None:
        return False

    try:
        result = checker()
        return bool(await _maybe_await(result))
    except Exception as exc:
        logger.debug("Perception reachability check failed: %s", exc)
        return False


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "voice_agent.main:app",
        host="0.0.0.0",
        port=get_settings().port,
        reload=False,
    )
