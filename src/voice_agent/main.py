"""FastAPI application entrypoint for the voice-agent service."""

from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from voice_agent.config import Settings, get_settings
from voice_agent.logging_config import configure_logging
from voice_agent.perception_client import PerceptionClient, PerceptionClientError
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

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_INDEX_PATH)


def _settings(request: Request) -> Settings:
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
    perception_state_url = f"{perception_base_url}/state/{encoded_session_id}"
    perception_audio_ws_url = _to_websocket_url(
        f"{perception_base_url}/audio/{encoded_session_id}"
    )

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
    }
    if warning is not None:
        payload["warning"] = warning
    return payload


def _to_websocket_url(url: str) -> str:
    if url.startswith("https://"):
        return f"wss://{url[len('https://') :]}"
    if url.startswith("http://"):
        return f"ws://{url[len('http://') :]}"
    return url


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
