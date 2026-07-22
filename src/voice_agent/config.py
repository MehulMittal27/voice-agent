"""Environment-backed configuration for voice-agent.

Wave 0 keeps configuration independent from the web app so later waves can
import settings without starting network clients or routes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - requirements.txt pins python-dotenv.
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False


DEFAULT_VOICE_PERCEPTION_URL = "http://127.0.0.1:8000"
DEFAULT_PERCEPTION_LANGUAGE = "en"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_DATA_PROVIDER = "mock"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_PORT = 8001
DEFAULT_ELEVENLABS_AGENT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_ELEVENLABS_AGENT_TTS_MODEL = "eleven_flash_v2_5"
DEFAULT_ELEVENLABS_MCP_BASE_PATH = ".elevenlabs-mcp-output"
DEFAULT_ELEVENLABS_MCP_OUTPUT_MODE = "files"


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables.

    Lowercase attributes are the Python API. Uppercase aliases are also
    available for callers that need names matching the environment variables.
    """

    openai_api_key: str = field(default="", repr=False)
    elevenlabs_api_key: str = field(default="", repr=False)
    elevenlabs_agent_id: str = ""
    voice_perception_url: str = DEFAULT_VOICE_PERCEPTION_URL
    perception_language: str = DEFAULT_PERCEPTION_LANGUAGE
    openai_model: str = DEFAULT_OPENAI_MODEL
    data_provider: str = DEFAULT_DATA_PROVIDER
    log_level: str = DEFAULT_LOG_LEVEL
    port: int = DEFAULT_PORT
    elevenlabs_agent_voice_id: str = DEFAULT_ELEVENLABS_AGENT_VOICE_ID
    elevenlabs_agent_tts_model: str = DEFAULT_ELEVENLABS_AGENT_TTS_MODEL
    elevenlabs_mcp_base_path: str = DEFAULT_ELEVENLABS_MCP_BASE_PATH
    elevenlabs_mcp_output_mode: str = DEFAULT_ELEVENLABS_MCP_OUTPUT_MODE

    ENV_ALIASES: ClassVar[dict[str, str]] = {
        "OPENAI_API_KEY": "openai_api_key",
        "ELEVENLABS_API_KEY": "elevenlabs_api_key",
        "ELEVENLABS_AGENT_ID": "elevenlabs_agent_id",
        "VOICE_PERCEPTION_URL": "voice_perception_url",
        "PERCEPTION_LANGUAGE": "perception_language",
        "OPENAI_MODEL": "openai_model",
        "DATA_PROVIDER": "data_provider",
        "LOG_LEVEL": "log_level",
        "PORT": "port",
        "ELEVENLABS_AGENT_VOICE_ID": "elevenlabs_agent_voice_id",
        "ELEVENLABS_AGENT_TTS_MODEL": "elevenlabs_agent_tts_model",
        "ELEVENLABS_MCP_BASE_PATH": "elevenlabs_mcp_base_path",
        "ELEVENLABS_MCP_OUTPUT_MODE": "elevenlabs_mcp_output_mode",
    }

    def __getattr__(self, name: str) -> Any:
        """Expose uppercase environment variable aliases as read-only attrs."""
        field_name = self.ENV_ALIASES.get(name)
        if field_name is None:
            raise AttributeError(name)
        return getattr(self, field_name)

    def as_env(self) -> dict[str, str]:
        """Return settings keyed by environment variable name."""
        return {
            name: str(getattr(self, field_name))
            for name, field_name in self.ENV_ALIASES.items()
        }

    def as_log_dict(self) -> dict[str, str | int]:
        """Return non-secret settings useful for startup logs."""
        return {
            "elevenlabs_agent_id": self.elevenlabs_agent_id,
            "voice_perception_url": self.voice_perception_url,
            "perception_language": self.perception_language,
            "openai_model": self.openai_model,
            "data_provider": self.data_provider,
            "log_level": self.log_level,
            "port": self.port,
            "elevenlabs_agent_voice_id": self.elevenlabs_agent_voice_id,
            "elevenlabs_agent_tts_model": self.elevenlabs_agent_tts_model,
        }


def _env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def _env_int(name: str, default: int) -> int:
    raw_value = _env_str(name, str(default))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def load_settings(dotenv_path: str | Path | None = None) -> Settings:
    """Load settings from `.env` and the process environment.

    Process environment values win over `.env` entries. Missing secrets are
    allowed at import time so local syntax checks can run before real keys are
    configured.
    """
    if dotenv_path is None:
        load_dotenv(override=False)
    else:
        load_dotenv(dotenv_path=dotenv_path, override=False)

    return Settings(
        openai_api_key=_env_str("OPENAI_API_KEY"),
        elevenlabs_api_key=_env_str("ELEVENLABS_API_KEY"),
        elevenlabs_agent_id=_env_str("ELEVENLABS_AGENT_ID"),
        voice_perception_url=_env_str(
            "VOICE_PERCEPTION_URL",
            DEFAULT_VOICE_PERCEPTION_URL,
        ).rstrip("/"),
        perception_language=_env_str(
            "PERCEPTION_LANGUAGE",
            DEFAULT_PERCEPTION_LANGUAGE,
        ).lower(),
        openai_model=_env_str("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        data_provider=_env_str("DATA_PROVIDER", DEFAULT_DATA_PROVIDER).lower(),
        log_level=_env_str("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        port=_env_int("PORT", DEFAULT_PORT),
        elevenlabs_agent_voice_id=_env_str(
            "ELEVENLABS_AGENT_VOICE_ID",
            DEFAULT_ELEVENLABS_AGENT_VOICE_ID,
        ),
        elevenlabs_agent_tts_model=_env_str(
            "ELEVENLABS_AGENT_TTS_MODEL",
            DEFAULT_ELEVENLABS_AGENT_TTS_MODEL,
        ),
        elevenlabs_mcp_base_path=_env_str(
            "ELEVENLABS_MCP_BASE_PATH",
            DEFAULT_ELEVENLABS_MCP_BASE_PATH,
        ),
        elevenlabs_mcp_output_mode=_env_str(
            "ELEVENLABS_MCP_OUTPUT_MODE",
            DEFAULT_ELEVENLABS_MCP_OUTPUT_MODE,
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings for normal application use."""
    return load_settings()
