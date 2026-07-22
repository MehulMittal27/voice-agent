"""Structured logging helpers with API key redaction."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

SECRET_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:sk-proj-|sk-|xi-)[^\s\"'`<>(){}\[\],;]*"
)
REDACTION_TEXT = "[REDACTED]"


def redact_secrets(value: Any) -> Any:
    """Redact OpenAI and ElevenLabs API-key shaped values.

    Strings containing tokens that start with `sk-proj-`, `sk-`, or `xi-` are
    redacted. Containers are handled recursively for callers that want to clean
    structured data before logging it.
    """
    if isinstance(value, str):
        return SECRET_PATTERN.sub(REDACTION_TEXT, value)
    if isinstance(value, dict):
        return {key: redact_secrets(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, set):
        return {redact_secrets(item) for item in value}
    return value


class StructuredRedactingFormatter(logging.Formatter):
    """JSON log formatter that redacts secrets after formatting."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.format_time(record),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return redact_secrets(encoded)

    @staticmethod
    def format_time(record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def coerce_log_level(level: str | int | None) -> int:
    """Convert a string or integer level into a logging level constant."""
    if level is None:
        return logging.INFO
    if isinstance(level, int):
        return level
    normalized = level.strip().upper()
    resolved = logging.getLevelName(normalized)
    if isinstance(resolved, int):
        return resolved
    raise ValueError(f"Unsupported log level: {level!r}")


def configure_logging(level: str | int | None = None) -> None:
    """Configure root logging for the service.

    When level is not supplied, `LOG_LEVEL` is read through the project config.
    The function is idempotent and replaces existing root handlers so repeated
    startup calls do not duplicate log lines.
    """
    if level is None:
        from voice_agent.config import get_settings

        level = get_settings().log_level

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredRedactingFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(coerce_log_level(level))

    logging.captureWarnings(True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger using the project's naming convention."""
    return logging.getLogger(name)
