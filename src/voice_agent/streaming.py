"""Translate upstream text deltas into OpenAI-compatible SSE chunks."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import uuid4


_DONE_EVENT = "data: [DONE]\n\n"


def _new_completion_id() -> str:
    return f"chatcmpl-{uuid4().hex}"


def _sse_json(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk_payload(
    completion_id: str,
    delta: dict[str, str],
    finish_reason: str | None,
) -> dict[str, object]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "delta": delta,
                "index": 0,
                "finish_reason": finish_reason,
            }
        ],
    }


async def openai_to_openai_sse(
    stream: AsyncIterator[str],
    completion_id: str | None = None,
) -> AsyncIterator[str]:
    """Yield OpenAI-format SSE events for an async stream of text deltas."""
    resolved_completion_id = completion_id or _new_completion_id()

    async for text_delta in stream:
        yield _sse_json(
            _chunk_payload(
                completion_id=resolved_completion_id,
                delta={"content": text_delta},
                finish_reason=None,
            )
        )

    yield _sse_json(
        _chunk_payload(
            completion_id=resolved_completion_id,
            delta={},
            finish_reason="stop",
        )
    )
    yield _DONE_EVENT


async def to_openai_sse(
    text_stream: AsyncIterator[str],
    completion_id: str,
) -> AsyncIterator[str]:
    """Compatibility wrapper around openai_to_openai_sse."""
    async for event in openai_to_openai_sse(text_stream, completion_id=completion_id):
        yield event


async def _smoke() -> None:
    async def _deltas() -> AsyncIterator[str]:
        for chunk in ("Guten", " Tag"):
            yield chunk

    async for event in openai_to_openai_sse(_deltas(), completion_id="chatcmpl-smoke"):
        print(event, end="")


if __name__ == "__main__":
    asyncio.run(_smoke())
