#!/usr/bin/env python3
"""Smoke test the voice-perception HTTP client against a live service."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from voice_agent.config import get_settings
from voice_agent.perception_client import PerceptionClient, PerceptionClientError


def print_json(label: str, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"{label}: {encoded}")


async def run() -> int:
    settings = get_settings()
    client = PerceptionClient(settings.voice_perception_url)

    print(f"VOICE_PERCEPTION_URL={settings.voice_perception_url}")
    print(f"PERCEPTION_LANGUAGE={settings.perception_language}")

    session_id: str | None = None
    try:
        session_id = await client.start_session(settings.perception_language)
    except (PerceptionClientError, ValueError) as exc:
        print("perception_reachable: false")
        print(f"start_session_error: {exc}")
        fallback_state = await client.get_state("unreachable-smoke-test")
        print_json("fail_soft_state", fallback_state)
        return 0

    try:
        print("perception_reachable: true")
        print(f"perception_session_id: {session_id}")
        state = await client.get_state(session_id)
        print_json("state", state)
        return 0
    finally:
        await client.end_session(session_id)
        print("end_session: requested (fail soft)")


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
