#!/usr/bin/env python3
"""Create or update the ElevenLabs Conversational AI agent for the demo.

This script intentionally uses the public ElevenLabs REST API with stdlib HTTP
helpers so it works in the hackathon environment without MCP or extra Python
dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

API_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_AGENT_NAME = "zollhof-clerk-demo"
DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George, multilingual and warm for demos.
DEFAULT_TTS_MODEL = "eleven_flash_v2_5"
DEFAULT_LLM_AUTH_TOKEN = "local-demo-placeholder"

PLACEHOLDER_PREFIXES = (
    "xi-your",
    "your-api-key",
    "<insert-",
)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load missing environment variables from a simple .env file.

    Existing process environment values win. Values are never printed.
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_required_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key or api_key.startswith(PLACEHOLDER_PREFIXES):
        raise SystemExit(
            "ELEVENLABS_API_KEY is not configured. Add it to .env or the environment. "
            "Do not paste the key into chat."
        )
    return api_key


def normalize_custom_llm_url(public_base_url: str) -> str:
    base = public_base_url.strip()
    if not base:
        raise ValueError("public base URL is required")
    if base.endswith("/v1/chat/completions"):
        url = base
    else:
        url = f"{base.rstrip('/')}/v1/chat/completions"
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("ElevenLabs Custom LLM URL must use https")
    if not parsed.netloc:
        raise ValueError("ElevenLabs Custom LLM URL must include a host")
    return url


def build_agent_payload(
    *,
    public_base_url: str,
    name: str = DEFAULT_AGENT_NAME,
    voice_id: str = DEFAULT_VOICE_ID,
    tts_model: str = DEFAULT_TTS_MODEL,
) -> dict[str, Any]:
    custom_llm_url = normalize_custom_llm_url(public_base_url)
    return {
        "name": name,
        "conversation_config": {
            "agent": {
                "first_message": "",
                "language": "de",
                "dynamic_variables": {
                    "dynamic_variable_placeholders": {
                        "perception_session_id": ""
                    }
                },
                "prompt": {
                    "prompt": "You are Frau Weber.",
                    "llm": "custom-llm",
                    "custom_llm": {
                        "url": custom_llm_url,
                        "model_id": "custom",
                        "api_type": "chat_completions",
                        "api_key": DEFAULT_LLM_AUTH_TOKEN,
                    },
                },
            },
            "tts": {
                "voice_id": voice_id,
                "model_id": tts_model,
                "stability": 0.55,
                "similarity_boost": 0.8,
                "speed": 1.0,
            },
            "turn": {
                "turn_eagerness": "normal",
            },
        },
    }


def request_json(method: str, path: str, api_key: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{API_BASE_URL}{path}",
        data=body,
        method=method,
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"ElevenLabs API error HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach ElevenLabs API: {exc.reason}") from exc
    return json.loads(data) if data else {}


def extract_agent_id(response: dict[str, Any], fallback: str | None = None) -> str:
    for key in ("agent_id", "id"):
        value = response.get(key)
        if isinstance(value, str) and value:
            return value
    if fallback:
        return fallback
    raise SystemExit(f"ElevenLabs response did not include an agent id: {response}")


def create_agent(args: argparse.Namespace) -> int:
    api_key = get_required_api_key()
    payload = build_agent_payload(
        public_base_url=args.public_base_url,
        name=args.name,
        voice_id=args.voice_id,
        tts_model=args.tts_model,
    )
    response = request_json("POST", "/convai/agents/create", api_key, payload)
    agent_id = extract_agent_id(response)
    print("Created ElevenLabs Conversational AI agent.")
    print(f"Custom LLM URL: {payload['conversation_config']['agent']['prompt']['custom_llm']['url']}")
    print(f"ELEVENLABS_AGENT_ID={agent_id}")
    return 0


def update_agent(args: argparse.Namespace) -> int:
    api_key = get_required_api_key()
    load_dotenv()
    agent_id = args.agent_id or os.environ.get("ELEVENLABS_AGENT_ID", "").strip()
    if not agent_id or agent_id.startswith("agent_your"):
        raise SystemExit("Provide --agent-id or set ELEVENLABS_AGENT_ID in .env/environment")
    payload = build_agent_payload(
        public_base_url=args.public_base_url,
        name=args.name,
        voice_id=args.voice_id,
        tts_model=args.tts_model,
    )
    response = request_json("PATCH", f"/convai/agents/{agent_id}", api_key, payload)
    resolved_id = extract_agent_id(response, fallback=agent_id)
    print("Updated ElevenLabs Conversational AI agent.")
    print(f"Custom LLM URL: {payload['conversation_config']['agent']['prompt']['custom_llm']['url']}")
    print(f"ELEVENLABS_AGENT_ID={resolved_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Create or update the ElevenLabs Conversational AI agent for voice-agent."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "public_base_url",
            help="Public HTTPS base URL for this FastAPI app, for example https://abc.ngrok-free.app",
        )
        subparser.add_argument("--name", default=os.environ.get("ELEVENLABS_AGENT_NAME", DEFAULT_AGENT_NAME))
        subparser.add_argument(
            "--voice-id",
            default=os.environ.get("ELEVENLABS_AGENT_VOICE_ID", DEFAULT_VOICE_ID),
            help="ElevenLabs voice ID. Defaults to a warm multilingual demo voice.",
        )
        subparser.add_argument(
            "--tts-model",
            default=os.environ.get("ELEVENLABS_AGENT_TTS_MODEL", DEFAULT_TTS_MODEL),
        )

    create = subparsers.add_parser("create", help="Create the Frau Weber demo agent")
    add_common(create)
    create.set_defaults(func=create_agent)

    update = subparsers.add_parser("update-url", help="Update an existing agent when ngrok changes")
    add_common(update)
    update.add_argument("--agent-id", help="Existing ElevenLabs agent ID. Defaults to ELEVENLABS_AGENT_ID.")
    update.set_defaults(func=update_agent)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
