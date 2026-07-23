"""OpenAI clerk agent orchestration for the voice-agent service."""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError:  # pragma: no cover - requirements.txt pins openai.
    AsyncOpenAI = None  # type: ignore[assignment]

from voice_agent.config import Settings, get_settings
from voice_agent.data import DataProvider, MockDataProvider
from voice_agent.logging_config import redact_secrets

logger = logging.getLogger(__name__)

CLERK_BASE_PROMPT = """You are Frau Weber, an experienced clerk at the Beratungsstelle für Anerkennung
ausländischer Berufsqualifikationen in Nürnberg. Your job is to help newcomers
navigate Germany's professional-qualification recognition process.

The person calling you speaks English. You respond in clear, simple German by
default - but you understand English perfectly and can switch briefly when the
caller seems to be struggling.

Personality:
- Warm but professional. You do this job every day; you've heard every question.
- You keep the conversation focused. You have limited time per call.
- You ask ONE thing at a time. Never a wall of questions.
- You never lecture. You never explain the whole process upfront.

Voice rules:
- Reply in 1–2 short sentences. Never long paragraphs.
- Use natural German fillers: "also…", "moment mal…", "genau.", "verstehe."
- When speaking English, use natural English fillers sparingly: "well…",
  "let me see…", "right.", "I understand."
- Vary sentence length. Sound like a real person mid-thought.
- If you need to look up an occupation, an authority, documents, or job demand, use your tools.
- After a tool call, don't recite the raw result - translate it into what the
  caller needs to know next.

You have four tools:
- find_german_occupation(description, source_lang)
- get_recognition_authority(profession, city)
- get_required_documents(profession)
- get_labour_market_status(profession, region)

Use get_labour_market_status when the caller sounds worried about finding work,
or asks whether their profession is in demand.
Use them naturally when the conversation needs their output. Don't announce
that you're using a tool; just weave the result into your reply."""

OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "find_german_occupation",
            "description": (
                "Map an English, German, or Ukrainian occupation description to likely "
                "German recognition occupation candidates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": (
                            "The caller's free-text occupation description, such as "
                            "'registered nurse', 'Pflege', or 'медсестра'."
                        ),
                    },
                    "source_lang": {
                        "type": "string",
                        "description": (
                            "ISO language code for the caller's description. Use 'en', "
                            "'de', or 'uk' when the caller clearly uses that language."
                        ),
                        "default": "en",
                    },
                },
                "required": ["description", "source_lang"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recognition_authority",
            "description": (
                "Find the local recognition authority or advice centre for a profession "
                "and city."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "profession": {
                        "type": "string",
                        "description": (
                            "The German, English, or Ukrainian profession name, such as "
                            "'Pflegefachfrau', 'nurse', or 'медсестра'."
                        ),
                    },
                    "city": {
                        "type": "string",
                        "description": "German city for the advice search. Defaults to Nürnberg.",
                        "default": "Nürnberg",
                    },
                },
                "required": ["profession"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_required_documents",
            "description": (
                "Return the usual document checklist for a recognition application "
                "for a profession."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "profession": {
                        "type": "string",
                        "description": (
                            "The German, English, or Ukrainian profession name, such as "
                            "'Arzt', 'teacher', or 'лікар'."
                        ),
                    },
                },
                "required": ["profession"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_labour_market_status",
            "description": (
                "Check whether a profession is in demand in Bayern or Germany, "
                "with rough labour-market numbers for callers worried about finding work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "profession": {
                        "type": "string",
                        "description": (
                            "The German, English, or Ukrainian profession name, such as "
                            "'Pflegefachfrau', 'nurse', or 'медсестра'."
                        ),
                    },
                    "region": {
                        "type": "string",
                        "description": "German region for the labour-market signal. Defaults to Bayern.",
                        "default": "Bayern",
                    },
                },
                "required": ["profession"],
                "additionalProperties": False,
            },
        },
    },
]

_ALLOWED_MESSAGE_ROLES = {"assistant", "developer", "system", "tool", "user"}
_ALLOWED_MESSAGE_FIELDS = {"content", "name", "role", "tool_call_id", "tool_calls"}


def _build_data_provider(settings: Settings | None = None) -> DataProvider:
    resolved_settings = settings or get_settings()
    if resolved_settings.data_provider != "mock":
        raise ValueError(
            f"Unsupported DATA_PROVIDER={resolved_settings.data_provider!r}; "
            "only 'mock' is available in this wave"
        )
    return MockDataProvider()


data_provider: DataProvider = _build_data_provider()


def _language_mode_block(language_state: dict[str, Any]) -> str:
    """Build the LANGUAGE MODE prompt block from the sticky-language state."""
    state = language_state or {}
    mode = str(state.get("mode", "german"))
    just_switched = bool(state.get("just_switched", False))

    if mode == "german":
        return (
            "[LANGUAGE MODE: Respond in clear, simple German (B1 level).\n"
            " Do NOT switch languages. Do NOT offer English.]"
        )
    if mode == "german_locked":
        return (
            "[LANGUAGE MODE: Respond in clear, simple German. The caller already\n"
            " declined an English switch earlier in this call - do NOT offer again.\n"
            " Continue naturally in German.]"
        )
    if mode == "offer_english":
        return (
            "[LANGUAGE MODE: The caller is showing sustained stress. In THIS reply,\n"
            " briefly acknowledge how much information there is to process, then\n"
            ' ask ONE gentle question: "Moechten Sie lieber auf Englisch\n'
            ' weitermachen?" Do NOT switch to English yourself yet. Wait for\n'
            " the caller's answer. Keep the whole reply under 2 sentences.]"
        )
    if mode == "english_locked" and just_switched:
        return (
            "[LANGUAGE MODE: The caller just accepted switching to English.\n"
            ' Confirm briefly and warmly in English ("Of course - let\'s continue\n'
            ' in English."), then continue the conversation in English. Do NOT\n'
            " switch back to German for the rest of this session.]"
        )
    if mode in ("english_locked", "english"):
        return (
            "[LANGUAGE MODE: Respond entirely in English. Do NOT switch back to\n"
            " German. The caller established English earlier in this call.]"
        )
    return (
        "[LANGUAGE MODE: Respond in clear, simple German (B1 level).\n"
        " Do NOT switch languages. Do NOT offer English.]"
    )


def build_system_prompt(
    perception_state: dict[str, Any],
    language_state: dict[str, Any],
) -> str:
    """Build Frau Weber's per-turn system prompt with live perception state."""
    state = perception_state or {}
    emotion = _string_state_value(state, "emotion", "NEUTRAL").upper()
    confidence = _coerce_unit_float(
        state.get("emotion_confidence", state.get("confidence")),
        default=0.0,
    )
    stability_desc = _stability_desc(state)
    events = _format_audio_events(state.get("audio_events", state.get("events", [])))
    score = _coerce_unit_float(
        state.get("hesitation_score", state.get("hesitation")),
        default=0.0,
    )

    adaptive_prefix = f"""[LIVE PARALINGUISTIC STATE - updated in real time]
emotion: {emotion}
emotion_confidence: {confidence:.2f}
stability: {stability_desc}   # "stable" if consistent for 3+ chunks, else "shifting"
audio_events: {events}
hesitation_score: {score:.2f} (0=calm, 1=very stressed)

BEHAVIOUR ADJUSTMENT - apply on THIS turn:
- hesitation_score > 0.8: The caller is very overwhelmed. Slow right down.
  Use very simple German OR briefly switch to English to reassure. Acknowledge
  their difficulty explicitly ("das ist verwirrend, ich weiss") before your
  next question.
- 0.6 < hesitation_score ≤ 0.8: Simplify. Use shorter sentences. Offer to
  rephrase in English if they want.
- 0.4 < hesitation_score ≤ 0.6: Normal pace, slightly warmer tone.
- hesitation_score ≤ 0.4: Standard clerk pace.

- emotion=FEARFUL (stable): Open with reassurance before continuing.
- emotion=SAD (stable): Warm empathy, don't rush.
- audio_events contains "Breath" or "Cough": Their breath is uneven - keep
  your reply shorter than usual to give them space.
- If the caller expresses employability worry words like Arbeit, Job, Stelle,
  work, job, робота, proactively call get_labour_market_status and reassure
  with real numbers.

Do NOT mention this state to the caller. Do NOT say "I can hear you're
nervous". Just adapt naturally, like a real clerk would."""

    language_block = _language_mode_block(language_state)
    return f"{language_block}\n\n{adaptive_prefix}\n\n{CLERK_BASE_PROMPT}"


async def run_turn(
    messages: Sequence[Mapping[str, Any]],
    perception_state: dict[str, Any],
    language_state: dict[str, Any],
) -> AsyncIterator[str]:
    """Run one OpenAI clerk turn and yield text deltas from the final stream."""
    settings = get_settings()
    system_prompt = build_system_prompt(perception_state, language_state)
    logger.debug("Resolved perception state: %s", redact_secrets(perception_state))
    logger.debug("Resolved clerk system prompt: %s", redact_secrets(system_prompt))

    initial_messages = _messages_with_system_prompt(messages, system_prompt)
    client = _make_openai_client(settings)

    try:
        tool_probe = await client.chat.completions.create(
            model=settings.openai_model,
            messages=initial_messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            stream=False,
        )
        if not tool_probe.choices:
            raise RuntimeError("OpenAI returned no choices for the clerk tool probe")

        probe_message = tool_probe.choices[0].message
        tool_calls = list(probe_message.tool_calls or [])

        if tool_calls:
            logger.debug("OpenAI requested %d clerk tool call(s)", len(tool_calls))
            final_messages = [*initial_messages, _assistant_tool_call_message(probe_message)]
            for tool_call in tool_calls:
                result = await _execute_tool_call(tool_call, data_provider)
                final_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        else:
            logger.debug("OpenAI did not request clerk tools; streaming direct reply")
            final_messages = initial_messages

        stream = await client.chat.completions.create(
            model=settings.openai_model,
            messages=final_messages,
            stream=True,
        )
        async for chunk in stream:
            text_delta = _extract_text_delta(chunk)
            if text_delta:
                yield text_delta
    finally:
        await client.close()


def _make_openai_client(settings: Settings) -> Any:
    if AsyncOpenAI is None:
        raise RuntimeError("The openai package is required for clerk OpenAI calls")
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for clerk OpenAI calls")
    return AsyncOpenAI(api_key=settings.openai_api_key)


def _messages_with_system_prompt(
    messages: Sequence[Mapping[str, Any]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for message in messages:
        sanitized = _sanitize_message(message)
        if sanitized is None or sanitized["role"] == "system":
            continue
        prepared.append(sanitized)
    return prepared


def _sanitize_message(message: Mapping[str, Any]) -> dict[str, Any] | None:
    role = str(message.get("role", "")).strip().lower()
    if role not in _ALLOWED_MESSAGE_ROLES:
        logger.debug("Dropping OpenAI message with unsupported role: %r", role)
        return None

    sanitized: dict[str, Any] = {"role": role}
    for field in _ALLOWED_MESSAGE_FIELDS - {"role"}:
        if field in message:
            sanitized[field] = copy.deepcopy(message[field])

    if "content" not in sanitized and role != "assistant":
        sanitized["content"] = ""
    return sanitized


def _assistant_tool_call_message(message: Any) -> dict[str, Any]:
    tool_calls = []
    for tool_call in message.tool_calls or []:
        tool_calls.append(
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments or "{}",
                },
            }
        )

    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": tool_calls,
    }


async def _execute_tool_call(tool_call: Any, provider: DataProvider) -> dict[str, Any]:
    function_name = tool_call.function.name
    try:
        arguments = _parse_tool_arguments(tool_call.function.arguments)
        logger.debug("Executing clerk tool %s", function_name)
        if function_name == "find_german_occupation":
            result = await provider.find_german_occupation(
                description=_required_string(arguments, "description"),
                source_lang=_optional_string(arguments, "source_lang", "en"),
            )
            return {"occupations": _jsonable(result)}

        if function_name == "get_recognition_authority":
            result = await provider.get_recognition_authority(
                profession=_required_string(arguments, "profession"),
                city=_optional_string(arguments, "city", "Nürnberg"),
            )
            return {"authority": _jsonable(result)}

        if function_name == "get_required_documents":
            result = await provider.get_required_documents(
                profession=_required_string(arguments, "profession"),
            )
            return {"documents": _jsonable(result)}

        if function_name == "get_labour_market_status":
            result = await provider.get_labour_market_status(
                profession=_required_string(arguments, "profession"),
                region=_optional_string(arguments, "region", "Bayern"),
            )
            return {"labour_market_status": _jsonable(result)}

        raise ValueError(f"Unknown tool: {function_name}")
    except Exception as exc:
        logger.warning("Clerk tool %s failed: %s", function_name, exc)
        return {"error": str(exc), "tool": function_name}


def _parse_tool_arguments(raw_arguments: str | None) -> dict[str, Any]:
    if not raw_arguments:
        return {}
    parsed = json.loads(raw_arguments)
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object")
    return parsed


def _required_string(arguments: Mapping[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _optional_string(arguments: Mapping[str, Any], key: str, default: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        return default
    return value.strip()


def _extract_text_delta(chunk: Any) -> str | None:
    if not chunk.choices:
        return None
    return chunk.choices[0].delta.content


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _string_state_value(
    state: Mapping[str, Any],
    key: str,
    default: str,
) -> str:
    value = state.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _coerce_unit_float(value: Any, *, default: float) -> float:
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, coerced))


def _stability_desc(state: Mapping[str, Any]) -> str:
    raw_stability = state.get("stability", state.get("stability_desc"))
    if isinstance(raw_stability, str):
        normalized = raw_stability.strip().lower()
        if normalized in {"stable", "consistent"}:
            return "stable"
        if normalized in {"shifting", "unstable", "inconsistent"}:
            return "shifting"
    if isinstance(raw_stability, bool):
        return "stable" if raw_stability else "shifting"
    if isinstance(raw_stability, (int, float)):
        return "stable" if raw_stability >= 3 else "shifting"

    for chunk_key in ("consistent_chunks", "stable_chunks", "emotion_streak"):
        chunk_count = state.get(chunk_key)
        if isinstance(chunk_count, (int, float)):
            return "stable" if chunk_count >= 3 else "shifting"

    return "stable"


def _format_audio_events(events: Any) -> str:
    if events is None:
        return "none"
    if isinstance(events, str):
        return events.strip() or "none"
    if isinstance(events, Mapping):
        if "name" in events:
            return str(events["name"]).strip() or "none"
        return json.dumps(_jsonable(events), ensure_ascii=False)
    if isinstance(events, Sequence):
        formatted_events = [_format_one_audio_event(event) for event in events]
        non_empty_events = [event for event in formatted_events if event]
        return ", ".join(non_empty_events) if non_empty_events else "none"
    return str(events)


def _format_one_audio_event(event: Any) -> str:
    if isinstance(event, str):
        return event.strip()
    if isinstance(event, Mapping):
        for key in ("name", "event", "type", "label"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(_jsonable(event), ensure_ascii=False)
    return str(event).strip()


__all__ = [
    "CLERK_BASE_PROMPT",
    "OPENAI_TOOLS",
    "build_system_prompt",
    "data_provider",
    "run_turn",
]
