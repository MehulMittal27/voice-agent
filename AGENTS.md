# AGENTS.md — Voice Agent Service

## What this is
Standalone service that runs a live voice conversation between a user (English speaker)
and an adaptive AI clerk playing a German recognition-office employee (Beratungsstelle
für Anerkennung ausländischer Berufsqualifikationen).

The clerk is powered by:
- **ElevenLabs Conversational AI** for STT + turn-taking + TTS (voice UI)
- **Claude Haiku 4.5** for dialogue generation (the clerk's brain)
- **voice-perception** service (companion repo — see below) for real-time
  emotion / hesitation signals
- **Mocked data layer** for occupation mapping / advice centres / required documents
  (swap for real BIBB + anabin + Integreat APIs tomorrow at the hackathon)

The user speaks English. The clerk responds in German with occasional English
clarifications when the perception layer detects high hesitation or fear.

## Companion service (required)
This service depends on the **voice-perception** service.

- **Repo:** https://github.com/mehulmittal27/voice-perception
- Clone and set it up per that repo's README before running voice-agent.
- It exposes an HTTP + WebSocket API that this service queries per turn to
  read the caller's current emotion, audio events, and hesitation score.
- The URL where voice-perception runs is configured via the
  `VOICE_PERCEPTION_URL` environment variable (see `.env.example`).
- If voice-perception is unreachable, voice-agent still runs but falls back
  to neutral perception defaults (no adaptive behaviour). `GET /health`
  reports `perception_reachable: false` in that state.

## Design principles
1. **Three specialised components meet inside our webhook.** ElevenLabs is the
   mouth/ears/turn-taker, Claude is the brain, perception is the empathy layer.
2. **The webhook is OpenAI-compatible.** ElevenLabs Custom LLM expects an
   OpenAI `/chat/completions` endpoint with Server-Sent Events streaming.
   We wrap Claude's Anthropic API to that format.
3. **Session correlation** — the browser generates a perception session ID,
   passes it to ElevenLabs as a dynamic variable, ElevenLabs forwards it to
   our webhook on every LLM call, we query voice-perception with it. Clean
   handoff, no shared state between services.
4. **Data layer is mocked tonight, real tomorrow.** All data lookups go
   through a `DataProvider` interface with a `MockDataProvider` implementation.
   Tomorrow, add `RealDataProvider` that hits BIBB / anabin / Integreat.
   Swap via env var — no dialogue code changes.
5. **Streaming end-to-end.** From Claude's first token, to our SSE chunk, to
   ElevenLabs' first audio chunk, to the user's ear — nothing waits for a
   full response. Target: user hears first word within ~1.5s of stopping speech.

## Tech stack (locked)
- Python 3.11
- FastAPI + uvicorn (HTTP + SSE streaming)
- Anthropic Python SDK (primary model: `claude-haiku-4-5`)
- httpx (async HTTP client for perception service calls)
- python-dotenv for config
- ngrok (or cloudflared) to expose the webhook to ElevenLabs
- No frontend framework — one `index.html` with the ElevenLabs JS SDK

## Repo structure
```
voice-agent/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   └── voice_agent/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app + routes
│       ├── webhook.py              # OpenAI-compatible /v1/chat/completions
│       ├── clerk.py                # Claude clerk agent, system prompt builder
│       ├── perception_client.py    # Async HTTP client to voice-perception
│       ├── session.py              # In-memory session store
│       ├── streaming.py            # Anthropic → OpenAI SSE translation
│       ├── data/
│       │   ├── __init__.py
│       │   ├── base.py             # DataProvider abstract interface
│       │   └── mock.py             # MockDataProvider — tonight's impl
│       ├── config.py               # Env-var loading
│       └── logging_config.py
├── static/
│   └── index.html                  # Browser demo UI
└── scripts/
    ├── test_webhook.py             # Simulate an ElevenLabs webhook call
    └── test_perception_client.py   # Verify voice-perception is reachable
```

## API contract

### `POST /session/start`
Request: `{}` (empty)
Response:
```json
{
  "perception_session_id": "<uuid>",
  "elevenlabs_agent_id": "<from env>"
}
```
Internally: calls voice-perception's `/session/start` **with the caller's
language** — `POST { "language": "<PERCEPTION_LANGUAGE from env>" }` — to
create the perception session, and stores the ID in the local session store
keyed by our conversation ID.

Voice-perception routes the transcript lane based on this: German uses
faster-whisper base, non-German languages fall through to SenseVoice's
supported set. Either way, Emotion2Vec handles emotion cross-lingually.
For this demo the caller speaks English, so `language=en`.

### `POST /v1/chat/completions`
The ElevenLabs Custom LLM webhook. OpenAI-compatible.

Request body (ElevenLabs sends this):
```json
{
  "model": "custom",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "stream": true,
  "perception_session_id": "<uuid>"
}
```
`perception_session_id` may arrive as a top-level field, inside `extra_body`,
or inside the last user message's metadata depending on ElevenLabs SDK version.
Handle all three; log which one hit.

Response: SSE stream in OpenAI format:
```
data: {"id":"...","choices":[{"delta":{"content":"Guten"},"index":0}]}\n\n
data: {"id":"...","choices":[{"delta":{"content":" Tag"},"index":0}]}\n\n
...
data: [DONE]\n\n
```

Behind the scenes:
1. Extract `perception_session_id`
2. `await perception_client.get_state(session_id)` — 100ms timeout, fail soft to defaults
3. Build Claude system prompt with the perception state injected
4. Open Anthropic streaming call (`messages.stream()`)
5. For each text delta from Claude, wrap in OpenAI SSE JSON and yield
6. Terminate with `data: [DONE]\n\n`

### `GET /health`
Response: `{ "status": "ok", "perception_reachable": true|false }`

### `GET /` (static)
Serves `static/index.html`.

## The clerk personality (get this right)

**Base persona** (in `clerk.py` as `CLERK_BASE_PROMPT`):

```
You are Frau Weber, an experienced clerk at the Beratungsstelle für Anerkennung
ausländischer Berufsqualifikationen in Nürnberg. Your job is to help newcomers
navigate Germany's professional-qualification recognition process.

The person calling you speaks English. You respond in clear, simple German by
default — but you understand English perfectly and can switch briefly when the
caller seems to be struggling.

Personality:
- Warm but professional. You do this job every day; you've heard every question.
- You keep the conversation focused. You have limited time per call.
- You ask ONE thing at a time. Never a wall of questions.
- You never lecture. You never explain the whole process upfront.

Voice rules:
- Reply in 1–2 short sentences. Never long paragraphs.
- Use natural German fillers: "also…", "moment mal…", "genau.", "verstehe."
- Vary sentence length. Sound like a real person mid-thought.
- If you need to look up an occupation, an authority, or documents, use your tools.
- After a tool call, don't recite the raw result — translate it into what the
  caller needs to know next.

You have three tools:
- find_german_occupation(description, source_lang)
- get_recognition_authority(profession, city)
- get_required_documents(profession)

Use them naturally when the conversation needs their output. Don't announce
that you're using a tool; just weave the result into your reply.
```

**Perception-driven adaptive prefix** (built per-turn in `clerk.py::build_system_prompt`):

```
[LIVE PARALINGUISTIC STATE — updated in real time]
emotion: {emotion}
emotion_confidence: {confidence:.2f}
stability: {stability_desc}   # "stable" if consistent for 3+ chunks, else "shifting"
audio_events: {events}
hesitation_score: {score:.2f} (0=calm, 1=very stressed)

BEHAVIOUR ADJUSTMENT — apply on THIS turn:
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
- audio_events contains "Breath" or "Cough": Their breath is uneven — keep
  your reply shorter than usual to give them space.

Do NOT mention this state to the caller. Do NOT say "I can hear you're
nervous". Just adapt naturally, like a real clerk would.
```

## Tools (mocked tonight, real tomorrow)

`data/base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Occupation:
    esco_id: str
    label_de: str
    label_en: str
    kldb_code: str | None
    confidence: float

@dataclass
class Authority:
    name: str
    city: str
    email: str
    phone: str
    website: str

@dataclass
class Document:
    name_de: str
    name_en: str
    notes: str

class DataProvider(ABC):
    @abstractmethod
    async def find_german_occupation(self, description: str, source_lang: str) -> list[Occupation]: ...
    @abstractmethod
    async def get_recognition_authority(self, profession: str, city: str = "Nürnberg") -> Authority | None: ...
    @abstractmethod
    async def get_required_documents(self, profession: str) -> list[Document]: ...
```

`data/mock.py` returns hardcoded plausible responses. Include enough
variety to cover ~5 professions (nurse, engineer, teacher, doctor, IT admin)
so the demo tomorrow can lock to any of them without changes.

Wire these as Anthropic tools using the SDK's tool-use format. The clerk can
call zero, one, or multiple tools per turn before generating its verbal reply.

## Streaming translation (Anthropic → OpenAI SSE)

`streaming.py::anthropic_to_openai_sse(stream)`:

For each event from Anthropic's `messages.stream()`:
- On `text_stream` yield → wrap as
  `{"id": id, "object": "chat.completion.chunk", "choices": [{"delta": {"content": text}, "index": 0, "finish_reason": null}]}`
- On stream end → yield final chunk with `finish_reason: "stop"`, then `data: [DONE]\n\n`

Between chunks: `f"data: {json.dumps(payload)}\n\n"`.

**Tool calls:** for the hackathon, execute tools server-side BEFORE the final
text stream — don't try to expose tool events through OpenAI-format SSE.
Concretely: call Anthropic non-streaming, if the response contains tool_use
blocks execute them locally, feed results back with a second Anthropic call,
this time streaming. The user shouldn't hear the tool call; they should hear
the resulting answer.

## Session correlation flow

1. Browser: `POST /session/start` on voice-agent → receives `perception_session_id`
2. Browser: initialises ElevenLabs SDK, passing `perception_session_id` inside
   `conversation_initiation_client_data.dynamic_variables`
3. Browser: also opens the audio WebSocket to voice-perception with the same
   `perception_session_id` — mic audio streams there for analysis
4. User speaks → ElevenLabs STT + turn detection → ElevenLabs calls our webhook
   with the messages and the dynamic variable
5. Webhook extracts `perception_session_id`, fetches state, calls Claude,
   streams response back
6. ElevenLabs speaks the response

**Key: one browser mic stream goes to two consumers.**
```javascript
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
// stream is shared between ElevenLabs SDK AND voice-perception WebSocket
```

## ElevenLabs dashboard configuration (do once tonight)

Log into elevenlabs.io → **Conversational AI → Agents → Create Agent**.

- **Name**: `zollhof-clerk-demo`
- **Voice**: pick a German-capable voice you like. Recommendation: audition
  3–4 voices tonight with a sample German sentence. Look for warm mid-pitch.
- **Voice model**: `eleven_flash_v2_5` (low latency, ~75ms first audio)
- **Language**: `de` for output; enable automatic language detection for input
  so it accepts English cleanly
- **First message**: leave empty
- **System prompt**: a placeholder like "You are Frau Weber." — the real prompt
  is injected via our webhook every turn
- **LLM**: `Custom LLM`
  - **Server URL**: your ngrok HTTPS URL + `/v1/chat/completions`
  - **Model ID**: any string (we ignore it)
  - **API Key**: any string (we don't enforce auth tonight; add a TODO)
- **Dynamic variables** (Advanced → Variables): add `perception_session_id`
- **Save** and copy the Agent ID → paste into `.env` as `ELEVENLABS_AGENT_ID`

## Environment (`.env.example`)
```
ANTHROPIC_API_KEY=sk-ant-...
ELEVENLABS_API_KEY=xi-...
ELEVENLABS_AGENT_ID=agent_...
VOICE_PERCEPTION_URL=http://127.0.0.1:8000
PERCEPTION_LANGUAGE=en          # ISO code of the CALLER's spoken language.
                                # The clerk's German TTS goes via ElevenLabs,
                                # not perception, so this is the user's language.
CLAUDE_MODEL_FAST=claude-haiku-4-5
DATA_PROVIDER=mock
LOG_LEVEL=INFO
PORT=8001
```

## ngrok setup (tonight)

Once the server runs on port 8001:
```bash
ngrok http 8001
```
Copy the HTTPS URL (e.g. `https://abc123.ngrok-free.app`) and paste it into
the ElevenLabs agent's Custom LLM Server URL field (with `/v1/chat/completions`
appended).

Free ngrok tunnels expire on restart — expect to update the URL in the
dashboard tomorrow morning. Consider `cloudflared` for a stable tunnel if
you want to avoid this.

## Testing rules

**Isolated tests (must pass before touching ElevenLabs):**
- `python scripts/test_perception_client.py` — verifies voice-perception is
  reachable and returns state
- `python scripts/test_webhook.py` — sends a fake ElevenLabs POST to our
  webhook and prints the SSE stream. Validates the full Claude + perception
  + tools pipeline without any voice component.

**End-to-end test:**
- Start voice-perception (see its repo README)
- Start voice-agent on 8001
- Start ngrok on 8001, paste URL into ElevenLabs dashboard
- Open the browser UI → Start → grant mic → see perception state updating
- Speak English → hear German reply within ~1.5s
- Speak with clearly anxious tone → hesitation score rises, clerk softens
- Ask the clerk about a profession → tool call → verbal answer

## Known gotchas
1. **ElevenLabs dynamic variable placement varies.** Log the entire incoming
   webhook body on the first request each session so you can see where
   `perception_session_id` actually landed.
2. **Perception client timeout matters.** If voice-perception is slow, don't
   block the whole webhook. 100ms timeout, fall back to defaults, log a warning.
3. **SSE requires specific headers.** `Content-Type: text/event-stream`,
   `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`.
4. **Anthropic streaming with tools is different from plain streaming.** For
   tonight, execute tools BEFORE the final text stream (non-streaming call to
   resolve tool use, then a second streaming call for the verbal response).
   This is simpler and reliable. Optimise later.
5. **CORS for local dev.** Enable CORS on the FastAPI app allowing the ngrok
   URL, since the browser will call both voice-perception and voice-agent.
6. **Perception language routing.** voice-perception picks its transcript
   engine per session based on the `language` sent at `/session/start`.
   `de` → faster-whisper base (best German ASR). Other codes → SenseVoice's
   supported set. If you demo a non-English caller tomorrow (e.g. Ukrainian),
   flip `PERCEPTION_LANGUAGE` in `.env` accordingly. Emotion + events are
   cross-lingual and unaffected.

## Non-goals (do not build tonight)
- Real BIBB/anabin/Integreat integration
- Auth on the webhook
- Persistent conversation history across restarts
- Multi-tenant session isolation beyond in-memory dict
- Handling concurrent sessions beyond 1 at a time (this is a demo)
- Deployment scripts

## Coding conventions
- Type hints everywhere
- Async where FastAPI expects it, sync elsewhere
- No global mutable state except `session_store` and `data_provider` singletons
- Log at INFO for lifecycle events, DEBUG for per-turn detail
- Every webhook call logs the perception state and the resolved system prompt
  (redact keys). Debug-first for hackathon.
