# voice-agent

Live voice demo for a German recognition-office clerk using ElevenLabs Conversational AI, OpenAI, and the companion `voice-perception` service.

## Quickstart

Run commands from the repo root.

### 0. Prerequisites

- Python 3.11
- `uv`
- An OpenAI API key and an ElevenLabs API key
- `ngrok` or another HTTPS tunnel for ElevenLabs to reach your local webhook
- The companion `voice-perception` service cloned and running separately: <https://github.com/mehulmittal27/voice-perception>

By default this app expects `voice-perception` at `http://127.0.0.1:8000`.
The browser demo uses same-origin voice-agent proxy endpoints for perception state and audio, so the standard UI path does not require browser CORS on `voice-perception`. If you bypass those proxy endpoints and call `voice-perception` directly from a browser, the companion service must be running and CORS-enabled for that origin.

### 1. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=xi-...
ELEVENLABS_AGENT_ID=
VOICE_PERCEPTION_URL=http://127.0.0.1:8000
PERCEPTION_LANGUAGE=en
OPENAI_MODEL=gpt-4o-mini
```

Leave `ELEVENLABS_AGENT_ID` blank until the agent is created. Keep all keys server-side only.

### 2. Install with uv

```bash
uv sync
```

`uv sync` creates `.venv`, installs the runtime dependencies, and installs the
`src/` package so `voice_agent` is importable from the repo root. No
`PYTHONPATH` export is needed.

### 3. Start voice-agent

```bash
uv run uvicorn voice_agent.main:app --port 8001 --reload
```

Keep this terminal running. Health check: <http://localhost:8001/health>.

### 4. Expose the local webhook

In a second terminal:

```bash
ngrok http 8001
```

Copy the HTTPS URL, for example `https://abc123.ngrok-free.app`.

### 5. Create or update the ElevenLabs agent

First time:

```bash
python3 scripts/elevenlabs_agent.py create https://abc123.ngrok-free.app
```

If you already have an `ELEVENLABS_AGENT_ID` in `.env`, update the existing agent instead:

```bash
python3 scripts/elevenlabs_agent.py update-url https://abc123.ngrok-free.app
```

The script configures the Conversational AI agent with German output, `eleven_flash_v2_5` TTS, the Custom LLM URL `<ngrok-url>/v1/chat/completions`, no Custom LLM auth for the unauthenticated local demo webhook, and the `perception_session_id` dynamic variable. It reads `ELEVENLABS_API_KEY` from `.env` or the environment and never prints the key.

Runtime note: the browser passes `perception_session_id` only as an ElevenLabs dynamic variable. Some agents reject runtime Custom LLM extra-body overrides. If ElevenLabs omits the dynamic variable from the webhook, voice-agent falls back to the single active browser-started perception session for the local demo path.

### 6. Save the agent ID and restart

After `create`, copy the printed line into `.env`:

```bash
ELEVENLABS_AGENT_ID=agent_...
```

Restart `uvicorn` so the browser session endpoint returns the agent ID.

### 7. Run the browser demo

Open <http://localhost:8001>, click **Start**, grant microphone access, and speak English. Frau Weber should respond in German while adapting to the live perception state.

## Refreshing ngrok URLs

Free ngrok URLs change after restart. Do not create a new ElevenLabs agent for that. Keep the same `ELEVENLABS_AGENT_ID` in `.env` and run:

```bash
python3 scripts/elevenlabs_agent.py update-url https://new-url.ngrok-free.app
```

Then continue using the same browser demo.

## Local validation

From the repo root:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall src scripts tests
uv run python scripts/test_perception_client.py
uv run python scripts/test_webhook.py
```

`test_perception_client.py` requires the companion service. `test_webhook.py`
exercises the OpenAI webhook path and requires `OPENAI_API_KEY`.

## Optional ElevenLabs MCP tooling

Use the direct `scripts/elevenlabs_agent.py` workflow above for normal setup. MCP is optional convenience only. If you use MCP, keep real API keys out of the committed `.mcp.json`; use private client config or environment variables instead. Generated MCP output belongs under `.elevenlabs-mcp-output/`, which is ignored by git.
