# voice-agent

Live voice demo for a German recognition-office clerk using ElevenLabs Conversational AI, OpenAI, and the companion `voice-perception` service.

## Quickstart

Run commands from the repo root.

### 0. Prerequisites

- Python 3.11
- An OpenAI API key and an ElevenLabs API key
- `ngrok` or another HTTPS tunnel for ElevenLabs to reach your local webhook
- The companion `voice-perception` service cloned and running separately: <https://github.com/mehulmittal27/voice-perception>

By default this app expects `voice-perception` at `http://127.0.0.1:8000`.

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

### 2. Create and activate a virtualenv

```bash
python -m venv .venv && source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start voice-agent

```bash
export PYTHONPATH=src
uvicorn voice_agent.main:app --port 8001 --reload
```

Keep this terminal running. Health check: <http://localhost:8001/health>.

### 5. Expose the local webhook

In a second terminal:

```bash
ngrok http 8001
```

Copy the HTTPS URL, for example `https://abc123.ngrok-free.app`.

### 6. Create or update the ElevenLabs agent

First time:

```bash
python3 scripts/elevenlabs_agent.py create https://abc123.ngrok-free.app
```

If you already have an `ELEVENLABS_AGENT_ID` in `.env`, update the existing agent instead:

```bash
python3 scripts/elevenlabs_agent.py update-url https://abc123.ngrok-free.app
```

The script configures the Conversational AI agent with German output, `eleven_flash_v2_5` TTS, the Custom LLM URL `<ngrok-url>/v1/chat/completions`, and the `perception_session_id` dynamic variable. It reads `ELEVENLABS_API_KEY` from `.env` or the environment and never prints the key.

### 7. Save the agent ID and restart

After `create`, copy the printed line into `.env`:

```bash
ELEVENLABS_AGENT_ID=agent_...
```

Restart `uvicorn` so the browser session endpoint returns the agent ID.

### 8. Run the browser demo

Open <http://localhost:8001>, click **Start**, grant microphone access, and speak English. Frau Weber should respond in German while adapting to the live perception state.

## Refreshing ngrok URLs

Free ngrok URLs change after restart. Do not create a new ElevenLabs agent for that. Keep the same `ELEVENLABS_AGENT_ID` in `.env` and run:

```bash
python3 scripts/elevenlabs_agent.py update-url https://new-url.ngrok-free.app
```

Then continue using the same browser demo.

## Local validation

With the virtualenv active and `PYTHONPATH=src` set:

```bash
python3 -m unittest discover -s tests -v
python scripts/test_perception_client.py
python scripts/test_webhook.py
```

`test_webhook.py` exercises the OpenAI webhook path and requires `OPENAI_API_KEY`.

## Optional ElevenLabs MCP tooling

Use the direct `scripts/elevenlabs_agent.py` workflow above for normal setup. MCP is optional convenience only. If you use MCP, keep real API keys out of the committed `.mcp.json`; use private client config or environment variables instead. Generated MCP output belongs under `.elevenlabs-mcp-output/`, which is ignored by git.
