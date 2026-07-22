# voice-agent

Voice Agent service for a live German recognition-office clerk demo using ElevenLabs Conversational AI, OpenAI, and the companion voice-perception service.

## ElevenLabs MCP setup

This repo includes project-level MCP configuration in `.mcp.json` for the official ElevenLabs MCP server (`elevenlabs/elevenlabs-mcp`). The config follows the official `uvx elevenlabs-mcp` pattern from <https://github.com/elevenlabs/elevenlabs-mcp>.

### 1. Install prerequisites

Install `uv` if `uvx` is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx --version
```

### 2. Configure local secrets

Copy the example environment file and add your local server-side keys:

```bash
cp .env.example .env
```

Set `ELEVENLABS_API_KEY` in `.env` from <https://elevenlabs.io/app/settings/api-keys>. Also set `OPENAI_API_KEY` for the FastAPI Custom LLM webhook and choose `OPENAI_MODEL` if you do not want the default. Do not paste keys into chat, commit them, or expose them in static/client-side code. Keys are for local/server-side tooling only.

The committed `.mcp.json` contains the official placeholder `"<insert-your-api-key-here>"`, not a real key. Replace that placeholder only in your private MCP client config, or use your MCP client's secret/environment-variable support to pass `ELEVENLABS_API_KEY` locally.

### 3. Use the project MCP config

The official quickstart config is `command: "uvx"`, `args: ["elevenlabs-mcp"]`, and an `env` block containing `ELEVENLABS_API_KEY`. This repo's `.mcp.json` keeps the same shape but uses a placeholder key so secrets stay out of git:

```json
{
  "mcpServers": {
    "ElevenLabs": {
      "command": "uvx",
      "args": ["elevenlabs-mcp"],
      "env": {
        "ELEVENLABS_API_KEY": "<insert-your-api-key-here>",
        "ELEVENLABS_MCP_BASE_PATH": ".elevenlabs-mcp-output",
        "ELEVENLABS_MCP_OUTPUT_MODE": "files"
      }
    }
  }
}
```

Before connecting an MCP client, copy the server block into private client configuration and replace the placeholder with your local key, or configure the client to inject `ELEVENLABS_API_KEY` from a private secret store. Keep that private config untracked.

Generated files are written under `.elevenlabs-mcp-output/`, which is ignored by git. These optional output settings are documented by the official MCP server.

### 4. Verify locally

Without using credits, confirm the official MCP package installs and can print a client config with a placeholder key:

```bash
pip install elevenlabs-mcp
python -m elevenlabs_mcp --api-key=dummy --print
```

If you prefer not to install the package globally, this equivalent `uvx` check is also useful:

```bash
uvx --from elevenlabs-mcp python -m elevenlabs_mcp --api-key=dummy --print
```

After adding a valid `ELEVENLABS_API_KEY`, verify the key can reach ElevenLabs without printing it:

```bash
python - <<'PY'
import os
import sys
import urllib.request
from pathlib import Path

if not os.environ.get("ELEVENLABS_API_KEY") and Path(".env").exists():
    for line in Path(".env").read_text().splitlines():
        if line.startswith("ELEVENLABS_API_KEY="):
            os.environ["ELEVENLABS_API_KEY"] = line.split("=", 1)[1].strip()
            break

key = os.environ.get("ELEVENLABS_API_KEY")
if not key or key.startswith("xi-your"):
    sys.exit("ELEVENLABS_API_KEY is not configured in the environment or .env")

request = urllib.request.Request(
    "https://api.elevenlabs.io/v1/user",
    headers={"xi-api-key": key},
)
with urllib.request.urlopen(request, timeout=20) as response:
    print(f"ElevenLabs reachable: HTTP {response.status}")
PY
```

Then open your MCP client and confirm the `ElevenLabs` server connects and lists tools. Tool calls may consume ElevenLabs credits.
