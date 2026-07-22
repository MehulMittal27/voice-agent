# voice-agent

Voice Agent service for a live German recognition-office clerk demo using ElevenLabs Conversational AI, Claude, and the companion voice-perception service.

## ElevenLabs MCP setup

This repo includes project-level MCP configuration in `.mcp.json` for the official ElevenLabs MCP server (`elevenlabs/elevenlabs-mcp`). It uses `uvx elevenlabs-mcp` so each developer can start the server without committing machine-local MCP setup.

### 1. Install prerequisites

Install `uv` if `uvx` is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx --version
```

### 2. Configure local secrets

Copy the example environment file and add your local keys:

```bash
cp .env.example .env
```

Set `ELEVENLABS_API_KEY` in `.env` from <https://elevenlabs.io/app/settings/api-keys>. Do not paste the key into chat, commit it, or expose it in static/client-side code. The ElevenLabs key is for local/server-side tooling only.

### 3. Use the project MCP config

MCP clients that support project config can load `.mcp.json` directly. The configured server is:

```json
{
  "mcpServers": {
    "ElevenLabs": {
      "command": "uvx",
      "args": ["elevenlabs-mcp"],
      "env": {
        "ELEVENLABS_MCP_BASE_PATH": ".elevenlabs-mcp-output",
        "ELEVENLABS_MCP_OUTPUT_MODE": "files"
      }
    }
  }
}
```

The committed config intentionally does not include `ELEVENLABS_API_KEY`. Make sure your MCP client loads `.env` or otherwise passes the local `ELEVENLABS_API_KEY` only on your machine. For clients that do not inherit environment variables from the project, copy the server block into your private user-level MCP config and add the key there, keeping that private config untracked.

Generated files are written under `.elevenlabs-mcp-output/`, which is ignored by git.

### 4. Verify locally

Without using credits, confirm the official MCP package installs and can print a client config with a placeholder key:

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
