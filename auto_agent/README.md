# Auto (Smart Router) Agent (LangGraph runtime)

The **"Auto" mode** agent. It answers nothing itself and uses **no tools** — it reads
your query, judges which of the *other* specialist agents fits best, and runs that
agent for you through the governed control plane. It runs as a long-lived HTTP
**runtime** — the AWCP "agent on an existing runtime" model — auto-detected by a
process-scanning registry like `agent_radar`.

- **LangGraph / local Ollama** model (no API keys) for the internal routing judge.
- **No tools bound.** Its only outward action is a governed **agent call**
  (`POST /user/ask` on the gateway) — the chosen agent then runs its own full
  governed MCP-tool pipeline (radar gate → Temporal → `execute_tool`).
- **Fully dynamic** — the routable set is discovered live from the control plane
  (`GET /user/agents`); dropping in a new agent folder makes it eligible with no
  code change. The router excludes itself and any other router (`framework=router`).
- The user-facing result shows **which agent was chosen** plus that agent's
  formatted output — never the internal judging.

## Run

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
# needs Ollama with a tool-capable model:  ollama pull llama3.1:8b
./run.sh                       # http://localhost:8106
```

Endpoints: `GET /` (task console), `GET /info`, `GET /health`,
`POST /tasks {goal}`, `GET /tasks/{id}`.

## How it appears in the user UI
Discovered automatically by the gateway (any folder with a `run.sh` is an agent),
so it shows in the agent picker as **"Auto (Smart Router)"** with an **AUTO** badge.
Selecting it shows a light note that this mode uses more credits than picking an
agent yourself — it runs a routing step *and then* the chosen agent. The single
signal the UI reads is the exposed `framework: "router"` field; nothing is
hardcoded and no gateway/MCP code is changed.

## Config (env)
- `AUTO_MODEL` (default `llama3.1:8b`) — model used for the routing judge.
- `OLLAMA_BASE` (default `http://localhost:11434`).
- `AUTO_PORT` (default `8106`).
- `AUTO_DELEGATE_TIMEOUT` (default `330`) — how long to wait on the delegated run.
- `AGENT_RADAR_URL` (default `http://localhost:8000`) — the gateway/control plane
  it discovers agents from and delegates through.

## Notes
- Uses more credits than a single agent: it spends its own routing-judge tokens
  **and** the chosen agent's tokens (each metered under its own agent).
- The delegated run is sent with an empty session, so it is metered as its own turn
  and does not pollute the chat's history — only the router's turn is recorded.
- `run.sh` launches with an **absolute** script path so the radar (different working
  directory) can read this file and detect the framework import.
