# LangGraph Runtime Agent

A **free, fully-working LangGraph agent** that runs as a long-lived HTTP **runtime** — the kind
of "agent on an existing runtime" the AWCP magazine describes. It's built so that the
`agent_radar` registry **auto-detects it** as a running `agent_framework`.

- **LangGraph** prebuilt **ReAct agent** with two real tools (`multiply`, `current_time`).
- **Free / local** — uses an **Ollama** model (`llama3.1:8b` by default). No API keys.
- **Runtime** — a persistent FastAPI service, so a process-scanning registry can see it.

## Run

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
# needs Ollama running with the model:  ollama pull llama3.1:8b
./run.sh                       # http://localhost:8100
```

Try it:
```bash
curl -XPOST localhost:8100/invoke -H 'Content-Type: application/json' \
  -d '{"input":"What is 23 multiplied by 19? Use your tool."}'
# -> {"output":"... 437","steps":4,"tools_used":["multiply"]}
```

Endpoints: `GET /` (info), `GET /health`, `POST /invoke {input}`.

## How `agent_radar` detects it

It's launched as `python <absolute>/agent_runtime.py`, and that file imports `langgraph` at the
top. The radar's scanner reads the referenced script, sees the `from langgraph...` import, and
registers it as `kind=agent_framework, framework=langgraph` (detected_via `script_import`). It
then onboards via a Temporal workflow and marks it **quarantined** (no telemetry/policy hooks),
exactly as designed.

To see it: run `agent_radar` (`../agent_radar/run.sh`, http://localhost:8090) and start this
agent — it appears in the radar table within a few seconds.

## Config (env)
- `LG_MODEL` (default `llama3.1:8b`) — any tool-capable Ollama model.
- `OLLAMA_BASE` (default `http://localhost:11434`).
- `LG_PORT` (default `8100`).

## Notes
- Standalone project — independent of `awcp_agents` and `agent_radar`.
- `run.sh` launches with an **absolute** script path on purpose, so the radar (running from a
  different working directory) can read this file and detect the framework import.
