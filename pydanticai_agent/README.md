# PydanticAI Runtime Agent

A **free, fully-working PydanticAI agent** that runs as a long-lived HTTP **runtime** — the AWCP
"agent on an existing runtime" model — built so the `agent_radar` registry **auto-detects it**
as a running `agent_framework`.

- **PydanticAI** agent with a real tool (`multiply`) for genuine tool-calling.
- **Free / local** — talks to **Ollama's OpenAI-compatible** endpoint (`/v1`). No API keys.
- **Runtime** — a persistent FastAPI service, so a process-scanning registry can see it.

## Run

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
# needs Ollama running with the model:  ollama pull llama3.1:8b
./run.sh                       # http://localhost:8102
```

Try it:
```bash
curl -XPOST localhost:8102/invoke -H 'Content-Type: application/json' \
  -d '{"input":"What is 12 times 8? Use the multiply tool."}'
# -> {"output":"... 96"}
```

Endpoints: `GET /` (info), `GET /health`, `POST /invoke {input}`.

## How `agent_radar` detects it

Launched as `python <absolute>/agent_runtime.py`, with `from pydantic_ai import Agent` at the
top. The radar reads the script, sees the `pydantic_ai` import, and registers it as
`kind=agent_framework, framework=pydantic_ai` (detected_via `script_import`), then onboards it
via a Temporal workflow and marks it **quarantined** (no telemetry/policy hooks).

> A `pydantic_ai` signature was added to `agent_radar` (in `detectors/base.py`) so this
> framework is recognized alongside LangGraph/CrewAI/AutoGen/LlamaIndex.

## Config (env)
- `PAI_MODEL` (default `llama3.1:8b`).
- `OLLAMA_BASE` (default `http://localhost:11434`).
- `PAI_PORT` (default `8102`).

## Notes
- Standalone project — independent of the other folders.
- `run.sh` uses an **absolute** script path on purpose so the radar (running from a different
  working directory) can read this file and detect the framework import.
