# CrewAI Runtime Agent

A **free, fully-working CrewAI agent** that runs as a long-lived HTTP **runtime** — the AWCP
magazine's "agent on an existing runtime" model — built so the `agent_radar` registry
**auto-detects it** as a running `agent_framework`.

- **CrewAI** single-agent crew; each request becomes a `Task` the crew runs.
- **Free / local** — uses an **Ollama** model (`ollama/llama3.1:8b` by default). No API keys.
- **Runtime** — a persistent FastAPI service, so a process-scanning registry can see it.
- Telemetry opted out (`CREWAI_TELEMETRY_OPT_OUT`) — no network phone-home.

## Run

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
# needs Ollama running with the model:  ollama pull llama3.1:8b
./run.sh                       # http://localhost:8101
```

Try it:
```bash
curl -XPOST localhost:8101/invoke -H 'Content-Type: application/json' \
  -d '{"input":"In one sentence, what is the capital of Japan?"}'
# -> {"output":"The capital of Japan is Tokyo."}
```

Endpoints: `GET /` (info), `GET /health`, `POST /invoke {input}`.

## How `agent_radar` detects it

Launched as `python <absolute>/agent_runtime.py`, with `from crewai import ...` at the top of
that file. The radar reads the referenced script, sees the `crewai` import, and registers it as
`kind=agent_framework, framework=crewai` (detected_via `script_import`), then onboards it via a
Temporal workflow and marks it **quarantined** (no telemetry/policy hooks).

Run `agent_radar` (`../agent_radar/run.sh`, http://localhost:8090) and start this agent — it
shows up in the radar table within a few seconds.

## Config (env)
- `CREW_MODEL` (default `ollama/llama3.1:8b`).
- `OLLAMA_BASE` (default `http://localhost:11434`).
- `CREW_PORT` (default `8101`).

## Notes
- Standalone project — independent of `awcp_agents`, `agent_radar`, and `langgraph_agent`.
- `run.sh` launches with an **absolute** script path on purpose, so the radar (running from a
  different working directory) can read this file and detect the framework import.
