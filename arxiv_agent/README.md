# arXiv Research Agent (LangGraph runtime)

A **free, fully-working arXiv research agent** that runs as a long-lived HTTP **runtime** — the
AWCP "agent on an existing runtime" model — built so a process-scanning registry like
`agent_radar` **auto-detects it** (as `agent_framework` / LangGraph).

- **LangGraph** prebuilt ReAct agent over a **local Ollama** model (no API keys).
- Real tools on the **free arXiv API**: `search_arxiv(query, max_results)` and `get_paper(id)`.
- **Runtime** — a persistent FastAPI service.

## Run

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
# needs Ollama with a tool-capable model:  ollama pull llama3.1:8b
# needs internet (queries arxiv.org)
./run.sh                       # http://localhost:8103
```

Try it:
```bash
curl -XPOST localhost:8103/invoke -H 'Content-Type: application/json' \
  -d '{"input":"Find 2 recent papers on graph neural networks for drug discovery and summarize them."}'
# -> {"output":"...paper titles + links + summaries...","tools_used":["search_arxiv"]}
```

Endpoints: `GET /` (info), `GET /health`, `POST /invoke {input}`.

## Checking detection (manual)
Start this agent, then look at `agent_radar` (http://localhost:8090): it should appear within a
few seconds as `kind=agent_framework`, `framework=langgraph`, `detected_via=script_import`,
`status=quarantined`, onboarded via a Temporal workflow. (It registers as *langgraph* because
that's the framework it's built on; the "arXiv" part is in its tools.)

## Config (env)
- `ARXIV_MODEL` (default `llama3.1:8b`).
- `OLLAMA_BASE` (default `http://localhost:11434`).
- `ARXIV_PORT` (default `8103`).

## Notes
- Standalone — not linked to anything. Free end-to-end (local model + free arXiv API).
- `run.sh` launches with an **absolute** script path so the radar (different working directory)
  can read this file and detect the framework import.
