#!/bin/bash
# Launch the LangGraph runtime agent.
# NOTE: launched with an ABSOLUTE script path so a process-scanning registry
# (agent_radar) can read this file and detect the `langgraph` import.
set -e
cd "$(dirname "$0")"

# auto-setup: create venv on first run, always sync requirements
if [ ! -x ".venv/bin/python" ]; then
  echo "📦 First run — creating venv…"
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
fi
./.venv/bin/pip install --quiet -r requirements.txt

LOG="${TMPDIR:-/tmp}/langgraph-agent.log"
echo "🧠 Starting LangGraph runtime agent (background) on http://localhost:${LG_PORT:-8100}"
echo "   (free / local Ollama model: ${LG_MODEL:-llama3.1:8b})"
nohup ./.venv/bin/python "$PWD/agent_runtime.py" > "$LOG" 2>&1 &
echo "✅ running — PID $!   logs: $LOG"
echo "   stop: pkill -f '$PWD/agent_runtime.py'"
