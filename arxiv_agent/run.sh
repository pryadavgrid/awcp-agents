#!/bin/bash
# Launch the arXiv research agent (absolute script path so agent_radar can read
# this file and detect the `langgraph` import).
set -e
cd "$(dirname "$0")"

# auto-setup: create venv on first run, always sync requirements
if [ ! -x ".venv/bin/python" ]; then
  echo "📦 First run — creating venv…"
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
fi
./.venv/bin/pip install --quiet -r requirements.txt

LOG="${TMPDIR:-/tmp}/arxiv-agent.log"
echo "📚 Starting arXiv research agent (background) on http://localhost:${ARXIV_PORT:-8103}"
echo "   (free / local Ollama model: ${ARXIV_MODEL:-llama3.1:8b} · free arXiv API)"
nohup ./.venv/bin/python "$PWD/agent_runtime.py" > "$LOG" 2>&1 &
echo "✅ running — PID $!   logs: $LOG"
echo "   stop: pkill -f '$PWD/agent_runtime.py'"
