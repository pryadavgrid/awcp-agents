#!/bin/bash
# Launch the Auto (Smart Router) agent (absolute script path so agent_radar can
# read this file and detect the `langgraph` import).
set -e
cd "$(dirname "$0")"
export AGENT_RADAR_URL="${AGENT_RADAR_URL:-http://localhost:8000}"  # register with the gateway-mounted radar

# auto-setup: create venv on first run, always sync requirements
if [ ! -x ".venv/bin/python" ]; then
  echo "📦 First run — creating venv…"
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
fi
./.venv/bin/pip install --quiet -r requirements.txt

LOG="${TMPDIR:-/tmp}/auto-agent.log"
echo "🧭 Starting Auto (Smart Router) agent (background) on http://localhost:${AUTO_PORT:-8106}"
echo "   (routes your query to the best specialist agent · local Ollama model: ${AUTO_MODEL:-llama3.1:8b})"
nohup ./.venv/bin/python "$PWD/auto_agent.py" > "$LOG" 2>&1 &
echo "✅ running — PID $!   logs: $LOG"
echo "   stop: pkill -f '$PWD/auto_agent.py'"
