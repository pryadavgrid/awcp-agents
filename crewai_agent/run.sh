#!/bin/bash
# Launch the CrewAI runtime agent.
# Launched with an ABSOLUTE script path so a process-scanning registry
# (agent_radar) can read this file and detect the `crewai` import.
set -e
cd "$(dirname "$0")"

# auto-setup: create venv on first run (crewai requires Python <3.14), always sync requirements
if [ ! -x ".venv/bin/python" ]; then
  echo "📦 First run — creating venv with python3.12…"
  /opt/homebrew/bin/python3.12 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
fi
./.venv/bin/pip install --quiet -r requirements.txt

LOG="${TMPDIR:-/tmp}/crewai-agent.log"
echo "👥 Starting CrewAI runtime agent (background) on http://localhost:${CREW_PORT:-8101}"
echo "   (free / local Ollama model: ${CREW_MODEL:-ollama/llama3.1:8b})"
nohup ./.venv/bin/python "$PWD/agent_runtime.py" > "$LOG" 2>&1 &
echo "✅ running — PID $!   logs: $LOG"
echo "   stop: pkill -f '$PWD/agent_runtime.py'"
