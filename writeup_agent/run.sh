#!/bin/bash
# Launch the Writeup report-writer agent (absolute script path so agent_radar can
# read this file and detect the `pydantic_ai` import).
set -e
cd "$(dirname "$0")"

# auto-setup: create venv on first run, always sync requirements
if [ ! -x ".venv/bin/python" ]; then
  echo "📦 First run — creating venv…"
  python3 -m venv .venv
  ./.venv/bin/pip install --quiet --upgrade pip
fi
./.venv/bin/pip install --quiet -r requirements.txt

LOG="${TMPDIR:-/tmp}/writeup-agent.log"
echo "📝 Starting Writeup report-writer agent (background) on http://localhost:${WRITEUP_PORT:-8107}"
echo "   (free / local Ollama model: ${WRITEUP_MODEL:-qwen2.5:7b})"
nohup ./.venv/bin/python "$PWD/writeup_agent.py" > "$LOG" 2>&1 &
echo "✅ running — PID $!   logs: $LOG"
echo "   stop: pkill -f '$PWD/writeup_agent.py'"
