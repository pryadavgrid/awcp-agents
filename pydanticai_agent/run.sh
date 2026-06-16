#!/bin/bash
# Launch the PydanticAI runtime agent (absolute script path so agent_radar can
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

LOG="${TMPDIR:-/tmp}/pydanticai-agent.log"
echo "🔷 Starting PydanticAI runtime agent (background) on http://localhost:${PAI_PORT:-8102}"
echo "   (free / local Ollama model: ${PAI_MODEL:-llama3.1:8b})"
nohup ./.venv/bin/python "$PWD/agent_runtime.py" > "$LOG" 2>&1 &
echo "✅ running — PID $!   logs: $LOG"
echo "   stop: pkill -f '$PWD/agent_runtime.py'"
