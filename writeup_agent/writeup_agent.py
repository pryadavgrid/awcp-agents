"""An autonomous, governed PydanticAI WORKER runtime — the REPORT-WRITER agent.

This is the AWCP magazine's "reliable agent" proof slice, dropped in as a new
folder (Operating Model Step 01: register + attach control hooks; Scenario C:
quarantine -> active once telemetry / flags / policy callbacks are observed).

Pulls GOALS (a topic) off the task queue and executes each in multiple steps:
  - read/compute tools: web_search, current_time, ...   (gather facts)
  - save_artifact  -> governed LOCAL write  (the report file; gated)
  - external_post  -> governed EXTERNAL write (high risk, gated + needs approval)

The write is the point: every report it produces is a write-capable action that
must pass the radar's gate (Operating Model Step 03). Because this agent is rated
`high` in the magazine, the persist step pauses for a narrow, expiring approval
token (Scenario B) before the report lands.

Queue/worker/governance/approval/UI all live in awcp_kit; this file only supplies
the PydanticAI agent + the run_goal() hook, exactly like the sibling agents —
nothing about governance is re-implemented here.

Run as:  python writeup_agent.py   (absolute path via run.sh so the detector sees
the `pydantic_ai` import and classifies the framework).
"""

import os

from pydantic_ai import Agent  # noqa: F401  (import marks this as PydanticAI)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from fastapi import FastAPI
import uvicorn

import awcp_kit as kit

MODEL = os.getenv("WRITEUP_MODEL", "qwen2.5:7b")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
PORT = int(os.getenv("WRITEUP_PORT", "8107"))
HERE = os.path.dirname(os.path.abspath(__file__))

# The agent's OUTBOX: every email this agent drafts lands here (via the MCP
# server's governed draft_email tool — see AWCP_EMAIL_DRAFT_DIR). Created up
# front so the folder exists even before the first draft.
OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IMPORTANT: this exact string is the agent's registered name AND the magazine
# key (awcp_magazine.json -> "writeup-writer"). policy.assigned_risk_for matches
# the magazine by exact name, so changing one without the other drops the agent
# to the __default__ (high) profile. Keep them in lock-step.
AGENT_NAME = "writeup-writer"

SYSTEM = (
    "You are a REPORT-WRITER agent. Given a GOAL (a topic or question), gather the "
    "facts you need (use web_search for anything you don't know, current_time for "
    "dates), then WRITE a clear, well-structured markdown report: a short title, 2-4 "
    "section headings, tight paragraphs, and a one-line summary at the end. "
    "When the report is ready you MUST persist it: call save_artifact(name, content) "
    "with a short kebab-case name and the full markdown as content. "
    "If the goal asks to EMAIL or SEND the report to someone, then ALSO call "
    "draft_email(to, subject, body) with that recipient, a short subject, and the "
    "report as the body — this saves a DRAFT into the agent's output/ folder for a "
    "human to review; nothing is actually sent. Both writes are governed actions — "
    "they may pause for operator approval; that is expected. "
    "After saving (and drafting the email if asked), return the full markdown report "
    "as your final answer.")

_model = OpenAIModel(MODEL, provider=OpenAIProvider(base_url=f"{OLLAMA_BASE}/v1", api_key="ollama"))

# --- tools: discovered dynamically from the MCP server (NONE defined here) ----
# No tools are declared in this file. The agent fetches the MCP server's catalog
# and binds it; every call runs on the server (governed + traced). save_artifact /
# external_post are advertised by the server and routed through the gate.
_specs = kit.discover_tools()
TOOLS = kit.build_tools("pydantic_ai", _specs)
TOOL_NAMES = [s["name"] for s in _specs]

AGENT = Agent(_model, system_prompt=SYSTEM, tools=TOOLS)


def _tools_from_messages(messages) -> list[str]:
    used: list[str] = []
    for m in messages or []:
        for part in getattr(m, "parts", []) or []:
            if getattr(part, "part_kind", "") == "tool-call":
                n = getattr(part, "tool_name", None)
                if n and n not in used:
                    used.append(n)
    return used


def run_goal(goal: str) -> dict:
    res = AGENT.run_sync(goal)
    out = getattr(res, "output", None)
    if out is None:
        out = getattr(res, "data", None)
    return {"result": str(out), "tools_used": _tools_from_messages(res.all_messages())}


app = FastAPI(title="Writeup Report-Writer Worker Runtime")

if __name__ == "__main__":
    kit.mount(
        app,
        meta={"agent": AGENT_NAME, "framework": "pydantic_ai",
              "model": MODEL, "tools": TOOL_NAMES, "dir": HERE,
              "purpose": "Report-writer — turns a topic into a governed markdown report; "
                         "email drafts land in output/ for human review (write-gated).",
              "format": "markdown", "accent": "#7A2E1E", "logo": "\U0001F4DD",
              "examples": ["Write a short report on the state of agent governance.",
                           "Draft a 3-section brief on retrieval-augmented generation.",
                           "Write a report on AI safety and email it to alice@example.com."]},
        run_goal=run_goal,
        port=PORT,
    )
    print(f"📝 Writeup REPORT-WRITER  →  http://localhost:{PORT}   (model={MODEL})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
