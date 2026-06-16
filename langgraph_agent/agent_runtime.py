"""An autonomous, governed LangGraph WORKER runtime (AWCP "agent on a runtime").

Not a chatbot: it pulls GOALS off a task queue and executes each one in multiple
steps using its tools, performing real governed WRITES:
  - read/compute tools: web_search, multiply, add, power, word_count, current_time
  - save_artifact  -> governed LOCAL write  (medium risk, gated)
  - external_post   -> governed EXTERNAL write (high risk, gated + needs approval)
The task queue, worker loop, governance, approval flow and UI live in awcp_kit;
this file only supplies the framework agent + the run_goal() hook.

Run as:  python agent_runtime.py   (launched with an ABSOLUTE path by run.sh so
the detector can read this file and see the `langgraph` import).
"""

import datetime
import os

from langgraph.graph import StateGraph  # noqa: F401  (import marks this as LangGraph)
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from fastapi import FastAPI
import uvicorn

import awcp_kit as kit

MODEL = os.getenv("LG_MODEL", "llama3.1:8b")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
PORT = int(os.getenv("LG_PORT", "8100"))
HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = (
    "You are an autonomous worker agent. You are given a GOAL and must accomplish "
    "it using your tools, deciding for yourself which tools apply. Use web_search "
    "for facts you do not know, the math tools for arithmetic. When you have "
    "produced a result, persist it with save_artifact, and if the goal asks you to "
    "submit/send/publish it, call external_post. Do not call a tool unless it helps. "
    "Base your answer on what your tools return; never output a raw tool call as text."
)


# --- read / compute tools -------------------------------------------------
@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current/real-world information (no API key)."""
    return kit.web_search(query, max_results)


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base ** exponent


@tool
def word_count(text: str) -> int:
    """Count the words in a piece of text."""
    return len(text.split())


@tool
def current_time() -> str:
    """Return the current local date/time (ISO-8601)."""
    return datetime.datetime.now().isoformat(timespec="seconds")


# --- governed WRITE tools (routed through the control-plane gate) ----------
@tool
def save_artifact(name: str, content: str) -> str:
    """Save a result artifact to disk. GOVERNED local write (gated by the control
    plane). Use when you have a result worth persisting."""
    return kit.save_artifact(name, content)


@tool
def external_post(summary: str) -> str:
    """Submit/publish a result to an external system over HTTP. HIGH-RISK governed
    write: it is gated AND pauses for operator approval before sending. Use only
    when the goal asks to submit, send, publish, or report the result externally."""
    return kit.external_post(summary)


TOOLS = [web_search, multiply, add, power, word_count, current_time,
         save_artifact, external_post]
TOOL_NAMES = [t.name for t in TOOLS]

_llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE, temperature=0)
AGENT = create_react_agent(_llm, tools=TOOLS)


def run_goal(goal: str) -> dict:
    """Framework hook: execute one goal end-to-end (multi-step) and return the
    final result + the tools it used. Governed writes happen inside the tools."""
    result = AGENT.invoke({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": goal},
    ]})
    msgs = result["messages"]
    tools_used = [tc["name"] for m in msgs for tc in (getattr(m, "tool_calls", None) or [])]
    return {"result": msgs[-1].content, "tools_used": tools_used}


app = FastAPI(title="LangGraph Worker Runtime")

if __name__ == "__main__":
    kit.mount(
        app,
        meta={"agent": "LangGraph Orchestrator", "framework": "langgraph",
              "model": MODEL, "tools": TOOL_NAMES, "dir": HERE,
              "purpose": "General research & compute orchestrator — multi-step web + math, then a clear written answer.",
              "format": "markdown", "accent": "#7c5cff", "logo": "\U0001F9E0",
              "examples": ["What is 25 × 4? Report it.",
                           "Research who won the 2024 Booker Prize and summarise it.",
                           "Find the population of Canada and Japan, then compare them."]},
        run_goal=run_goal,
        port=PORT,
    )
    print(f"🧠 LangGraph WORKER  →  http://localhost:{PORT}   (model={MODEL})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
