"""An autonomous, governed LangGraph WORKER runtime (AWCP "agent on a runtime").

Not a chatbot: it pulls GOALS off a task queue and executes each one in multiple
steps using its tools, performing real governed WRITES:
  - read/compute tools: web_search, multiply, add, power, word_count, current_time
  - save_artifact  -> governed LOCAL write  (medium risk, gated)
  - external_post   -> governed EXTERNAL write (high risk, gated + needs approval)
The task queue, worker loop, governance, approval flow and UI live in awcp_kit;
this file only supplies the framework agent + the run_goal() hook.

Run as:  python langgraph_agent.py   (launched with an ABSOLUTE path by run.sh so
the detector can read this file and see the `langgraph` import).
"""

import json
import os

from langgraph.graph import StateGraph  # noqa: F401  (import marks this as LangGraph)
from langgraph.prebuilt import create_react_agent
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


# --- tools: discovered dynamically from the MCP server (NONE defined here) ----
# This agent declares no tools of its own. At startup it asks the MCP server which
# tools it offers (list_runtime_tools) and binds them; every call runs on the
# server, governed by the radar gate and traced into Temporal + OTel. Add a tool
# to the server's tools/ folder and it shows up here automatically.
TOOLS = kit.build_tools("langgraph")
TOOL_NAMES = [t.name for t in TOOLS]

_llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE, temperature=0)
# Same model constrained to emit valid JSON — used to pull a tool's arguments out
# of a plain-language "call <tool> with ..." instruction (see _extract_args).
_json_llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE, temperature=0, format="json")
AGENT = create_react_agent(_llm, tools=TOOLS)


# ── Direct tool invocation ────────────────────────────────────────────────────
# When the user explicitly says "call <tool> with <args> and return the raw
# result", the small local model is unreliable: it often narrates the call as text
# instead of emitting it, and when it does run the tool it summarises the output
# instead of returning it. For that request we bypass the ReAct loop and invoke the
# named tool DIRECTLY (still governed — the tool forwards through the MCP pipeline),
# then return its raw output verbatim.
_RAW_HINTS = ("raw result", "raw output", "raw json", "raw response", "verbatim",
              "exact output", "exactly as", "unmodified", "without summar",
              "don't summar", "do not summar", "return the result", "full output")
_CALL_VERBS = ("call ", "invoke ", "execute ")


def _strip_fences(text: str) -> str:
    """Drop a leading ```lang line and trailing ``` from a fenced block."""
    t = (text or "").strip()
    if t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _deep_json(v):
    """Recursively expand string values that are THEMSELVES JSON objects/arrays
    (optionally ```-fenced) into real structure. Lossless: nothing is dropped and
    scalar strings (passwords, postcodes) are left untouched — it only makes an
    envelope whose scraped content is a JSON string (e.g. firecrawl's `markdown`
    field) render as readable nested JSON instead of one escaped blob."""
    if isinstance(v, str):
        inner = _strip_fences(v).strip()
        if inner[:1] in ("{", "[") or v.strip().startswith("```"):
            try:
                return _deep_json(json.loads(inner))
            except Exception:  # noqa: BLE001
                return v
        return v
    if isinstance(v, dict):
        return {k: _deep_json(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_deep_json(x) for x in v]
    return v


def _pretty(content) -> str:
    """Pretty-print `content` if it is (or contains) JSON; else return it as-is."""
    s = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    try:
        return json.dumps(_deep_json(json.loads(_strip_fences(s))), indent=2, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return s.strip()


def _as_block(content) -> str:
    """Render raw tool output for the markdown UI: pretty value in a fenced block,
    with a fence long enough to survive any backtick run inside the content."""
    pretty = _pretty(content)
    lang = ""
    try:
        json.loads(pretty)
        lang = "json"
    except Exception:  # noqa: BLE001
        pass
    longest = cur = 0
    for ch in pretty:
        cur = cur + 1 if ch == "`" else 0
        longest = max(longest, cur)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{pretty}\n{fence}"


def _named_tool_in_goal(goal: str):
    """The bound tool whose exact name appears as a token in the goal (longest wins),
    or None. Token match avoids e.g. the `add` tool matching the word 'address'."""
    cleaned = "".join(c if (c.isalnum() or c == "_") else " " for c in (goal or "").lower())
    tokens = set(cleaned.split())
    best = None
    for t in TOOLS:
        if t.name.lower() in tokens and (best is None or len(t.name) > len(best.name)):
            best = t
    return best


def _wants_raw(goal: str) -> bool:
    g = (goal or "").lower()
    return any(h in g for h in _RAW_HINTS)


def _direct_target(goal: str):
    """The tool to invoke directly, if the goal is an explicit 'call <tool> …'
    (optionally 'raw') request; else None (use the normal ReAct agent)."""
    tool = _named_tool_in_goal(goal)
    if tool is None:
        return None
    g = (goal or "").lower()
    return tool if (_wants_raw(goal) or any(v in g for v in _CALL_VERBS)) else None


def _extract_args(goal: str, tool) -> dict | None:
    """Pull the named tool's call arguments out of the plain-language goal, using the
    JSON-constrained model. Returns a dict filtered to the tool's real parameters,
    or None if extraction failed."""
    arg_names = list(getattr(tool, "args", {}) or {})
    prompt = (
        f"Extract the arguments for calling the function `{tool.name}` from the "
        f"INSTRUCTION. The function parameters are: {arg_names or 'unknown'}. Reply "
        f"with ONLY a JSON object mapping parameter names to values; include only "
        f"parameters that actually have a value in the instruction.\n"
        f"INSTRUCTION: {goal}")
    try:
        data = json.loads(_strip_fences(str(_json_llm.invoke(prompt).content)))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    return {k: v for k, v in data.items() if not arg_names or k in arg_names}


def _invoke_directly(goal: str, tool):
    """Run the tool directly (governed via the MCP forwarder). Raw output, or None
    if we couldn't build the args / the tool errored (caller then falls back)."""
    args = _extract_args(goal, tool)
    if args is None:
        return None
    try:
        return tool.invoke(args)
    except Exception:  # noqa: BLE001
        return None


def _tool_outputs(msgs) -> list:
    """(name, content) for each tool result message, in the order they were run."""
    return [(getattr(m, "name", "") or "", m.content)
            for m in msgs if getattr(m, "type", "") == "tool"]


def run_goal(goal: str) -> dict:
    """Framework hook: execute one goal end-to-end (multi-step) and return the
    final result + the tools it used. Governed writes happen inside the tools."""
    # Explicit "call <tool> … (raw)" request → invoke the tool directly and return
    # its raw output, rather than trusting the flaky ReAct loop to call + not summarise.
    tool = _direct_target(goal)
    if tool is not None:
        raw = _invoke_directly(goal, tool)
        if raw is not None:
            return {"result": _as_block(raw), "tools_used": [tool.name]}
        # extraction/invocation failed — fall through to the normal agent

    result = AGENT.invoke({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": goal},
    ]})
    msgs = result["messages"]
    tools_used = [tc["name"] for m in msgs for tc in (getattr(m, "tool_calls", None) or [])]

    # If the user asked for the RAW result and a tool actually ran, return the tool
    # output verbatim instead of the model's prose summary of it.
    if _wants_raw(goal):
        outs = _tool_outputs(msgs)
        if outs:
            return {"result": _as_block(outs[-1][1]), "tools_used": tools_used}

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
