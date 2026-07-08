"""An autonomous, governed PydanticAI WORKER runtime (AWCP agent-on-a-runtime).

Pulls GOALS off a task queue and executes each in multiple steps:
  - read/compute tools: web_search, multiply, add, word_count, current_time
  - save_artifact  -> governed LOCAL write  (medium risk, gated)
  - external_post   -> governed EXTERNAL write (high risk, gated + needs approval)
Queue/worker/governance/approval/UI live in awcp_kit; this file supplies the
PydanticAI agent + the run_goal() hook.

Run as:  python pydanticai_agent.py   (absolute path via run.sh so the detector sees
the `pydantic_ai` import).
"""

import json
import os

from pydantic_ai import Agent  # noqa: F401  (import marks this as PydanticAI)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from fastapi import FastAPI
import uvicorn

import awcp_kit as kit

MODEL = os.getenv("PAI_MODEL", "llama3.1:8b")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
PORT = int(os.getenv("PAI_PORT", "8102"))
HERE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = (
    "You are a STRUCTURED-DATA agent. Given a GOAL, produce the requested "
    "information as a SINGLE valid JSON object with clear snake_case keys and "
    "concise values. Answer directly from your own knowledge whenever you can; "
    "use a tool ONLY when you genuinely need data you don't have (web_search for "
    "unknown facts, the math tools for arithmetic). "
    "NEVER describe, narrate, or write out a tool call, a function call, code, or a "
    "curl command as your answer — the framework calls tools for you, so you either "
    "let it call the tool or you output the data yourself. Your FINAL reply must be "
    "ONLY the JSON object: no prose, no explanation, no markdown code fences. If the "
    "goal asks for a list of items, wrap them under a top-level key whose value is a "
    "JSON array. If the goal asks to save or submit the result, call save_artifact / "
    "external_post first.")

# Tool-free fallback persona. Small local models, faced with a big tool catalog,
# sometimes reply with a NARRATION of a tool call (code / curl) instead of either
# calling a tool or answering. When the primary (tool-bound) pass yields no clean
# JSON, we re-ask with NO tools and this stricter prompt so the model just answers.
DIRECT_SYSTEM = (
    "You are a data agent. Answer the user's request DIRECTLY from your own "
    "knowledge. Reply with ONLY ONE valid JSON object using snake_case keys — no "
    "prose, no markdown, no code fences, and NEVER write code, function calls, or "
    "curl commands. If the request is a list of items, put them in a JSON array "
    "under a single top-level key.")

_model = OpenAIModel(MODEL, provider=OpenAIProvider(base_url=f"{OLLAMA_BASE}/v1", api_key="ollama"))

# --- tools: discovered dynamically from the MCP server (NONE defined here) ----
# No tools are declared in this file. The agent fetches the MCP server's catalog
# and binds it; every call runs on the server (governed + traced).
_specs = kit.discover_tools()
TOOLS = kit.build_tools("pydantic_ai", _specs)
TOOL_NAMES = [s["name"] for s in _specs]

AGENT = Agent(_model, system_prompt=SYSTEM, tools=TOOLS)
# Same model, no tools — used only to recover a clean answer when the tool-bound
# agent narrates a tool call instead of returning data (see run_goal).
DIRECT_AGENT = Agent(_model, system_prompt=DIRECT_SYSTEM)
# Tool-free helper that ONLY pulls call arguments out of a plain-language
# "call <tool> with ..." instruction (see _extract_args / direct invocation).
ARG_AGENT = Agent(_model, system_prompt=(
    "You extract function-call arguments. Output ONLY a JSON object mapping the "
    "given parameter names to the values found in the instruction; include a "
    "parameter only if the instruction provides its value. No prose, no code fences."))

# Explicit "call <tool> with <args> (and return the raw result)" requests are
# handled by invoking the named tool DIRECTLY (still governed — via kit.call_tool),
# because the small local model otherwise narrates the call or, worse, FABRICATES a
# plausible-looking result instead of actually running the tool.
_RAW_HINTS = ("raw result", "raw output", "raw json", "raw response", "verbatim",
              "exact output", "exactly as", "unmodified", "without summar",
              "don't summar", "do not summar", "return the result", "full output")
_CALL_VERBS = ("call ", "invoke ", "execute ")


def _strip_fences(text: str) -> str:
    """Drop a leading ```lang line and a trailing ``` if the model wrapped its
    JSON in a markdown code fence."""
    t = (text or "").strip()
    if t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _parse_json_values(text: str) -> list:
    """Parse `text` into a list of top-level JSON values. Returns [] if it's not
    JSON at all. Handles both a single value AND the common small-model failure of
    emitting several concatenated JSON objects (one per item) with no array."""
    t = _strip_fences(str(text or ""))
    if not t:
        return []
    try:
        return [json.loads(t)]
    except Exception:  # noqa: BLE001
        pass
    dec, vals, i, n = json.JSONDecoder(), [], 0, len(t)
    while i < n:
        while i < n and t[i] in " \t\r\n,":
            i += 1
        if i >= n:
            break
        try:
            val, i = dec.raw_decode(t, i)
        except ValueError:
            break
        vals.append(val)
    return vals


def _is_tool_call_shape(v) -> bool:
    """True if a parsed JSON value DESCRIBES a tool/function call rather than being
    the requested data. Small local models, given a big tool catalog, often reply
    with e.g. {"name": "some_tool", "parameters": {...}} instead of calling it."""
    if not isinstance(v, dict):
        return False
    keys = {str(k).lower() for k in v}
    return bool(
        {"name", "parameters"} <= keys
        or {"name", "arguments"} <= keys
        or ({"function", "arguments"} & keys and "arguments" in keys)
        or {"tool_call", "function_call"} & keys
        or {"queries", "schema"} <= keys
    )


def _data_answer(text: str) -> str | None:
    """Pretty-printed JSON if `text` is a genuine structured DATA answer; None if
    it's empty, not JSON, a bare scalar (e.g. "search"), or a description of a tool
    call. Concatenated objects are folded into a single JSON array."""
    vals = _parse_json_values(text)
    if not vals or any(_is_tool_call_shape(v) for v in vals):
        return None
    payload = vals[0] if len(vals) == 1 else vals
    if not isinstance(payload, (dict, list)):  # bare "search" / 42 / true isn't an answer
        return None
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _deep_json(v):
    """Recursively expand string values that are THEMSELVES JSON (optionally
    ```-fenced) into real structure — lossless; scalar strings are left as-is. Makes
    a scrape envelope whose content is a JSON string render as readable nested JSON."""
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


def _as_block(content) -> str:
    """Render raw tool output: pretty JSON (embedded JSON expanded) in a fenced
    block whose fence is long enough to survive any backtick run inside."""
    s = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    try:
        pretty = json.dumps(_deep_json(json.loads(_strip_fences(s))), indent=2, ensure_ascii=False)
        lang = "json"
    except Exception:  # noqa: BLE001
        pretty, lang = s.strip(), ""
    longest = cur = 0
    for ch in pretty:
        cur = cur + 1 if ch == "`" else 0
        longest = max(longest, cur)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{pretty}\n{fence}"


def _named_spec_in_goal(goal: str) -> dict | None:
    """The tool spec whose exact name appears as a token in the goal (longest wins),
    or None. Token match avoids the `add` tool matching the word 'address'."""
    cleaned = "".join(c if (c.isalnum() or c == "_") else " " for c in (goal or "").lower())
    tokens = set(cleaned.split())
    best = None
    for s in _specs:
        nm = str(s.get("name", ""))
        if nm and nm.lower() in tokens and (best is None or len(nm) > len(best["name"])):
            best = s
    return best


def _wants_raw(goal: str) -> bool:
    g = (goal or "").lower()
    return any(h in g for h in _RAW_HINTS)


def _direct_target(goal: str) -> dict | None:
    """The tool spec to invoke directly if the goal is an explicit 'call <tool> …'
    (optionally 'raw') request; else None (use the normal extractor agent)."""
    spec = _named_spec_in_goal(goal)
    if spec is None:
        return None
    g = (goal or "").lower()
    return spec if (_wants_raw(goal) or any(v in g for v in _CALL_VERBS)) else None


def _extract_args(goal: str, spec: dict) -> dict | None:
    """Pull the named tool's call arguments out of the plain-language goal. Returns a
    dict filtered to the tool's real parameters, or None if extraction failed."""
    names = [p.get("name") for p in (spec.get("parameters") or []) if p.get("name")]
    prompt = (f"Extract the arguments for calling `{spec['name']}` from the "
              f"INSTRUCTION. Parameters: {names or 'unknown'}.\nINSTRUCTION: {goal}")
    vals = _parse_json_values(_run_output(ARG_AGENT.run_sync(prompt)))
    if not vals or not isinstance(vals[0], dict):
        return None
    return {k: v for k, v in vals[0].items() if not names or k in names}


def _invoke_directly(goal: str, spec: dict) -> str | None:
    """Run the named tool directly through the governed MCP path. Raw output, or
    None if we couldn't build args / the tool errored (caller then falls back)."""
    args = _extract_args(goal, spec)
    if args is None:
        return None
    try:
        return kit.call_tool(spec["name"], args,
                             risk=str(spec.get("risk", "low")), scope=spec["name"])
    except Exception:  # noqa: BLE001
        return None


def _tools_from_messages(messages) -> list[str]:
    used: list[str] = []
    for m in messages or []:
        for part in getattr(m, "parts", []) or []:
            if getattr(part, "part_kind", "") == "tool-call":
                n = getattr(part, "tool_name", None)
                if n and n not in used:
                    used.append(n)
    return used


def _run_output(res) -> str:
    out = getattr(res, "output", None)
    if out is None:
        out = getattr(res, "data", None)
    return "" if out is None else str(out)


def run_goal(goal: str) -> dict:
    # Explicit "call <tool> … (raw)" request → invoke the named tool directly
    # (governed) and return its raw output, rather than letting the model narrate or
    # FABRICATE a result. Handles the "return the raw result of X" case for any tool.
    spec = _direct_target(goal)
    if spec is not None:
        raw = _invoke_directly(goal, spec)
        if raw is not None:
            return {"result": _as_block(raw), "tools_used": [spec["name"]]}
        # extraction/invocation failed — fall through to the normal extractor agent

    # Primary pass: the governed, tool-bound agent. If it does real work its tool
    # runs stay on the timeline; if it answers directly we get clean JSON here.
    res = AGENT.run_sync(goal)
    tools_used = _tools_from_messages(res.all_messages())
    answer = _data_answer(_run_output(res))  # strict: reject tool-call narration

    # Fallback: the tool-bound pass returned no data — the small local model either
    # NARRATED a tool call (code/curl) or emitted tool-call-shaped JSON instead of
    # the answer. Re-ask with NO tools so it just returns the JSON it already knows.
    # A couple of retries absorb the odd degenerate reply from the 8B model.
    out2 = ""
    for _ in range(3):
        if answer is not None:
            break
        out2 = _run_output(DIRECT_AGENT.run_sync(goal))
        answer = _data_answer(out2)
    if answer is None:  # give back whatever we have rather than nothing
        answer = _strip_fences(out2) or "{}"

    return {"result": answer, "tools_used": tools_used}


app = FastAPI(title="PydanticAI Worker Runtime")

if __name__ == "__main__":
    kit.mount(
        app,
        meta={"agent": "PydanticAI Extractor", "framework": "pydantic_ai",
              "model": MODEL, "tools": TOOL_NAMES, "dir": HERE,
              "purpose": "Structured-data extractor — returns clean, validated JSON for any query.",
              "format": "json", "accent": "#2a7de1", "logo": "\U0001F537",
              "examples": ["Extract the key facts about the Eiffel Tower as JSON.",
                           "Give me {name, capital, population, currency} for France.",
                           "Summarise the company Anthropic into structured fields."]},
        run_goal=run_goal,
        port=PORT,
    )
    print(f"🔷 PydanticAI WORKER  →  http://localhost:{PORT}   (model={MODEL})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
