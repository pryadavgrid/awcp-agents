"""An autonomous, governed AUTO-ROUTER WORKER runtime (AWCP agent-on-a-runtime).

This is the "Auto" mode agent. It does NOT answer queries itself and it uses NO
tools. Instead it:

  1. discovers the OTHER worker agents live from the control plane
     (GET /user/agents on the gateway) — nothing about the agent set is hardcoded;
  2. runs an internal judging prompt over the query + each agent's self-declared
     purpose/examples to pick the single best specialist for the request;
  3. delegates the query to that agent via a governed AGENT CALL through the
     control plane (POST /user/ask on the gateway) — the chosen agent then runs
     its whole governed MCP-tool pipeline (radar gate -> Temporal -> execute_tool).

So the only outward action this agent takes is an AGENT CALL routed through the
control plane; it never calls an MCP tool of its own. The user-facing result
shows WHICH agent was chosen plus that agent's formatted output — nothing about
the internal judging.

Run as:  python auto_agent.py   (absolute path via run.sh so the detector sees
the `langgraph` import).
"""

import ast
import json
import os
import time
import urllib.parse
import urllib.request

from langgraph.graph import StateGraph  # noqa: F401  (marks this as LangGraph)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from fastapi import FastAPI
import uvicorn

import awcp_kit as kit

MODEL = os.getenv("AUTO_MODEL", "llama3.1:8b")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
PORT = int(os.getenv("AUTO_PORT", "8106"))
HERE = os.path.dirname(os.path.abspath(__file__))
SELF_ID = os.path.basename(HERE)  # the folder name = this agent's id in /user/agents

# How long to wait on the delegated agent's blocking run. The gateway's own
# /user/ask cap is AWCP_ASK_TIMEOUT (default 300s); we wait a little longer so the
# gateway's own timeout wins, not ours. Env-driven — nothing about timing is baked.
DELEGATE_TIMEOUT = float(os.getenv("AUTO_DELEGATE_TIMEOUT", "330"))

JUDGE_SYSTEM = (
    "You are a ROUTER. You are given a user's REQUEST and a numbered CATALOG of "
    "specialist agents. Each catalog entry is that agent's self-declared AGENT "
    "CARD: its purpose, its skills (the tools it can run), and example tasks it "
    "handles well. Compare the REQUEST against every card and choose the SINGLE "
    "agent whose purpose and skills best match — do NOT default to the first "
    "entry. Answer with ONLY a JSON object of the form "
    '{"agent_id": "<exact id from the catalog>", "reason": "<one short '
    'sentence>"}. Output nothing else — no prose, no markdown, no code fences.'
)


# --- no tools: this agent only makes governed AGENT CALLS via the control plane -
# The router binds NO MCP tools. Its one outward action is delegating to another
# agent through the gateway, which keeps that agent's own governed tool pipeline.
TOOLS: list = []
TOOL_NAMES: list = []

# format="json" makes Ollama constrain decoding to valid JSON, so the judge's
# answer always parses — no prose drift into the first-candidate fallback.
_llm = ChatOllama(model=MODEL, base_url=OLLAMA_BASE, temperature=0, format="json")


def _get_json(url: str, timeout: float = 4.0) -> dict:
    """GET an absolute URL and parse JSON. Fail-open: {} on any error."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return {}


def _agent_card(a: dict) -> dict:
    """Read one agent's self-declared AGENT CARD — the A2A card it serves at
    {url}/.well-known/agent.json (identity, description, skills) merged with the
    `purpose` from its own /info (the card's description falls back to a generic
    string; /info carries the human-written purpose). Direct backend reads (no
    browser/CORS limits). Returns {} for a stopped agent (no url); routing then
    falls back to the name + examples the gateway list already carries."""
    url = a.get("url")
    if not url:
        return {}
    card = _get_json(f"{url}/.well-known/agent.json")
    info = _get_json(f"{url}/info")
    skills = [s.get("id") or s.get("name") for s in (card.get("skills") or [])
              if isinstance(s, dict)]
    return {
        "purpose": str(info.get("purpose") or card.get("description") or ""),
        "skills": [s for s in skills if s],
    }


def _card_from_source(agent_id: str) -> dict:
    """Static agent-card fallback for a STOPPED candidate: parse its self-declared
    purpose/examples out of the sibling agent folder's source (the same meta dict
    it would serve live on /info). Without this a stopped agent shows up in the
    judge catalog as "(no description)" and the judge drifts toward whichever
    agent happens to be running. AST-based (no exec); {} on any failure."""
    path = os.path.join(os.path.dirname(HERE), agent_id, f"{agent_id}.py")
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:  # noqa: BLE001
        return {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        card: dict = {}
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value in ("purpose", "examples"):
                try:
                    card[k.value] = ast.literal_eval(v)
                except Exception:  # noqa: BLE001 — non-literal value, skip
                    pass
        if card.get("purpose"):
            return {"purpose": str(card["purpose"]),
                    "examples": list(card.get("examples") or [])}
    return {}


def _candidates() -> list[dict]:
    """The specialist agents the router may route to, discovered live from the
    control plane. Excludes this router itself and any other router-role agent
    (framework 'router'), so nothing about the routable set is hardcoded — a new
    agent folder is eligible automatically. Each entry is enriched with the
    agent's self-declared AGENT CARD (purpose + skills, read from its own
    /.well-known/agent.json and /info) for better judging.
    Fail-open: returns [] if the gateway is unreachable."""
    agents = kit._radar_get("/user/agents", timeout=8.0)
    if not isinstance(agents, list):
        return []
    out: list[dict] = []
    for a in agents:
        if not isinstance(a, dict):
            continue
        if a.get("id") == SELF_ID or a.get("framework") == "router":
            continue
        a = dict(a)
        card = _agent_card(a)
        if not card.get("purpose"):
            # Stopped (no live /info) — fall back to the static card in its source
            # so a not-yet-started specialist competes on equal footing.
            src = _card_from_source(str(a.get("id") or ""))
            card = {**src, **{k: v for k, v in card.items() if v}}
        a.update({k: v for k, v in card.items() if v})
        out.append(a)
    return out


def _catalog(candidates: list[dict]) -> str:
    """Render the candidate agent cards into a compact numbered catalog for the
    judge: id, name, purpose, skills (tools), and example tasks."""
    lines = []
    for i, a in enumerate(candidates, 1):
        name = a.get("name") or a.get("id")
        purpose = a.get("purpose") or "(no description)"
        skills = a.get("skills") or a.get("tools") or []
        examples = a.get("examples") or []
        ex = "; ".join(examples[:3])
        lines.append(f"{i}. id={a.get('id')} | name={name} | purpose: {purpose}"
                     + (f" | skills: {', '.join(skills[:8])}" if skills else "")
                     + (f" | examples: {ex}" if ex else ""))
    return "\n".join(lines)


_STOPWORDS = frozenset(
    "a an the and or of to in on for with about is are what how why me my your "
    "please can you it this that from as at by".split())


def _fallback_choice(query: str, candidates: list[dict]) -> tuple[dict, str]:
    """Deterministic no-LLM fallback: score each candidate by keyword overlap
    between the query and its agent card (purpose + skills + examples + name).
    Replaces the old blind `candidates[0]` fallback, which always landed on the
    alphabetically-first agent whenever the judge model was down or unparseable."""
    qwords = {w.strip(".,;:!?()[]'\"") for w in query.lower().split()}
    qwords = {w for w in qwords if w not in _STOPWORDS and len(w) > 2}
    best, best_score = candidates[0], -1.0
    for a in candidates:
        text = " ".join([
            str(a.get("name") or ""), str(a.get("purpose") or ""),
            " ".join(a.get("skills") or []), " ".join(a.get("examples") or []),
        ]).lower()
        score = sum(1.0 for w in qwords if w in text)
        if score > best_score:
            best, best_score = a, score
    return best, "closest match by agent-card keywords"


def _parse_choice(query: str, text: str, candidates: list[dict]) -> tuple[dict, str]:
    """Resolve the judge's answer to one candidate. Robust to extra prose: pull the
    first JSON object, match its agent_id against the catalog (exact, then
    substring); fall back to the keyword scorer so routing never dead-ends AND
    never silently lands on the alphabetically-first agent."""
    ids = {a.get("id"): a for a in candidates}
    chosen_id, reason = "", ""
    try:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start:end + 1])
            chosen_id = str(obj.get("agent_id", "")).strip()
            reason = str(obj.get("reason", "")).strip()
    except Exception:  # noqa: BLE001
        chosen_id, reason = "", ""

    if chosen_id in ids:
        return ids[chosen_id], reason
    # substring match (model may echo the name or a partial id)
    low = (chosen_id or text or "").lower()
    for aid, a in ids.items():
        if aid and aid.lower() in low:
            return a, reason
    for a in candidates:  # match on name as a last structured attempt
        nm = (a.get("name") or "").lower()
        if nm and nm in low:
            return a, reason
    return _fallback_choice(query, candidates)


def _choose(query: str, candidates: list[dict]) -> tuple[dict, str]:
    """Internal judging step: pick the best agent for the query by comparing it
    against each candidate's agent card (purpose + skills + examples)."""
    prompt = (f"REQUEST:\n{query}\n\nCATALOG:\n{_catalog(candidates)}\n\n"
              "Reply with the JSON object only.")
    try:
        resp = _llm.invoke([SystemMessage(content=JUDGE_SYSTEM),
                            HumanMessage(content=prompt)])
        return _parse_choice(query, str(resp.content), candidates)
    except Exception:  # noqa: BLE001 — model down -> deterministic card-keyword match
        return _fallback_choice(query, candidates)


# Sub-run states that mean "settled" — mirror the gateway's own TERMINAL set.
_SETTLED = {"done", "failed", "blocked", "canceled", "awaiting_approval"}


def _delegate(agent_id: str, query: str) -> dict:
    """Run the chosen agent through the control plane and return its settled result.

    A governed AGENT CALL that stays ONE task: the submit carries THIS router
    task's id as parent_task_id, so the delegated run JOINS the router's own
    Temporal workflow and context-graph run — its every governed step (LLM
    calls, tool calls, checkpoints) lands on this task's timeline directly.
    The router's live progress % climbs with that real work.

    Older specialists that don't understand parent_task_id still start their own
    workflow; for those the watch loop below MIRRORS each new governed step onto
    this task's timeline as before (the sub record has no `joined` flag).

    The watch polls the DELEGATED AGENT DIRECTLY (its own lightweight in-memory
    /tasks record), NOT the gateway's heavy /user/status — so watching adds no
    Temporal/subprocess load to the control plane. The gateway is asked exactly once
    at the end for the AUTHORITATIVE, block-adjusted result.

    Empty session: the delegated run is metered as its own turn and never pollutes
    this chat's history — the router's own turn is the one recorded for this chat."""
    my_task = kit._CURRENT.get("task") or {}
    my_tid = my_task.get("id") or ""
    sub = kit._radar_call(
        "/user/submit",
        {"agent": agent_id, "input": query, "auto_start": True, "session": "",
         "parent_task_id": my_tid},
        timeout=max(120.0, DELEGATE_TIMEOUT),
    )
    sub_task = sub.get("task_id")
    agent_url = sub.get("agent_url")
    workflow_id = sub.get("workflow_id") or ""
    if not sub_task or not agent_url:
        # Submit failed (e.g. agent couldn't start) — fall back to a blocking ask so
        # the user still gets a result or the real error.
        return kit._radar_call(
            "/user/ask",
            {"agent": agent_id, "input": query, "auto_start": True, "session": "",
             "parent_task_id": my_tid},
            timeout=DELEGATE_TIMEOUT,
        )

    task_url = f"{agent_url}/tasks/{urllib.parse.quote(sub_task)}"

    # Mark the delegation itself as a SUBCALL on this router's Temporal workflow:
    # one tool_called event named agent_call:<id> — the radar's existing
    # tool_called → execution_tool_call mapping turns it into a named activity
    # (tool=agent_call:<id>) with zero Temporal-side changes.
    if my_tid:
        kit._emit_execution_event(my_tid, "tool_called",
                                  tool_name=f"agent_call:{agent_id}",
                                  risk="low", subcall=True,
                                  sub_task_id=sub_task,
                                  sub_workflow_id=workflow_id)

    mirrored = 0            # governed steps of the sub-run already reflected
    rec: dict = {}
    deadline = time.monotonic() + DELEGATE_TIMEOUT
    while time.monotonic() < deadline:
        rec = _get_json(task_url, timeout=5.0) or rec
        status = str(rec.get("status", "")).lower()

        # A JOINED sub-run (parent_task_id honoured) reports its steps into THIS
        # task's workflow itself — mirroring would duplicate every activity, so
        # skip it. For an older specialist that started its own workflow, mirror
        # each NEW governed step as a tool_called, namespaced "<agent_id>::<tool>"
        # so the Temporal timeline reads unambiguously as work done INSIDE the
        # subcall. A tool-free (pure-LLM) run records no steps and simply stays on
        # "Thinking" until it settles — like a plain LLM agent already does here.
        steps = rec.get("steps") or []
        if not rec.get("joined"):
            for step in steps[mirrored:]:
                if my_tid:
                    kit._emit_execution_event(my_tid, "tool_called",
                                              tool_name=f"{agent_id}::{step.get('action')}",
                                              risk=step.get("risk", "low"),
                                              subcall=True)
        mirrored = max(mirrored, len(steps))

        if status in _SETTLED:
            break
        if my_task.get("cancel"):      # user stopped the router run — stop waiting
            break
        time.sleep(2.0)

    # One authoritative read from the control plane: it suppresses answers the
    # policy layer blocked (the agent's own record may still hold them) and carries
    # the accurate block reason. Fall back to the agent's record if it's empty.
    # A JOINED sub-run never had its own workflow — its authoritative outcome
    # (including any policy block) lives on THIS router task's shared workflow.
    if rec.get("joined") and kit._CURRENT_EXEC_WF.get(my_tid):
        workflow_id = kit._CURRENT_EXEC_WF[my_tid]
    status_path = (f"/user/status/{urllib.parse.quote(agent_id)}/"
                   f"{urllib.parse.quote(sub_task)}?workflow_id="
                   f"{urllib.parse.quote(workflow_id)}")
    authoritative = kit._radar_get(status_path, timeout=10.0)
    settled = dict(authoritative) if authoritative else dict(rec)
    # Carry the join marker: the caller must not re-report tools whose real
    # events already landed on this router task's own workflow.
    settled["joined"] = bool(rec.get("joined"))
    return settled


def run_goal(goal: str) -> dict:
    candidates = _candidates()
    if not candidates:
        return {"result": "Auto mode found no specialist agents to route to right "
                          "now. Try again once the agent bundle is up, or pick an "
                          "agent manually.", "tools_used": []}

    chosen, reason = _choose(goal, candidates)
    chosen_id = chosen.get("id")
    chosen_name = chosen.get("name") or chosen_id

    sub = _delegate(chosen_id, goal)

    header = f"**Auto-routed to {chosen_name}**" + (f" — {reason}" if reason else "")

    # Surface a control-plane block from the delegated run instead of an answer.
    if sub.get("blocked") or str(sub.get("status", "")).lower() == "blocked":
        title = sub.get("blocked_title") or "⛔ Blocked by the control plane"
        why = sub.get("blocked_reason") or sub.get("error") or ""
        body = f"{title}" + (f"\n\n{why}" if why else "")
        return {"result": f"{header}\n\n{body}", "tools_used": [],
                "delegated_to": chosen_id}

    answer = str(sub.get("result", "")).strip()
    if not answer:
        answer = ("The chosen agent returned no output"
                  + (f" ({sub.get('error')})" if sub.get("error") else "") + ".")

    # Only the chosen agent + its formatted output reach the UI — never the judging.
    return {
        "result": f"{header}\n\n{answer}",
        # Carry the delegated agent's real tools so the run timeline reflects the
        # work that actually happened downstream.
        "tools_used": sub.get("tools_used", []) or [],
        # A JOINED run's tool events are already on this task's workflow — tell
        # the kit not to replay them (they would duplicate on the timeline).
        "tools_reported": bool(sub.get("joined")),
        "delegated_to": chosen_id,
    }


app = FastAPI(title="Auto Router Worker Runtime")

if __name__ == "__main__":
    kit.mount(
        app,
        meta={"agent": "Auto (Smart Router)", "framework": "router",
              "model": MODEL, "tools": TOOL_NAMES, "dir": HERE,
              "purpose": "Auto mode — reads your query, judges which specialist agent "
                         "fits best, and runs that agent for you. Uses more credits "
                         "than picking an agent yourself.",
              "format": "markdown", "accent": "#7c5cff", "logo": "\U0001F9ED",
              # `framework: "router"` is the single signal that flows through the
              # gateway's existing /user/agents payload, so BOTH the UI (to badge
              # this agent + show the credit warning) and this router (to exclude
              # itself and any other router) detect it without any gateway change.
              # `role`/`cost_warning` are honest self-description surfaced on this
              # agent's own /info + about page.
              "role": "router",
              "cost_warning": "Auto mode runs an extra routing step and then the "
                              "chosen agent, so it uses more credits than selecting "
                              "an agent yourself.",
              "examples": ["Find recent papers on retrieval-augmented generation.",
                           "Write a short brief on the benefits of solar energy.",
                           "Extract the key facts about the Eiffel Tower as JSON.",
                           "What is in this file?"]},
        run_goal=run_goal,
        port=PORT,
    )
    print(f"\U0001F9ED  Auto Router WORKER  →  http://localhost:{PORT}   (model={MODEL})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
