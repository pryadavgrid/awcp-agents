# Writeup Agent — governed report-writer (PydanticAI)

An autonomous, governed **report-writer** worker: give it a topic and it
researches (via the MCP server's `web_search`), writes a markdown report,
persists it with the governed `save_artifact` write, and — when the goal asks
to email the report to someone — drafts the email with the governed
`draft_email` write.

## Email drafts land in `output/`

Every email this agent drafts is saved into this folder:

```
writeup_agent/output/<timestamp>-<subject-slug>.md
```

Each draft carries a `to / subject / date / status: draft (not sent)`
front-matter header followed by the body. **Nothing is ever sent** — a human
reviews the draft and mails it themselves. The destination is the MCP server's
`draft_email` tool dir (`AWCP_EMAIL_DRAFT_DIR`, default
`<agents bundle>/writeup_agent/output`).

## Governance / approvals

- The agent registers with the AWCP radar as **`writeup-writer`** — the same
  key it has in the control plane's `awcp_magazine.json` (risk `high`). Keep
  the two names in lock-step.
- Tools are discovered from the MCP server catalog and run **only** on the MCP
  server, behind the radar's write-action gate.
- Write-tier tools (`save_artifact` medium, `draft_email`/`external_post`
  high) pause the task and surface an approval request **centrally in the AWCP
  control-plane UI's Approvals panel** (`POST /approvals` on the radar) as well
  as in this agent's own task console. The operator's approve/deny releases
  the pause.

## Control-plane dependencies (outside this folder)

Everything agent-side lives in this folder. Three things live in the control
plane because the architecture gives them no other home:

1. **`draft_email` tool** — `src/awcp/execution/tools/draft_email.py` in the
   control-plane repo (branch `feature/governed-email-drafts`). Agents in this
   bundle execute tools *only* on the MCP governance server (the kit has no
   local-execution path), so the drafting capability must be a server-side
   tool — that is exactly what makes it gated, traced, and approvable.
2. **Magazine entry** — `writeup-writer` in the radar's `awcp_magazine.json`
   (same branch). Optional in effect (`__default__` is also `high`), but makes
   this agent's high-risk rating explicit.
3. **Operator policy row** — `tools.draft_email: {risk: high}` saved via
   `PUT /policy` (governance DB, not a file in any repo). Needed because the
   OPA agent's SLM auto-rated `draft_email` as `low`, which would skip the
   approval pause. If the policy store is ever reset, re-add this rule from
   the control-plane UI's Policy tab.

## Run

```bash
./run.sh            # starts on :8104 (WRITEUP_PORT), model qwen2.5:7b (WRITEUP_MODEL)
```

Or start it from the AWCP User UI / gateway (`POST /user/agents/writeup_agent/start`)
or the local control panel (`python3 control_panel.py` → http://localhost:8099) —
the folder is discovered automatically, so it appears in both UIs with no
registration step.
