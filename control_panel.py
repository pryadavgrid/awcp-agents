#!/usr/bin/env python3
"""AWCP Agent Control Panel — one UI to start/stop every runtime agent.

Discovery is fully DYNAMIC and not hardcoded: every sub-folder of this `agents/`
directory that contains a `run.sh` is treated as an agent (id = folder name).
Add or remove an agent folder and the panel updates automatically — nothing about
the specific agents, their ports, or models is baked in. Running state is found by
matching the agent's own `agent_runtime.py` path among live processes, and the
listening port (for the "Open UI" link) is discovered from the live process, not
assumed. (The HTML page itself is a static template — that part is intentionally
hardcoded, as requested.)

Stdlib only. Run with:  python3 control_panel.py   (defaults to http://localhost:8099)
Override the port with PANEL_PORT.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.getenv("PANEL_PORT", "8099"))


# ----------------------------------------------------------------------
# Dynamic discovery + process control (no agent-specific knowledge)
# ----------------------------------------------------------------------
def discover() -> list[dict]:
    """Every sub-folder with a run.sh is an agent. id = folder name."""
    agents = []
    for name in sorted(os.listdir(AGENTS_DIR)):
        d = os.path.join(AGENTS_DIR, name)
        run = os.path.join(d, "run.sh")
        if os.path.isdir(d) and os.path.isfile(run):
            agents.append(
                {
                    "id": name,
                    "dir": d,
                    "run": run,
                    "runtime": os.path.join(d, "agent_runtime.py"),
                }
            )
    return agents


def _find(agent_id: str) -> dict | None:
    return next((a for a in discover() if a["id"] == agent_id), None)


def _pids(agent: dict) -> list[int]:
    """PIDs whose command line references this agent's own agent_runtime.py."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", agent["runtime"]], capture_output=True, text=True
        )
        return [int(p) for p in out.stdout.split() if p.strip()]
    except Exception:
        return []


def _listening_port(pid: int) -> int | None:
    """Discover the TCP port a running agent is actually listening on."""
    try:
        out = subprocess.run(
            # -a ANDs the selectors so we get ONLY this pid's LISTEN sockets
            # (without -a, lsof ORs them and returns every process's ports).
            ["lsof", "-nP", "-a", "-p", str(pid), "-iTCP", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
        )
        m = re.search(r":(\d+)\s*\(LISTEN\)", out.stdout)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def status() -> list[dict]:
    items = []
    for a in discover():
        pids = _pids(a)
        port = _listening_port(pids[0]) if pids else None
        items.append(
            {"id": a["id"], "running": bool(pids), "pids": pids, "port": port}
        )
    return items


def start(agent_id: str) -> tuple[bool, str]:
    a = _find(agent_id)
    if not a:
        return False, "unknown agent"
    if _pids(a):
        return True, "already running"
    subprocess.Popen(
        ["bash", a["run"]],
        cwd=a["dir"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True, "starting"


def stop(agent_id: str) -> tuple[bool, str]:
    a = _find(agent_id)
    if not a:
        return False, "unknown agent"
    subprocess.run(["pkill", "-f", a["runtime"]])
    return True, "stopping"


# ----------------------------------------------------------------------
# HTTP server
# ----------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _serve_index(self) -> None:
        idx = os.path.join(AGENTS_DIR, "index.html")
        if os.path.isfile(idx):
            with open(idx, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        else:
            self._send(200, UI_HTML.encode(), "text/html; charset=utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_index()
        elif path == "/api/agents":
            self._json(status())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/")
        m = re.fullmatch(r"/api/agents/([^/]+)/(start|stop)", path)
        if m:
            ok, msg = (start if m.group(2) == "start" else stop)(m.group(1))
            self._json({"ok": ok, "message": msg}, 200 if ok else 404)
        elif path == "/api/start-all":
            res = {a["id"]: start(a["id"])[1] for a in discover()}
            self._json({"ok": True, "results": res})
        elif path == "/api/stop-all":
            res = {a["id"]: stop(a["id"])[1] for a in discover()}
            self._json({"ok": True, "results": res})
        else:
            self._json({"error": "not found"}, 404)


UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AWCP Agent Control Panel</title>
<style>
:root{--bg:#0b0f17;--panel:#121826;--line:#1f2937;--fg:#e5e7eb;--mut:#9ca3af;--acc:#6366f1;--ok:#22c55e;--off:#6b7280;--red:#ef4444}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:18px 24px;border-bottom:1px solid var(--line);background:var(--panel);display:flex;align-items:center;gap:14px;flex-wrap:wrap}
header h1{font-size:17px;margin:0}
.sub{color:var(--mut);font-size:12px}
.bulk{margin-left:auto;display:flex;gap:8px}
.wrap{max-width:820px;margin:24px auto;padding:0 20px;display:flex;flex-direction:column;gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;display:flex;align-items:center;gap:14px}
.dot{width:11px;height:11px;border-radius:50%;background:var(--off);flex:none}
.dot.on{background:var(--ok);box-shadow:0 0 8px rgba(34,197,94,.6)}
.name{font-weight:600}
.meta{color:var(--mut);font-size:12px}
.spacer{margin-left:auto}
a.open{color:#a5b4fc;font-size:12px;text-decoration:none;border:1px solid var(--line);padding:5px 10px;border-radius:8px}
a.open[hidden]{display:none}
button{border:0;border-radius:9px;padding:8px 16px;font-weight:600;cursor:pointer;color:#fff}
button.start{background:var(--acc)}
button.stop{background:transparent;border:1px solid var(--line);color:var(--mut)}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--fg)}
button:disabled{opacity:.45;cursor:default}
.empty{color:var(--mut);text-align:center;margin-top:40px}
</style>
</head>
<body>
<header>
  <div>
    <h1>AWCP Agent Control Panel</h1>
    <div class="sub" id="sub">discovering agents…</div>
  </div>
  <div class="bulk">
    <button class="ghost" id="startAll">Start all</button>
    <button class="stop"  id="stopAll">Stop all</button>
  </div>
</header>
<div class="wrap" id="list"></div>
<script>
const list = document.getElementById('list');
const sub  = document.getElementById('sub');
async function api(path, method='GET'){ const r = await fetch(path,{method}); return r.json(); }
function card(a){
  const open = a.port ? `<a class="open" href="http://localhost:${a.port}" target="_blank">Open UI :${a.port}</a>` : `<a class="open" hidden></a>`;
  return `<div class="card" data-id="${a.id}">
    <span class="dot ${a.running?'on':''}"></span>
    <div>
      <div class="name">${a.id}</div>
      <div class="meta">${a.running ? ('running · pid '+a.pids.join(', ')) : 'stopped'}</div>
    </div>
    <span class="spacer"></span>
    ${open}
    <button class="start" ${a.running?'disabled':''} onclick="act('${a.id}','start')">Start</button>
    <button class="stop"  ${a.running?'':'disabled'} onclick="act('${a.id}','stop')">Stop</button>
  </div>`;
}
async function refresh(){
  const agents = await api('/api/agents');
  const running = agents.filter(a=>a.running).length;
  sub.textContent = `${agents.length} agent${agents.length!==1?'s':''} discovered · ${running} running`;
  list.innerHTML = agents.length ? agents.map(card).join('') :
    '<div class="empty">No agent folders found. Drop a folder with a run.sh into agents/.</div>';
}
async function act(id, what){
  document.querySelectorAll(`[data-id="${id}"] button`).forEach(b=>b.disabled=true);
  await api(`/api/agents/${id}/${what}`,'POST');
  setTimeout(refresh, what==='start'? 1500 : 600);
}
document.getElementById('startAll').onclick = async ()=>{ await api('/api/start-all','POST'); setTimeout(refresh,1800); };
document.getElementById('stopAll').onclick  = async ()=>{ await api('/api/stop-all','POST');  setTimeout(refresh,800); };
refresh(); setInterval(refresh, 3000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(f"🎛  AWCP Agent Control Panel  →  http://localhost:{PORT}")
    print(f"    discovering agents in: {AGENTS_DIR}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
