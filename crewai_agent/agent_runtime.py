"""An autonomous, governed CrewAI WORKER runtime (AWCP agent-on-a-runtime).

Pulls GOALS off a task queue and executes each as a CrewAI task. CrewAI tool
calling is version/model dependent, so the deterministic finalize in awcp_kit
guarantees the runtime's output is still routed through the control plane
(save_artifact = governed local write; external_post = high-risk gated write).

Run as:  python agent_runtime.py   (absolute path via run.sh so the detector sees
the `crewai` import).
"""

import datetime
import os

os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")

from crewai import Agent, Task, Crew, LLM  # noqa: E402  (marks this as CrewAI)

from fastapi import FastAPI  # noqa: E402
import uvicorn  # noqa: E402

import awcp_kit as kit  # noqa: E402

MODEL = os.getenv("CREW_MODEL", "ollama/llama3.1:8b")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
PORT = int(os.getenv("CREW_PORT", "8101"))
HERE = os.path.dirname(os.path.abspath(__file__))

_llm = LLM(model=MODEL, base_url=OLLAMA_BASE, temperature=0)

# Governed tools as CrewAI tools (added when this CrewAI version exposes the
# decorator; the deterministic finalize covers governance either way).
TOOLS = []
TOOL_NAMES = ["web_search", "save_artifact", "external_post"]
try:
    from crewai.tools import tool as crew_tool  # noqa: E402

    @crew_tool("web_search")
    def web_search(query: str) -> str:
        """Search the web for current/real-world information (no API key)."""
        return kit.web_search(query)

    @crew_tool("save_artifact")
    def save_artifact(content: str) -> str:
        """Save a result artifact to disk. GOVERNED local write (gated)."""
        return kit.save_artifact("result", content)

    @crew_tool("external_post")
    def external_post(summary: str) -> str:
        """Submit/publish a result externally. HIGH-RISK governed write (gated +
        operator approval)."""
        return kit.external_post(summary)

    TOOLS = [web_search, save_artifact, external_post]
except Exception:  # noqa: BLE001
    TOOLS, TOOL_NAMES = [], []

WORKER = Agent(
    role="Research Writer",
    goal="Research the topic and write a clear, well-structured report about it.",
    backstory="A skilled writer who researches a topic and turns it into a concise, "
              "readable report — using short markdown headings and tight paragraphs.",
    llm=_llm, tools=TOOLS, verbose=False, allow_delegation=False,
)


def run_goal(goal: str) -> dict:
    task = Task(description=goal,
                expected_output="A clear, well-structured report in markdown — short "
                                "headings, tight paragraphs, and a brief summary.",
                agent=WORKER)
    crew = Crew(agents=[WORKER], tasks=[task], verbose=False)
    return {"result": str(crew.kickoff()), "tools_used": TOOL_NAMES}


app = FastAPI(title="CrewAI Worker Runtime")

if __name__ == "__main__":
    kit.mount(
        app,
        meta={"agent": "CrewAI Writer", "framework": "crewai",
              "model": MODEL, "tools": TOOL_NAMES, "dir": HERE,
              "purpose": "Content & report writer — researches a topic, then drafts a structured write-up.",
              "format": "markdown", "accent": "#0f9d8f", "logo": "\U0001F465",
              "examples": ["Write a short brief on the benefits of solar energy.",
                           "Draft a 3-paragraph report on the history of the internet.",
                           "Write a concise explainer on what an LLM is."]},
        run_goal=run_goal,
        port=PORT,
    )
    print(f"👥 CrewAI WORKER  →  http://localhost:{PORT}   (model={MODEL})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
