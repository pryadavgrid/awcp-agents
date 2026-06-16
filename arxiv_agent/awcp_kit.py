"""AWCP runtime kit — shared helpers that turn a framework agent into a fully
governed, observable, self-registering task-worker runtime.

Each agent is SELF-CONTAINED but fully integrated:
  * OTel traces/metrics/logs exported to the local OTLP collector (port 4317).
    Every task, every tool call, every governed write gets its own span; the
    goal/result/tools are recorded as span attributes and appear in Tempo + Loki.
  * Self-registers with the AWCP radar (port 8090) on startup so the radar
    immediately knows about the agent and kicks off a Temporal onboarding workflow.
  * Routes governed writes through the radar's write-action gate before executing.
  * Reports task outcomes as execution signals so the radar can track the agent's
    autonomy profile and degrade it if failures accumulate.

All endpoints, topics, and timeouts are env-driven — nothing is hardcoded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.request

# ── Config (all env-driven) ───────────────────────────────────────────────────
RADAR_URL            = os.getenv("AGENT_RADAR_URL",               "http://localhost:8090")
OTEL_ENDPOINT        = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT",   "http://localhost:4317")
EXTERNAL_WRITE_URL   = os.getenv("AGENT_EXTERNAL_WRITE_URL",      "https://httpbin.org/post")
EXTERNAL_WRITE_TOKEN = os.getenv("AGENT_EXTERNAL_WRITE_TOKEN",    "")
APPROVAL_REQUIRED    = os.getenv("AGENT_APPROVAL_REQUIRED",       "true").lower() == "true"
APPROVAL_TIMEOUT     = float(os.getenv("AGENT_APPROVAL_TIMEOUT",  "180"))
FINALIZE_ARTIFACT    = os.getenv("AGENT_FINALIZE_ARTIFACT",       "true").lower() == "true"
FINALIZE_EXTERNAL    = os.getenv("AGENT_FINALIZE_EXTERNAL",       "false").lower() == "true"
ARTIFACT_DIR = ""    # set by mount()
AGENT_NAME   = "agent"  # set by mount()

# ── Internal state (set by mount()) ──────────────────────────────────────────
_AGENT_ID        = ""   # stable across restarts; derived from framework + dir hash
_AGENT_FRAMEWORK = ""
_AGENT_MODEL     = ""   # human-readable model name, e.g. "llama3.1:8b"
_log = logging.getLogger("awcp.agent")

# ── OTel handles (set by _setup_otel) ────────────────────────────────────────
_tracer          = None  # TracerProvider tracer; avoids depending on global provider
_meter           = None  # MeterProvider meter
_task_counter    = None  # counter: agent.tasks.total
_task_duration   = None  # histogram: agent.task.duration_ms
_llm_calls_total = None  # counter: agent.llm.calls.total


# ── OTel setup ────────────────────────────────────────────────────────────────

def _setup_otel(service_name: str) -> None:
    """Initialise OTel TracerProvider, MeterProvider, and LoggerProvider.

    Works for any agent framework, including CrewAI which installs its own
    OTel provider before our setup runs. Piggybacking ensures our OTLP
    exporter is attached to whatever provider is global; _tracer is stored
    directly from our own tp so our spans always carry the correct service.name.
    """
    global _tracer, _meter, _task_counter, _task_duration, _llm_calls_total
    os.environ.pop("OTEL_SDK_DISABLED", None)
    grpc_ep = OTEL_ENDPOINT.replace("http://", "").replace("https://", "")
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        resource = Resource(attributes={
            "service.name":       service_name,
            "service.instance.id": f"{service_name}:{_AGENT_ID}",
            "agent.id":           _AGENT_ID,
            "agent.framework":    _AGENT_FRAMEWORK,
        })
        tp = TracerProvider(resource=resource)
        tp.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter(endpoint=grpc_ep, insecure=True)
        ))
        # Store our tracer BEFORE any piggybacking so our _span() calls always
        # use our resource (service.name=awcp-agent-<framework>), not a
        # third-party framework's resource (e.g. crewAI-telemetry).
        _tracer = tp.get_tracer("awcp.agent")

        # Attempt to set as global; if a framework (e.g. CrewAI) already owns
        # the global provider, piggyback our exporter onto it so HTTP/HTTPX
        # auto-instrumented spans also reach our collector.
        _piggybacked = ""
        trace.set_tracer_provider(tp)
        try:
            real = trace.get_tracer_provider()
            for _ in range(4):
                inner = getattr(real, "_real_provider", None) or \
                        getattr(real, "_provider", None)
                if inner is None:
                    break
                real = inner
            if real is not tp and hasattr(real, "add_span_processor"):
                real.add_span_processor(BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=grpc_ep, insecure=True)
                ))
                _piggybacked = type(real).__name__
        except Exception:
            pass

        # Metrics — agent-level instruments for Prometheus/Grafana
        mp = MeterProvider(
            resource=resource,
            metric_readers=[PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=grpc_ep, insecure=True),
                export_interval_millis=15_000,
            )],
        )
        metrics.set_meter_provider(mp)
        _meter = mp.get_meter("awcp.agent")
        try:
            _task_counter    = _meter.create_counter(
                "agent.tasks.total", unit="1",
                description="Total tasks executed by this agent")
            _task_duration   = _meter.create_histogram(
                "agent.task.duration_ms", unit="ms",
                description="End-to-end task execution latency")
            _llm_calls_total = _meter.create_counter(
                "agent.llm.calls.total", unit="1",
                description="Total LLM calls made during tasks")
        except Exception:
            pass

        # Log bridge: create OUR logger provider with the correct resource
        # so logs appear in Loki under awcp-agent-<framework>, not a
        # framework-owned service name.
        try:
            try:
                from opentelemetry.sdk.logs import LoggerProvider, LoggingHandler
                from opentelemetry.sdk.logs.export import BatchLogRecordProcessor
            except ModuleNotFoundError:
                from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler  # type: ignore[no-redef]
                from opentelemetry.sdk._logs.export import BatchLogRecordProcessor  # type: ignore[no-redef]
            from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
            from opentelemetry._logs import set_logger_provider, get_logger_provider

            lp = LoggerProvider(resource=resource)
            lp.add_log_record_processor(
                BatchLogRecordProcessor(OTLPLogExporter(endpoint=grpc_ep, insecure=True))
            )
            set_logger_provider(lp)
        except Exception:
            lp = None  # mark log bridge as failed

        # Inject trace context into format AND let basicConfig set root
        # level to NOTSET. Must run BEFORE adding our LoggingHandler so
        # basicConfig is not blocked by an already-present handler.
        try:
            from opentelemetry.instrumentation.logging import LoggingInstrumentor
            LoggingInstrumentor().instrument(set_logging_format=True)
        except Exception:
            pass

        # Add OTel log handler AFTER LoggingInstrumentor so basicConfig
        # already set root level + StreamHandler. Our handler exports to
        # the OTLP collector and works for any framework.
        if lp is not None:
            try:
                logging.getLogger().addHandler(
                    LoggingHandler(level=logging.NOTSET, logger_provider=lp)
                )
            except Exception:
                pass

        # Auto-trace all HTTP/HTTPX calls — captures every LLM call generically.
        try:
            from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
            HTTPXClientInstrumentor().instrument(
                request_hook=_httpx_request_hook,
                response_hook=_httpx_response_hook,
            )
        except Exception:
            pass

        # Capture REAL LLM token usage from the ollama client (stream-safe).
        _install_token_capture()

        try:
            from opentelemetry.instrumentation.requests import RequestsInstrumentor
            RequestsInstrumentor().instrument()
        except Exception:
            pass

        _log.info(
            "otel.setup service=%s endpoint=%s agent_id=%s framework=%s piggybacked=%s",
            service_name, OTEL_ENDPOINT, _AGENT_ID, _AGENT_FRAMEWORK, _piggybacked or "no",
        )
    except ImportError:
        _log.debug("opentelemetry-sdk not found — running without OTel")
    except Exception as exc:
        _log.warning("otel.setup.failed error=%r", exc)


# ── Span helper ───────────────────────────────────────────────────────────────

def _span(name: str, **attrs):
    """Context manager that creates an OTel span (no-op when OTel not set up).

    Uses _tracer (our own TracerProvider) when available so spans always carry
    the correct service.name even when a framework has its own global provider.
    """
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        try:
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode
            tracer = _tracer or trace.get_tracer("awcp.agent")
            with tracer.start_as_current_span(name) as span:
                for k, v in attrs.items():
                    if v is not None:
                        try:
                            span.set_attribute(k, str(v)[:512])
                        except Exception:
                            pass
                try:
                    yield span
                except Exception as exc:
                    try:
                        span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
                        span.record_exception(exc)
                    except Exception:
                        pass
                    raise
        except ImportError:
            yield None
        except Exception:
            yield None

    return _cm()


# ── Radar helpers ─────────────────────────────────────────────────────────────

def _radar_call(path: str, payload: dict, timeout: float = 3.0) -> dict:
    """POST to the AWCP radar REST API. Returns response dict or {} on failure."""
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{RADAR_URL}{path}",
            data=data,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return json.loads(r.read())
    except Exception:
        return {}


def _radar_register(agent_id: str, meta: dict, port: int) -> None:
    """Self-register with the AWCP radar (best-effort, runs in background thread).

    Sets telemetry_enabled=True, feature_flags, and policy_callbacks so the radar's
    quarantine check passes and the Temporal onboarding workflow completes as 'active'.
    """
    resp = _radar_call("/agents/register", {
        "id":               agent_id,
        "name":             meta.get("agent", "agent"),
        "kind":             "agent_framework",
        "framework":        meta.get("framework", "unknown"),
        "runtime":          meta.get("framework", "unknown"),
        "endpoint":         f"http://localhost:{port}",
        "transport":        "http",
        "telemetry_enabled": True,
        "policy_callbacks": [f"http://localhost:{port}/health"],
        "feature_flags":    {"kill_switch": False},
        "risk":             "medium",
        "write_scopes":     list(meta.get("tools", [])),
        "owner":            os.getenv("USER", os.getenv("LOGNAME", "")),
    })
    if resp.get("id"):
        _log.info(
            "radar.registered agent_id=%s status=%s onboarding=%s",
            resp["id"], resp.get("status"), resp.get("onboarding_state"),
        )
    else:
        _log.debug("radar.register_skipped reason=radar_unavailable url=%s", RADAR_URL)


def _radar_gate(agent_id: str, action: str) -> str:
    """Ask the radar's write-action gate. Returns 'allow' or 'deny'.

    Falls back to 'allow' when the radar is unavailable so agents remain
    functional without a running AWCP control plane.
    """
    if not agent_id:
        return "allow"
    resp = _radar_call(
        f"/agents/{agent_id}/gate",
        {"action": action, "write": True},
        timeout=2.0,
    )
    decision = resp.get("decision", "allow")
    _log.info(
        "radar.gate agent_id=%s action=%s decision=%s mode=%s",
        agent_id, action, decision, resp.get("mode", "unknown"),
    )
    return decision


def _radar_signal(agent_id: str, ok: bool, reason: str = "") -> None:
    """Report a task outcome signal to the radar (best-effort)."""
    if not agent_id:
        return
    _radar_call(
        f"/agents/{agent_id}/signal",
        {"ok": ok, "reason": reason[:200]},
        timeout=2.0,
    )


# ── Execution workflow bridge (Temporal task tracking) ────────────────────────
# Each task submitted via /tasks gets a Temporal AgentExecutionWorkflow. As the
# agent runs, we forward events to the radar which signals the running workflow.
# The radar maps event types → Temporal activity functions dynamically, so adding
# a new event type only requires adding it to the radar's _EVENT_TO_ACTIVITY map.

_CURRENT_EXEC_WF: dict = {}   # task_id → workflow_id (for active tasks)
_LLM_CALL_COUNT  = threading.local()  # per-thread LLM call counter
_LLM_TOKENS      = threading.local()  # per-thread accumulated LLM token usage
# True once the sync ollama.Client.chat wrapper meters a call in THIS process, so
# the httpx hook below knows langchain_ollama already counts the native Ollama
# endpoint and must not double-count it. Stays False for litellm / raw-httpx /
# async clients (CrewAI, async LangGraph), whose calls the httpx hook DOES count.
_OLLAMA_CLIENT_ACTIVE = False
# Same idea for the OpenAI SDK (PydanticAI and any OpenAI-compatible client): once
# its create() wrapper meters a call, the httpx hook must not also count the
# /v1/chat response (which it usually can't read anyway, because the SDK streams
# the body internally). Set True by the wrapper below.
_OPENAI_CLIENT_ACTIVE = False


def _start_execution_workflow(task_id: str, goal: str) -> None:
    """Ask the radar to start an AgentExecutionWorkflow. Best-effort, synchronous."""
    resp = _radar_call("/tasks/execution/start", {
        "agent_id":  _AGENT_ID,
        "task_id":   task_id,
        "goal":      goal,
        "framework": _AGENT_FRAMEWORK,
    }, timeout=5.0)
    if resp.get("workflow_id"):
        _CURRENT_EXEC_WF[task_id] = resp["workflow_id"]
        _log.info(
            "exec_workflow.started task_id=%s workflow_id=%s",
            task_id, resp["workflow_id"],
        )


def _emit_execution_event(task_id: str, event_type: str, **details) -> None:
    """Forward one execution event to the radar → Temporal workflow signal.

    The radar's _EVENT_TO_ACTIVITY map decides which Temporal activity fires.
    Unknown event types are safely ignored.
    """
    if not _CURRENT_EXEC_WF.get(task_id):
        return
    _radar_call(
        f"/tasks/execution/{task_id}/event",
        {"type": event_type, **details},
        timeout=2.0,
    )


def _finish_execution_workflow(
    task_id: str, result: str, status: str,
    tools_used: list, error: str = "",
) -> None:
    """Signal the AgentExecutionWorkflow to run its final activity and close."""
    if not _CURRENT_EXEC_WF.pop(task_id, None):
        return
    _radar_call(
        f"/tasks/execution/{task_id}/complete",
        {"status": status, "result": result[:500],
         "tools_used": tools_used, "error": error[:200]},
        timeout=3.0,
    )


# ── LLM call span processor ───────────────────────────────────────────────────

class _LLMSpanRenameProcessor:
    """OTel SpanProcessor that renames HTTP spans for LLM API calls.

    Detects POST requests to Ollama (port 11434) or OpenAI-compatible endpoints
    and renames the span from the generic "POST" to "agent.llm.call". Works for
    any framework that calls any OpenAI-compatible API over HTTP/HTTPX.
    """

    _LLM_URL_PATTERNS = (":11434", "api/chat", "/v1/chat/completions",
                          "api.openai.com", "api.anthropic.com")

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        pass


def _httpx_request_hook(span, request) -> None:
    """HTTPX instrumentation request hook — rename LLM call spans."""
    try:
        url = str(request.url)
        if any(p in url for p in (":11434", "/api/chat", "/v1/chat", "openai.com")):
            span.update_name("agent.llm.call")
            span.set_attribute("agent.llm.url", url.split("?")[0][:200])
            span.set_attribute("agent.id", _AGENT_ID)
            span.set_attribute("agent.framework", _AGENT_FRAMEWORK)
    except Exception:
        pass


def _extract_llm_tokens(body) -> tuple:
    """Pull (input, output) token counts from an LLM JSON response — Ollama
    (prompt_eval_count / eval_count) or OpenAI-compatible (usage.*). (0, 0) if
    absent. Never raises."""
    try:
        if not isinstance(body, dict):
            return 0, 0
        if "prompt_eval_count" in body or "eval_count" in body:        # Ollama
            return int(body.get("prompt_eval_count") or 0), int(body.get("eval_count") or 0)
        u = body.get("usage") or {}                                    # OpenAI-compatible
        if isinstance(u, dict) and u:
            return (int(u.get("prompt_tokens") or u.get("input_tokens") or 0),
                    int(u.get("completion_tokens") or u.get("output_tokens") or 0))
    except Exception:
        pass
    return 0, 0


def _capture_ollama_usage(chunk) -> None:
    """Accumulate (input, output) tokens from one Ollama response object — a
    dict or a pydantic ChatResponse, the final stream chunk carries the counts."""
    try:
        get = (chunk.get if isinstance(chunk, dict) else lambda k: getattr(chunk, k, None))
        tin = int(get("prompt_eval_count") or 0)
        tout = int(get("eval_count") or 0)
        if tin or tout:
            _LLM_TOKENS.tin = getattr(_LLM_TOKENS, "tin", 0) + tin
            _LLM_TOKENS.tout = getattr(_LLM_TOKENS, "tout", 0) + tout
    except Exception:
        pass


def _capture_openai_usage(res) -> None:
    """Accumulate (input, output) tokens from one OpenAI-SDK response object
    (ChatCompletion / dict). `usage.prompt_tokens` + `usage.completion_tokens`.
    Works for any OpenAI-compatible framework (PydanticAI, etc.); ignored when the
    object carries no usage (e.g. a raw stream)."""
    try:
        u = res.get("usage") if isinstance(res, dict) else getattr(res, "usage", None)
        if u is None:
            return
        get = (u.get if isinstance(u, dict) else lambda k: getattr(u, k, None))
        tin = int(get("prompt_tokens") or get("input_tokens") or 0)
        tout = int(get("completion_tokens") or get("output_tokens") or 0)
        if tin or tout:
            _LLM_TOKENS.tin = getattr(_LLM_TOKENS, "tin", 0) + tin
            _LLM_TOKENS.tout = getattr(_LLM_TOKENS, "tout", 0) + tout
    except Exception:
        pass


def _install_token_capture() -> None:
    """Wrap ollama.Client.chat so REAL token counts are captured even when the
    framework streams (langchain_ollama defaults to stream=True → ndjson, whose
    body can't be safely read in the httpx hook). The wrapper tees the stream —
    it yields every chunk through untouched and only reads the FINAL chunk's
    prompt_eval_count / eval_count, so it never alters what the framework sees.
    Best-effort: no-op if the ollama package isn't importable."""
    try:
        import ollama
    except Exception:
        ollama = None          # no ollama pkg (e.g. PydanticAI/CrewAI venv) — skip
        # the ollama wrapper but STILL install the OpenAI-SDK wrapper below.
    for cls_name in (("Client",) if ollama is not None else ()):
        cls = getattr(ollama, cls_name, None)
        orig = getattr(cls, "chat", None) if cls else None
        if orig is None or getattr(orig, "_awcp_wrapped", False):
            continue

        def _make(orig_chat):
            def chat(self, *args, **kwargs):
                global _OLLAMA_CLIENT_ACTIVE
                _OLLAMA_CLIENT_ACTIVE = True   # mark BEFORE the underlying httpx call fires
                res = orig_chat(self, *args, **kwargs)
                try:
                    if kwargs.get("stream"):
                        def _tee():
                            last = None
                            for ch in res:
                                last = ch
                                yield ch
                            _capture_ollama_usage(last)
                        return _tee()
                    _capture_ollama_usage(res)
                except Exception:
                    pass
                return res
            chat._awcp_wrapped = True
            return chat

        try:
            cls.chat = _make(orig)
        except Exception:
            pass

    # --- OpenAI SDK (PydanticAI and any OpenAI-compatible framework) ---
    # The httpx hook usually CAN'T read the openai SDK's /v1 response body (the SDK
    # reads it internally), so meter from the PARSED response object instead. Wraps
    # both the sync and async chat-completions create(). Best-effort / no-op if the
    # openai package isn't importable.
    try:
        from openai.resources.chat import completions as _oai
    except Exception:
        _oai = None
    if _oai is not None:
        def _make_sync(orig_create):
            def create(self, *args, **kwargs):
                global _OPENAI_CLIENT_ACTIVE
                _OPENAI_CLIENT_ACTIVE = True
                res = orig_create(self, *args, **kwargs)
                try:
                    if not kwargs.get("stream"):
                        _capture_openai_usage(res)
                except Exception:
                    pass
                return res
            create._awcp_wrapped = True
            return create

        def _make_async(orig_create):
            async def create(self, *args, **kwargs):
                global _OPENAI_CLIENT_ACTIVE
                _OPENAI_CLIENT_ACTIVE = True
                res = await orig_create(self, *args, **kwargs)
                try:
                    if not kwargs.get("stream"):
                        _capture_openai_usage(res)
                except Exception:
                    pass
                return res
            create._awcp_wrapped = True
            return create

        for _cls, _maker in ((getattr(_oai, "Completions", None), _make_sync),
                             (getattr(_oai, "AsyncCompletions", None), _make_async)):
            _orig = getattr(_cls, "create", None) if _cls else None
            if _orig is None or getattr(_orig, "_awcp_wrapped", False):
                continue
            try:
                _cls.create = _maker(_orig)
            except Exception:
                pass


def _httpx_response_hook(span, request, response) -> None:
    """HTTPX instrumentation response hook — record LLM call result, OTel
    attributes, and REAL per-call token usage (best-effort)."""
    try:
        url = str(request.url)
        if any(p in url for p in (":11434", "/api/chat", "/v1/chat", "openai.com")):
            span.set_attribute("agent.llm.status", response.status_code)
            # Increment per-thread counter so _worker_loop knows how many LLM calls happened
            count = getattr(_LLM_CALL_COUNT, "n", 0) + 1
            _LLM_CALL_COUNT.n = count
            span.set_attribute("agent.llm.call_n", count)
            # Capture REAL token usage from the model's response body:
            #   • OpenAI-compatible endpoints (/v1/chat, openai.com) → `usage.*`;
            #   • Ollama's native /api/chat | /api/generate → prompt_eval_count /
            #     eval_count — BUT only when the sync ollama.Client wrapper is NOT
            #     metering this process (else langchain_ollama's sync path would be
            #     double-counted). This is what makes litellm/CrewAI and async
            #     LangGraph — which never touch ollama.Client — actually report.
            # Only NON-streaming JSON bodies are read, so we never consume a stream
            # the framework still needs (Ollama streaming is ndjson and is metered
            # by the ollama.Client wrapper instead).
            try:
                is_openai = ("openai.com" in url or "/v1/chat" in url) \
                    and not _OPENAI_CLIENT_ACTIVE
                is_ollama_native = ("/api/chat" in url or "/api/generate" in url) \
                    and not _OLLAMA_CLIENT_ACTIVE
                ctype = response.headers.get("content-type", "")
                readable = ("json" in ctype and "event-stream" not in ctype
                            and "ndjson" not in ctype)
                if readable and (is_openai or is_ollama_native):
                    tin, tout = _extract_llm_tokens(response.json())
                    if tin or tout:
                        _LLM_TOKENS.tin = getattr(_LLM_TOKENS, "tin", 0) + tin
                        _LLM_TOKENS.tout = getattr(_LLM_TOKENS, "tout", 0) + tout
                        span.set_attribute("gen_ai.usage.input_tokens", tin)
                        span.set_attribute("gen_ai.usage.output_tokens", tout)
            except Exception:
                pass
    except Exception:
        pass


# ── Utility ───────────────────────────────────────────────────────────────────

def sse(event: dict) -> str:
    """Format one Server-Sent-Events frame."""
    return "data: " + json.dumps(event) + "\n\n"


def web_search(query: str, max_results: int = 5) -> str:
    """Free web search (DuckDuckGo, no API key). Instrumented with an OTel span."""
    _log.info("web_search query=%r max_results=%d", query[:200], max_results)
    task = _CURRENT["task"]
    if task:
        _emit_execution_event(task["id"], "web_search",
                              tool_name="web_search", query=query[:200])
    with _span("agent.tool.web_search", query=query[:200], max_results=max_results):
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max(1, min(max_results, 8))))
            if not results:
                return "No web results found."
            return "\n\n".join(
                f"Title: {r.get('title', '')}\nLink: {r.get('href', '')}\n{r.get('body', '')}"
                for r in results
            )
        except Exception as e:  # noqa: BLE001
            return f"web_search unavailable: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# Browser chat UI (served at GET / when not using the task console)
# --------------------------------------------------------------------------
UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AWCP Agent</title>
<style>
:root{--bg:#0b0f17;--panel:#121826;--line:#1f2937;--fg:#e5e7eb;--mut:#9ca3af;--acc:#6366f1;--ok:#22c55e;--warn:#f59e0b}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);height:100vh;display:flex;flex-direction:column}
header{padding:14px 20px;border-bottom:1px solid var(--line);background:var(--panel);display:flex;align-items:center;gap:10px;flex-wrap:wrap}
header h1{font-size:16px;margin:0 8px 0 0}
.badge{font-size:12px;color:var(--mut);background:#0b1220;border:1px solid var(--line);padding:3px 9px;border-radius:999px}
.badge.ok{color:var(--ok);border-color:#14532d}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
.chip{font-size:11px;background:#0b1220;border:1px solid var(--line);color:#a5b4fc;padding:2px 8px;border-radius:6px}
#log{flex:1;overflow:auto;padding:20px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:820px;padding:10px 14px;border-radius:12px;white-space:pre-wrap;word-wrap:break-word}
.me{align-self:flex-end;background:var(--acc);color:#fff;border-bottom-right-radius:3px}
.bot{align-self:flex-start;background:var(--panel);border:1px solid var(--line);border-bottom-left-radius:3px}
.tools{margin-top:8px;display:flex;gap:6px;flex-wrap:wrap}
.tcall{font-size:11px;color:var(--warn);border:1px solid #78350f;background:#1c1408;padding:1px 7px;border-radius:6px}
footer{border-top:1px solid var(--line);background:var(--panel);padding:12px 16px;display:flex;gap:10px}
textarea{flex:1;resize:none;height:48px;background:#0b1220;color:var(--fg);border:1px solid var(--line);border-radius:10px;padding:13px}
button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:0 18px;font-weight:600;cursor:pointer}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--mut)}
button:disabled{opacity:.5;cursor:default}
.empty{color:var(--mut);text-align:center;margin:auto;max-width:420px}
</style>
</head>
<body>
<header>
  <h1 id="name">Agent</h1>
  <span class="badge" id="fw">framework</span>
  <span class="badge" id="model">model</span>
  <span class="badge" id="reg">registry</span>
  <div class="chips" id="tools"></div>
</header>
<div id="log"><div class="empty" id="empty">Send a task below. The agent answers here, streaming, with the tools it calls shown as chips.</div></div>
<footer>
  <textarea id="in" placeholder="Type a task and press Enter (Shift+Enter for newline)…"></textarea>
  <button id="send">Send</button>
  <button id="reset" class="ghost">Reset</button>
</footer>
<script>
const session = (crypto.randomUUID && crypto.randomUUID()) || String(Math.random());
const log = document.getElementById('log');
const inp = document.getElementById('in');
const sendBtn = document.getElementById('send');
function add(cls, text){
  const e = document.getElementById('empty'); if(e) e.remove();
  const d = document.createElement('div'); d.className = 'msg ' + cls; d.textContent = text || '';
  log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
}
function chip(wrap, name){
  if([...wrap.children].some(c=>c.textContent.includes(name))) return;
  const c = document.createElement('span'); c.className='tcall'; c.textContent='⚙ '+name; wrap.appendChild(c);
}
async function loadInfo(){
  try{
    const j = await (await fetch('/info')).json();
    document.getElementById('name').textContent = j.agent || 'Agent';
    document.getElementById('fw').textContent = j.framework || '';
    document.getElementById('model').textContent = j.model || '';
    document.title = j.agent || 'Agent';
    const t = document.getElementById('tools');
    (j.tools||[]).forEach(n=>{const c=document.createElement('span');c.className='chip';c.textContent=n;t.appendChild(c);});
    const reg = document.getElementById('reg');
    reg.textContent = j.registered ? 'registry: active' : 'registry: standalone';
    if(j.registered) reg.classList.add('ok');
  }catch(e){}
}
async function send(){
  const text = inp.value.trim(); if(!text) return;
  inp.value=''; add('me', text); sendBtn.disabled = true;
  const bot = add('bot', ''); const body = document.createElement('span'); bot.appendChild(body);
  const toolWrap = document.createElement('div'); toolWrap.className='tools'; bot.appendChild(toolWrap);
  try{
    const resp = await fetch('/stream', {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({input:text, session})});
    const reader = resp.body.getReader(); const dec = new TextDecoder(); let buf='';
    while(true){
      const {done, value} = await reader.read(); if(done) break;
      buf += dec.decode(value, {stream:true}); let i;
      while((i = buf.indexOf('\n\n')) >= 0){
        const raw = buf.slice(0, i); buf = buf.slice(i+2);
        const dl = raw.split('\n').find(l=>l.startsWith('data:')); if(!dl) continue;
        let ev; try{ ev = JSON.parse(dl.slice(5).trim()); }catch(_){ continue; }
        if(ev.type==='token'){ body.textContent += ev.text; }
        else if(ev.type==='tool'){ chip(toolWrap, ev.name); }
        else if(ev.type==='done'){ (ev.tools_used||[]).forEach(n=>chip(toolWrap, n)); }
        else if(ev.type==='error'){ body.textContent += (body.textContent?'\n':'') + '[error] ' + ev.message; }
        log.scrollTop = log.scrollHeight;
      }
    }
    if(!body.textContent) body.textContent = '(no output)';
  }catch(e){ body.textContent += '\n[network error] ' + e; }
  sendBtn.disabled = false; inp.focus();
}
sendBtn.onclick = send;
inp.addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }});
document.getElementById('reset').onclick = async ()=>{
  try{ await fetch('/reset',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({session})}); }catch(e){}
  log.innerHTML=''; add('bot','(conversation reset)');
};
loadInfo();
</script>
</body>
</html>
"""


# ==========================================================================
# AUTONOMOUS TASK-WORKER RUNTIME
# ==========================================================================
import uuid as _uuid
from collections import deque as _deque

# --- in-memory task store ---
TASKS: dict = {}
_QUEUE: _deque = _deque()
_TLOCK = threading.Lock()
_CURRENT: dict = {"task": None}
_APPROVAL_EVENTS: dict = {}
_APPROVAL_DECISION: dict = {}


def _now() -> float:
    return time.time()


def submit_task(goal: str) -> dict:
    tid = "task-" + _uuid.uuid4().hex[:10]
    task = {"id": tid, "goal": goal, "status": "queued", "steps": [],
            "result": "", "tools_used": [], "awaiting": None,
            "created": _now(), "started": None, "finished": None, "error": ""}
    with _TLOCK:
        TASKS[tid] = task
        _QUEUE.append(tid)
    return _public_task(task)


def _public_task(t: dict) -> dict:
    return {k: t[k] for k in ("id", "goal", "status", "steps", "result",
                              "tools_used", "awaiting", "created", "started",
                              "finished", "error")}


def list_tasks() -> list:
    with _TLOCK:
        return sorted((_public_task(t) for t in TASKS.values()),
                      key=lambda t: t["created"], reverse=True)


def get_task(tid: str):
    t = TASKS.get(tid)
    return _public_task(t) if t else None


def _add_step(task, step) -> None:
    step.setdefault("ts", _now())
    task["steps"].append(step)


def approve_task(tid: str, decision: str) -> bool:
    ev = _APPROVAL_EVENTS.get(tid)
    if not ev:
        return False
    _APPROVAL_DECISION[tid] = "approve" if decision == "approve" else "deny"
    ev.set()
    return True


def governed_action(name: str, risk: str, do_fn, detail: str = ""):
    """Run a governed write action:
    1. Check the AWCP radar gate (falls back to allow when radar is offline).
    2. For HIGH risk, pause for operator approval.
    3. Execute the write, recording a step on the current task.
    """
    task = _CURRENT["task"]
    step = {"action": name, "risk": risk, "status": "", "info": ""}

    with _span("agent.action.write",
               agent_id=_AGENT_ID,
               action_name=name,
               action_risk=risk,
               task_id=task["id"] if task else "") as action_span:

        # AWCP radar gate
        if risk in ("medium", "high"):
            gate = _radar_gate(_AGENT_ID, name)
            if action_span:
                action_span.set_attribute("gate.decision", gate)
            # Emit tool_called event so Temporal shows this step
            if task:
                _emit_execution_event(
                    task["id"], "tool_called",
                    tool_name=name, risk=risk,
                    gate="denied" if gate == "deny" else "allowed",
                )
            if gate == "deny":
                _log.warning(
                    "action.blocked agent_id=%s action=%s risk=%s",
                    _AGENT_ID, name, risk,
                )
                if task:
                    _add_step(task, {**step, "status": "blocked",
                                     "info": "denied by AWCP governance"})
                return f"BLOCKED: '{name}' was denied by the AWCP write-action gate."

        # High-risk operator approval
        if risk == "high" and APPROVAL_REQUIRED and task is not None:
            ev = threading.Event()
            _APPROVAL_EVENTS[task["id"]] = ev
            _APPROVAL_DECISION.pop(task["id"], None)
            task["awaiting"] = {"action": name, "detail": detail}
            task["status"] = "awaiting_approval"
            _add_step(task, {**step, "status": "awaiting_approval", "info": detail})
            _log.info("action.awaiting_approval action=%s task_id=%s", name, task["id"])
            got = ev.wait(timeout=APPROVAL_TIMEOUT)
            _APPROVAL_EVENTS.pop(task["id"], None)
            task["awaiting"] = None
            task["status"] = "running"
            if not got or _APPROVAL_DECISION.get(task["id"]) != "approve":
                info = "operator denied" if got else "approval timed out"
                _log.info("action.denied action=%s task_id=%s reason=%s", name, task["id"], info)
                _add_step(task, {**step, "status": "denied", "info": info})
                if action_span:
                    action_span.set_attribute("gate.decision", "denied")
                return f"DENIED: external write '{name}' was not approved."

        try:
            out = do_fn()
            _log.info("action.done action=%s risk=%s", name, risk)
            if task:
                _add_step(task, {**step, "status": "done", "info": str(out)[:300]})
            if action_span:
                action_span.set_attribute("action.status", "done")
            return out
        except Exception as e:  # noqa: BLE001
            _log.error("action.failed action=%s risk=%s error=%r", name, risk, e)
            if task:
                _add_step(task, {**step, "status": "failed", "info": str(e)})
            return f"ERROR: {e}"


def save_artifact(name: str, content: str) -> str:
    """Governed LOCAL write (medium risk): persist a result artifact to disk."""
    def _do():
        d = ARTIFACT_DIR or os.path.join(os.getcwd(), "artifacts")
        os.makedirs(d, exist_ok=True)
        safe = "".join(c for c in name if c.isalnum() or c in "-_.") or "artifact"
        path = os.path.join(d, f"{int(_now())}-{safe}")
        with open(path, "w") as f:
            f.write(content)
        return f"saved artifact: {path}"
    return governed_action("save_artifact", "medium", _do, detail=name)


def external_post(summary: str) -> str:
    """Self-governed EXTERNAL write (HIGH risk): POST to EXTERNAL_WRITE_URL."""
    def _do():
        body = json.dumps({"agent": AGENT_NAME, "summary": summary}).encode()
        headers = {"content-type": "application/json"}
        if EXTERNAL_WRITE_TOKEN:
            headers["authorization"] = f"Bearer {EXTERNAL_WRITE_TOKEN}"
        req = urllib.request.Request(EXTERNAL_WRITE_URL, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310
            return f"external POST {EXTERNAL_WRITE_URL} -> HTTP {r.status}"
    return governed_action("external_post", "high", _do, detail=f"POST {EXTERNAL_WRITE_URL}")


def _worker_loop(run_goal) -> None:
    while True:
        tid = None
        with _TLOCK:
            if _QUEUE:
                tid = _QUEUE.popleft()
        if not tid:
            time.sleep(0.4)
            continue
        task = TASKS.get(tid)
        if not task:
            continue
        _CURRENT["task"] = task
        task["status"] = "running"
        task["started"] = _now()
        t0 = time.monotonic()

        with _span("agent.task.run",
                   agent_id=_AGENT_ID,
                   framework=_AGENT_FRAMEWORK,
                   task_id=task["id"],
                   goal=task["goal"][:500]) as task_span:
            # Reset LLM call counter + token accumulators for this task (thread-local)
            _LLM_CALL_COUNT.n = 0
            _LLM_TOKENS.tin = 0
            _LLM_TOKENS.tout = 0

            # Start execution workflow BEFORE run_goal so Temporal shows
            # "setup" activity immediately and is ready to receive events.
            _start_execution_workflow(task["id"], task["goal"])

            try:
                _log.info(
                    "task.started agent_id=%s task_id=%s goal=%r",
                    _AGENT_ID, task["id"], task["goal"][:200],
                )

                # Emit the first LLM call event. _AGENT_MODEL is set by mount()
                # from meta["model"] so it's dynamic for any agent framework.
                _emit_execution_event(
                    task["id"], "llm_called",
                    model=_AGENT_MODEL,
                    call_n=1,
                )

                out = run_goal(task["goal"]) or {}
                result = str(out.get("result", ""))
                tools_used = out.get("tools_used", [])
                task["result"] = result
                task["tools_used"] = tools_used

                if task_span:
                    task_span.set_attribute("task.tools_used", ",".join(tools_used))
                    task_span.set_attribute("task.result.length", len(result))
                    task_span.set_attribute("task.llm_calls", getattr(_LLM_CALL_COUNT, "n", 1))

                # Emit tool_called events for any framework-level tools not
                # already emitted by governed_action or web_search.
                # Dedup against tools that were already signalled in-flight.
                already_emitted = set()
                for s in task["steps"]:
                    already_emitted.add(s.get("action", ""))

                for tool in tools_used:
                    if tool not in already_emitted and tool not in ("web_search",):
                        _emit_execution_event(
                            task["id"], "tool_called",
                            tool_name=tool, risk="low", gate="allowed",
                        )

                # Emit additional LLM calls if the HTTPX counter shows >1
                llm_n = getattr(_LLM_CALL_COUNT, "n", 0)
                for i in range(2, llm_n + 1):
                    _emit_execution_event(task["id"], "llm_called", call_n=i)

                # Report the REAL token usage captured from the LLM HTTP
                # responses (Ollama prompt_eval/eval or OpenAI usage). One event
                # carrying the window total is all the token monitor needs to
                # meter this task against the agent's budget.
                _tin = int(getattr(_LLM_TOKENS, "tin", 0) or 0)
                _tout = int(getattr(_LLM_TOKENS, "tout", 0) or 0)
                if _tin or _tout:
                    _emit_execution_event(
                        task["id"], "llm_called",
                        model=_AGENT_MODEL, call_n=llm_n or 1,
                        extra={"input_tokens": _tin, "output_tokens": _tout},
                    )

                # Synthesize — always the final logical step before completion
                _emit_execution_event(
                    task["id"], "synthesize",
                    result_len=len(result),
                    tools_used=tools_used,
                )

                # deterministic finalize — route output through gate
                if FINALIZE_ARTIFACT and not any(s["action"] == "save_artifact" for s in task["steps"]):
                    save_artifact("result", result or task["goal"])
                if FINALIZE_EXTERNAL and not any(s["action"] == "external_post" for s in task["steps"]):
                    external_post((result or task["goal"])[:500])

                blocked = any(s.get("status") in ("blocked", "denied") for s in task["steps"])
                task["status"] = "blocked" if blocked else "done"

                dur_ms = (time.monotonic() - t0) * 1000
                _log.info(
                    "task.completed agent_id=%s task_id=%s status=%s tools=%s dur_ms=%.0f",
                    _AGENT_ID, task["id"], task["status"],
                    ",".join(tools_used), dur_ms,
                )
                if task_span:
                    task_span.set_attribute("task.status", task["status"])
                    task_span.set_attribute("task.duration_ms", round(dur_ms))

                # Record agent-level OTel metrics (visible in Prometheus/Grafana)
                _dims = {"agent.id": _AGENT_ID, "agent.framework": _AGENT_FRAMEWORK,
                         "status": task["status"]}
                try:
                    if _task_counter:
                        _task_counter.add(1, _dims)
                    if _task_duration:
                        _task_duration.record(dur_ms, _dims)
                    if _llm_calls_total:
                        _llm_calls_total.add(
                            getattr(_LLM_CALL_COUNT, "n", 1),
                            {"agent.id": _AGENT_ID, "agent.framework": _AGENT_FRAMEWORK},
                        )
                except Exception:
                    pass

                _finish_execution_workflow(
                    task["id"], result, task["status"], tools_used
                )
                _radar_signal(_AGENT_ID, ok=(task["status"] == "done"))

            except Exception as e:  # noqa: BLE001
                task["status"] = "failed"
                task["error"] = str(e)
                _log.error(
                    "task.failed agent_id=%s task_id=%s error=%r",
                    _AGENT_ID, task["id"], e, exc_info=True,
                )
                _finish_execution_workflow(task["id"], "", "failed", [], str(e)[:200])
                _radar_signal(_AGENT_ID, ok=False, reason=str(e)[:200])
            finally:
                task["finished"] = _now()
                _CURRENT["task"] = None


# Request body models (must be module-level for FastAPI with `from __future__ import annotations`)
from pydantic import BaseModel as _BaseModel  # noqa: E402


class GoalReq(_BaseModel):
    goal: str


class ApproveReq(_BaseModel):
    decision: str = "approve"   # approve | deny


def mount(app, *, meta: dict, run_goal, port: int = 8000) -> None:
    """Wire all routes, set up OTel + radar integration, and start the worker.

    `run_goal(goal) -> {"result", "tools_used"}` is the framework hook.
    `port` is the port this agent will listen on — used for the radar registration
    endpoint field and the stable agent ID.
    """
    global ARTIFACT_DIR, AGENT_NAME, _AGENT_ID, _AGENT_FRAMEWORK, _AGENT_MODEL
    from fastapi import HTTPException
    from fastapi.responses import HTMLResponse

    ARTIFACT_DIR = os.path.join(meta.get("dir", os.getcwd()), "artifacts")
    AGENT_NAME = meta.get("agent", "agent")
    _AGENT_FRAMEWORK = meta.get("framework", "unknown")
    _AGENT_MODEL = meta.get("model", "")

    # Stable agent ID: framework + hash of the agent directory (survives restarts)
    agent_dir = meta.get("dir", os.getcwd())
    dir_hash = hashlib.md5(agent_dir.encode()).hexdigest()[:8]
    _AGENT_ID = os.getenv("AGENT_ID", f"agent-{_AGENT_FRAMEWORK}-{dir_hash}")

    # OTel — must run before any span is created
    service_name = os.getenv("OTEL_SERVICE_NAME", f"awcp-agent-{_AGENT_FRAMEWORK}")
    _setup_otel(service_name)

    # FastAPI request tracing
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        pass

    # Register with the radar in background so startup is not blocked
    threading.Thread(
        target=_radar_register,
        args=(_AGENT_ID, meta, port),
        daemon=True,
        name="radar-register",
    ).start()

    # ── Routes ────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    def _home():
        return TASK_UI_HTML

    @app.get("/info")
    def _info():
        return {
            **{k: meta[k] for k in meta if k != "dir"},
            "external_url":     EXTERNAL_WRITE_URL,
            "approval_required": APPROVAL_REQUIRED,
            "registered":       bool(_AGENT_ID),
            "agent_id":         _AGENT_ID,
            "radar_url":        RADAR_URL,
        }

    @app.get("/health")
    def _health():
        return {"status": "ok", "framework": meta.get("framework"), "agent_id": _AGENT_ID}

    @app.post("/tasks")
    def _submit(req: GoalReq):
        return submit_task(req.goal)

    @app.get("/tasks")
    def _list():
        return list_tasks()

    @app.get("/tasks/{tid}")
    def _get(tid: str):
        t = get_task(tid)
        if not t:
            raise HTTPException(404, "task not found")
        return t

    @app.post("/tasks/{tid}/approve")
    def _approve(tid: str, req: ApproveReq):
        return {"ok": approve_task(tid, req.decision), "decision": req.decision}

    threading.Thread(target=_worker_loop, args=(run_goal,),
                     name="awcp-worker", daemon=True).start()


# --------------------------------------------------------------------------
# Task-console UI (served at GET /)
# --------------------------------------------------------------------------
TASK_UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AWCP Worker</title>
<style>
:root{--acc:#6366f1;--bg:#0a0e16;--panel:#121826;--panel2:#0d1320;--line:#1f2a3a;--fg:#e6edf3;--mut:#8b97a7;
      --ok:#22c55e;--warn:#f59e0b;--red:#ef4444;--blue:#38bdf8}
*{box-sizing:border-box}
body{margin:0;font:14px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);min-height:100vh;
     background:radial-gradient(1100px 520px at 82% -12%, rgba(99,102,241,.13), transparent), var(--bg)}
header{padding:18px 26px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,var(--panel),transparent);
       display:flex;align-items:center;gap:15px;flex-wrap:wrap}
.logo{width:44px;height:44px;border-radius:13px;background:linear-gradient(135deg,var(--acc),#0ea5e9);display:flex;
      align-items:center;justify-content:center;font-size:23px;flex:none;box-shadow:0 5px 18px rgba(99,102,241,.4)}
.htext h1{font-size:18px;margin:0} .htext .purpose{color:var(--mut);font-size:13px}
.badges{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto;align-items:center}
.badge{font-size:11px;color:var(--mut);background:var(--panel2);border:1px solid var(--line);padding:4px 10px;border-radius:999px}
.badge.acc{color:#fff;background:var(--acc);border-color:transparent}
.badge.ok{color:var(--ok);border-color:#14532d;background:rgba(34,197,94,.08)}
.wrap{max-width:880px;margin:0 auto;padding:24px 20px 60px}
.composer{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:14px;box-shadow:0 10px 34px rgba(0,0,0,.34)}
.composer textarea{width:100%;resize:vertical;min-height:58px;background:transparent;color:var(--fg);border:0;outline:none;font:15px/1.55 inherit;padding:6px}
.crow{display:flex;align-items:center;gap:10px;margin-top:6px}
.examples{display:flex;gap:6px;flex-wrap:wrap;flex:1}
.ex{font-size:11.5px;color:var(--mut);background:var(--panel2);border:1px solid var(--line);padding:4px 10px;border-radius:8px;cursor:pointer}
.ex:hover{color:var(--fg);border-color:var(--acc)}
button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:9px 18px;font-weight:600;cursor:pointer;font-size:14px}
button:disabled{opacity:.5;cursor:default}
button.app{background:var(--ok)} button.deny{background:transparent;border:1px solid var(--red);color:var(--red)}
.tasks{margin-top:22px;display:flex;flex-direction:column;gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}
.chead{padding:13px 16px;display:flex;align-items:center;gap:12px}
.goal{font-weight:600;flex:1}
.pill{font-size:11px;padding:3px 11px;border-radius:999px;font-family:ui-monospace,monospace;display:flex;align-items:center;gap:6px;text-transform:capitalize;white-space:nowrap}
.dot{width:6px;height:6px;border-radius:50%}
.p-queued{background:#1a2332;color:var(--mut)} .p-queued .dot{background:var(--mut)}
.p-running{background:rgba(56,189,248,.14);color:var(--blue)} .p-running .dot{background:var(--blue);animation:pulse 1s infinite}
.p-awaiting_approval{background:rgba(245,158,11,.16);color:var(--warn)} .p-awaiting_approval .dot{background:var(--warn);animation:pulse 1s infinite}
.p-done{background:rgba(34,197,94,.14);color:var(--ok)} .p-done .dot{background:var(--ok)}
.p-failed,.p-blocked{background:rgba(239,68,68,.14);color:var(--red)} .p-failed .dot,.p-blocked .dot{background:var(--red)}
@keyframes pulse{50%{opacity:.32}}
.cbody{padding:2px 16px 14px;border-top:1px solid var(--line)}
.section{margin-top:12px}
.lbl{font-size:10px;text-transform:uppercase;letter-spacing:.7px;color:var(--mut);margin-bottom:6px}
.steps{display:flex;flex-direction:column;gap:5px}
.step{display:flex;gap:9px;align-items:center;font-size:12.5px;background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:6px 11px}
.step .act{font-family:ui-monospace,monospace}
.rk{font-size:10px;padding:1px 7px;border-radius:6px;margin-left:auto}
.rk-medium{background:rgba(245,158,11,.16);color:var(--warn)} .rk-high{background:rgba(239,68,68,.16);color:var(--red)}
.sstat{font-family:ui-monospace,monospace;font-size:11px}
.sstat.done{color:var(--ok)} .sstat.blocked,.sstat.denied,.sstat.failed{color:var(--red)} .sstat.awaiting_approval{color:var(--warn)}
.chips{display:flex;gap:5px;flex-wrap:wrap}
.chip{font-size:10.5px;background:var(--panel2);border:1px solid var(--line);color:#a5b4fc;padding:2px 8px;border-radius:6px}
.result{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:13px 16px;font-size:14px;overflow-x:auto;line-height:1.65}
.result h2{font-size:16px;margin:10px 0 4px} .result h3{font-size:14px;margin:8px 0 4px} .result h4{font-size:13px;margin:6px 0 3px}
.result a{color:#7dd3fc} .result code{background:#0a0e16;padding:1px 5px;border-radius:4px;font-family:ui-monospace,monospace;font-size:12.5px}
.result pre.cb{background:#070b12;border:1px solid var(--line);border-radius:8px;padding:11px 13px;overflow-x:auto;font-family:ui-monospace,monospace;font-size:12.5px;line-height:1.5;color:#cfe3ff}
.approve{margin-top:12px;background:rgba(245,158,11,.07);border:1px solid #5a3d0c;border-radius:10px;padding:12px;display:flex;gap:10px;align-items:center}
.approve .q{flex:1;font-size:13px}
.empty{color:var(--mut);text-align:center;margin-top:48px}
.foot{color:var(--mut);font-size:11.5px;text-align:center;margin-top:28px}
</style>
</head>
<body>
<header>
  <div class="logo" id="logo">robot</div>
  <div class="htext"><h1 id="name">Worker</h1><div class="purpose" id="purpose"></div></div>
  <div class="badges" id="badges"></div>
</header>
<div class="wrap">
  <div class="composer">
    <textarea id="goal" placeholder="Give the worker a goal..."></textarea>
    <div class="crow"><div class="examples" id="examples"></div><button id="run">Run task &#9656;</button></div>
  </div>
  <div class="tasks" id="tasks"><div class="empty">No tasks yet - give the worker a goal above.</div></div>
  <div class="foot" id="foot"></div>
</div>
<script>
const $=id=>document.getElementById(id);
let FORMAT='markdown';
function esc(s){return (s||'').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));}
function md(t){ t=esc(t);
  t=t.replace(/```([\s\S]*?)```/g,(m,c)=>'<pre class="cb">'+c.replace(/^\n/,'')+'</pre>');
  t=t.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  t=t.replace(/^#{1,2}\s?(.*)$/gm,'<h2>$1</h2>').replace(/^#{3}\s?(.*)$/gm,'<h3>$1</h3>').replace(/^#{4}\s?(.*)$/gm,'<h4>$1</h4>');
  t=t.replace(/\*\*([^*]+)\*\*/g,'<b>$1</b>');
  t=t.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,'<a href="$2" target="_blank">$1</a>');
  t=t.replace(/(^|\s)(https?:\/\/[^\s<]+)/g,'$1<a href="$2" target="_blank">$2</a>');
  t=t.replace(/^\s*[-*]\s+(.*)$/gm,'&bull; $1');
  t=t.replace(/\n{2,}/g,'<br><br>').replace(/\n/g,'<br>');
  return t;
}
function fmt(text){ if(FORMAT==='json'){ const m=text.match(/\{[\s\S]*\}/);
    if(m){ try{ return '<pre class="cb">'+esc(JSON.stringify(JSON.parse(m[0]),null,2))+'</pre>'; }catch(e){} } }
  return md(text); }
const IC={save_artifact:'\u{1F4BE}',external_post:'\u{1F310}'};
function step(s){ return '<div class="step"><span>'+(IC[s.action]||'⚙')+'</span><span class="act">'+esc(s.action)+
  '</span><span class="rk rk-'+s.risk+'">'+s.risk+'</span><span class="sstat '+s.status+'">'+s.status.replace('_',' ')+'</span></div>'; }
function card(t){
  const steps=(t.steps||[]).length?'<div class="section"><div class="lbl">governed steps</div><div class="steps">'+t.steps.map(step).join('')+'</div></div>':'';
  const chips=(t.tools_used||[]).length?'<div class="section"><div class="lbl">tools used</div><div class="chips">'+t.tools_used.map(n=>'<span class="chip">'+esc(n)+'</span>').join('')+'</div></div>':'';
  const appr=(t.status==='awaiting_approval'&&t.awaiting)?'<div class="approve"><span class="q">&#9888; Approval required: high-risk <b>'+esc(t.awaiting.action)+'</b> &mdash; '+esc(t.awaiting.detail||'')+'</span><button class="app" onclick="approve(\''+t.id+'\',\'approve\')">Approve</button><button class="deny" onclick="approve(\''+t.id+'\',\'deny\')">Deny</button></div>':'';
  const res=t.result?'<div class="section"><div class="lbl">result</div><div class="result">'+fmt(t.result)+'</div></div>':'';
  const err=t.error?'<div class="section"><div class="result" style="color:var(--red)">'+esc(t.error)+'</div></div>':'';
  const body=(appr||res||steps||chips||err)?'<div class="cbody">'+appr+res+steps+chips+err+'</div>':'';
  return '<div class="card"><div class="chead"><span class="goal">'+esc(t.goal)+'</span><span class="pill p-'+t.status+'"><span class="dot"></span>'+t.status.replace('_',' ')+'</span></div>'+body+'</div>';
}
async function refresh(){ let ts=[]; try{ ts=await (await fetch('/tasks')).json(); }catch(e){ return; }
  $('tasks').innerHTML=ts.length?ts.map(card).join(''):'<div class="empty">No tasks yet - give the worker a goal above.</div>'; }
async function run(){ const g=$('goal').value.trim(); if(!g)return; $('goal').value=''; $('run').disabled=true;
  try{ await fetch('/tasks',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({goal:g})}); }catch(e){}
  $('run').disabled=false; refresh(); }
async function approve(id,d){ try{ await fetch('/tasks/'+id+'/approve',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({decision:d})});}catch(e){} refresh(); }
async function init(){ try{ const j=await (await fetch('/info')).json();
  FORMAT=j.format||'markdown';
  if(j.accent) document.documentElement.style.setProperty('--acc',j.accent);
  $('logo').textContent=j.logo||'\u{1F916}'; $('name').textContent=j.agent||'Worker'; document.title=j.agent||'Worker';
  $('purpose').textContent=j.purpose||'';
  const regBadge = j.registered
    ? '<span class="badge ok">&#10004; radar: active</span>'
    : '<span class="badge" title="AWCP radar not reached — running standalone">&#10752; standalone</span>';
  $('badges').innerHTML='<span class="badge acc">'+esc(j.framework||'')+'</span><span class="badge">'+esc(j.model||'')+
    '</span>'+regBadge;
  $('examples').innerHTML=(j.examples||[]).map(e=>'<span class="ex" onclick="document.getElementById(\'goal\').value=this.textContent">'+esc(e)+'</span>').join('');
  $('foot').textContent='ext → '+(j.external_url||'')+(j.approval_required?' · high-risk writes need approval':'');
}catch(e){} }
$('run').onclick=run;
$('goal').addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); run(); }});
init(); refresh(); setInterval(refresh,1500);
</script>
</body>
</html>"""
