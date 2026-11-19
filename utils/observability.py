"""
Structured logging, wide events and OpenTelemetry tracing shared by the
frontend (Panel) and backend (FastAPI).

Every log line is a single JSON object on stdout. The main unit is the *wide
event*: one context-rich line per request / user action, built up while the
work runs and emitted once at the end:

    with obs.event("upload.submit", filename=name) as ev:
        ...
        ev.add(pages=12, chunks=40)

Each line carries service, deployment, version and the current OTel trace id,
so frontend and backend lines of the same action can be joined on trace_id.

Spans are only exported when OTEL_EXPORTER_OTLP_ENDPOINT is set; otherwise
trace ids are still generated and used for correlation in the logs.
"""
from __future__ import annotations

import contextvars
import json
import threading
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

SERVICE = "unknown"
DEPLOYMENT = os.getenv("SERVICE_NAME", "")
VERSION = os.getenv("GIT_COMMIT", "dev")

_EVENT_LOGGER = logging.getLogger("event")
_current_event: contextvars.ContextVar[Optional["WideEvent"]] = contextvars.ContextVar(
    "kva_current_event", default=None
)
_configured = False

# Warnings/errors logged while an event is active are also attached to it.
_MAX_ATTACHED_LOGS = 20


def trunc(value: Any, limit: int = 200) -> Any:
    """Shorten long strings so events stay readable."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"…(+{len(value) - limit})"
    return value


def _trace_ids() -> Dict[str, str]:
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return {}
    return {"trace_id": format(ctx.trace_id, "032x"), "span_id": format(ctx.span_id, "016x")}


def _deep_merge(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


class WideEvent:
    """One context-rich log line, emitted once when the unit of work ends."""

    def __init__(self, name: str, **fields: Any):
        self.name = name
        self.fields: Dict[str, Any] = {}
        self.outcome: Optional[str] = None
        self.level = logging.INFO
        self._start = time.perf_counter()
        self._emitted = False
        self._lock = threading.Lock()
        self._ids = _trace_ids()
        self.add(**fields)

    def add(self, **fields: Any) -> "WideEvent":
        # Events are filled from worker threads (parallel LLM calls, uploads).
        with self._lock:
            _deep_merge(self.fields, fields)
        return self

    def incr(self, key: str, by: int = 1) -> None:
        with self._lock:
            self.fields[key] = self.fields.get(key, 0) + by

    def set_outcome(self, outcome: str) -> None:
        self.outcome = outcome
        if outcome == "error":
            self.level = logging.ERROR

    def set_error(self, exc: Optional[BaseException] = None, reason: Optional[str] = None,
                  outcome: str = "error") -> None:
        err: Dict[str, Any] = {}
        if exc is not None:
            err["type"] = type(exc).__name__
            err["message"] = trunc(str(exc), 1000)
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            err["stack"] = trunc(tb, 4000)
        if reason:
            err["reason"] = trunc(reason, 1000)
        self.fields["error"] = err
        self.set_outcome(outcome)
        self.level = logging.ERROR

    def discard(self) -> None:
        """Drop this event (e.g. sampled-out health checks)."""
        self._emitted = True

    def attach_log(self, record: logging.LogRecord) -> None:
        logs = self.fields.setdefault("logs", [])
        if len(logs) < _MAX_ATTACHED_LOGS:
            logs.append({"level": record.levelname.lower(), "logger": record.name,
                         "msg": trunc(record.getMessage(), 500)})

    def emit(self) -> None:
        if self._emitted:
            return
        self._emitted = True
        payload: Dict[str, Any] = {
            "duration_ms": round((time.perf_counter() - self._start) * 1000, 1),
            "outcome": self.outcome or "success",
        }
        payload.update(self.fields)
        _EVENT_LOGGER.log(self.level, self.name,
                          extra={"wide_event": payload, "event_ids": self._ids})


def current() -> Optional[WideEvent]:
    return _current_event.get()


def add(**fields: Any) -> None:
    """Add fields to the active wide event (no-op when there is none)."""
    ev = _current_event.get()
    if ev is not None:
        ev.add(**fields)


def incr(key: str, by: int = 1) -> None:
    ev = _current_event.get()
    if ev is not None:
        ev.incr(key, by)


def set_outcome(outcome: str) -> None:
    ev = _current_event.get()
    if ev is not None:
        ev.set_outcome(outcome)


def set_error(exc: Optional[BaseException] = None, reason: Optional[str] = None,
              outcome: str = "error") -> None:
    ev = _current_event.get()
    if ev is not None:
        ev.set_error(exc, reason, outcome)


def tracer() -> trace.Tracer:
    return trace.get_tracer("kva")


@contextmanager
def event(name: str, *, span: bool = True, **fields: Any) -> Iterator[WideEvent]:
    """Start a span + wide event; emit it when the block ends.

    An exception escaping the block marks the event as an error and re-raises.
    """
    span_cm = tracer().start_as_current_span(name) if span else None
    if span_cm is not None:
        span_cm.__enter__()
    ev = WideEvent(name, **fields)
    token = _current_event.set(ev)
    try:
        yield ev
    except BaseException as exc:  # noqa: BLE001 - re-raised
        if "error" not in ev.fields:
            ev.set_error(exc, outcome=ev.outcome if ev.outcome not in (None, "success") else "error")
        raise
    finally:
        _current_event.reset(token)
        ev.emit()
        if span_cm is not None:
            span_cm.__exit__(None, None, None)


@contextmanager
def use_event(ev: WideEvent) -> Iterator[WideEvent]:
    """Make an existing event current (e.g. inside a worker thread)."""
    token = _current_event.set(ev)
    try:
        yield ev
    finally:
        _current_event.reset(token)


def log_event(name: str, outcome: str = "success", **fields: Any) -> None:
    """Emit a one-shot event (no duration of its own)."""
    ev = WideEvent(name, **fields)
    ev.set_outcome(outcome)
    ev.emit()


def run_in_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap fn so it runs with the caller's contextvars (trace + wide event).

    Use for threading.Thread targets and executor submissions:
        threading.Thread(target=obs.run_in_context(worker))
        loop.run_in_executor(pool, obs.run_in_context(fn))
    """
    ctx = contextvars.copy_context()

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return ctx.copy().run(fn, *args, **kwargs)

    return wrapper


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "service": SERVICE,
        }
        if DEPLOYMENT:
            data["deployment"] = DEPLOYMENT
        data["version"] = VERSION

        wide = getattr(record, "wide_event", None)
        if wide is not None:
            data["event"] = record.getMessage()
            data.update(getattr(record, "event_ids", None) or _trace_ids())
            data.update(wide)
        else:
            data["event"] = "log"
            data["logger"] = record.name
            data["msg"] = record.getMessage()
            data.update(_trace_ids())
            if record.exc_info:
                data["stack"] = trunc(self.formatException(record.exc_info), 4000)
        return json.dumps(data, ensure_ascii=False, default=str)


class _AttachToEventHandler(logging.Handler):
    """Copies warnings/errors into the active wide event, if any."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == "event":
            return
        ev = _current_event.get()
        if ev is not None:
            ev.attach_log(record)


def setup(service: str, quiet_loggers: Optional[Dict[str, int]] = None) -> None:
    """Configure JSON logging and tracing for this process. Idempotent."""
    global SERVICE, DEPLOYMENT, VERSION, _configured
    SERVICE = service
    DEPLOYMENT = os.getenv("SERVICE_NAME", DEPLOYMENT)
    VERSION = os.getenv("GIT_COMMIT", VERSION)
    if _configured:
        return
    _configured = True

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(JsonFormatter())
    attach = _AttachToEventHandler(level=logging.WARNING)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(stream)
    root.addHandler(attach)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

    # Route third-party loggers that install their own handlers through root.
    levels = {
        "uvicorn": logging.INFO,
        "uvicorn.error": logging.INFO,
        "uvicorn.access": logging.WARNING,  # replaced by http.request events
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "urllib3": logging.WARNING,
        "tornado.access": logging.ERROR,     # 404 scanner noise; requests are UI events
        "bokeh": logging.WARNING,
        "event": logging.INFO,
    }
    levels.update(quiet_loggers or {})
    for name, level in levels.items():
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.propagate = True
        lg.setLevel(level)

    resource = Resource.create({
        "service.name": f"kva-{service}",
        "service.version": VERSION,
        "deployment.environment": DEPLOYMENT or "unknown",
    })
    provider = TracerProvider(resource=resource)
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
    except Exception as exc:  # pragma: no cover - instrumentation is optional
        logging.getLogger("observability").warning(f"httpx instrumentation unavailable: {exc}")

    log_event("process.start", python=sys.version.split()[0],
              otlp_export=bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")))
