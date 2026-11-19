"""
Frontend observability helpers for the Panel app.

- ui_event(): a wide event for one user action, tagged with the Panel user
  and Bokeh session id. Backend calls made inside it share its trace_id.
- backend_client(): the one httpx client factory for talking to the backend;
  it forwards user/session so backend events can be joined to UI events.
- session lifecycle events (session.start / session.end).

Browser-side events (websocket drops, JS errors, disabled clicks, file picks)
come from static/kva-telemetry.js via nginx -> backend /telemetry.
"""
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import httpx
import panel as pn

from utils import observability as obs

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
CLIENT_VERSION = os.getenv("GIT_COMMIT", "dev")


def session_fields() -> Dict[str, Any]:
    """User and Bokeh session id of the session currently being served."""
    fields: Dict[str, Any] = {}
    try:
        user = pn.state.user
        if user:
            fields["user"] = user
        doc = pn.state.curdoc
        if doc is not None and doc.session_context is not None:
            fields["session_id"] = doc.session_context.id
    except Exception:
        pass
    return fields


@contextmanager
def ui_event(name: str, **fields: Any) -> Iterator[obs.WideEvent]:
    """Wide event + span for one UI action, tagged with user/session."""
    with obs.event(name, **session_fields(), **fields) as ev:
        yield ev


def ui_log(name: str, outcome: str = "success", **fields: Any) -> None:
    """One-shot UI event (no duration), tagged with user/session."""
    obs.log_event(name, outcome=outcome, **session_fields(), **fields)


def _identity_headers() -> Dict[str, str]:
    # Prefer the active event: it captured user/session before any await,
    # when pn.state still pointed at the right session.
    ev = obs.current()
    source = ev.fields if ev is not None else session_fields()
    headers = {"X-KVA-Client-Version": CLIENT_VERSION}
    if source.get("user"):
        headers["X-KVA-User"] = str(source["user"])
    if source.get("session_id"):
        headers["X-KVA-Session"] = str(source["session_id"])
    return headers


def backend_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """Async client for the backend. Use as `async with backend_client() as c:`."""
    return httpx.AsyncClient(base_url=BACKEND_URL, timeout=timeout, headers=_identity_headers())


def backend_client_sync(timeout: float = 30.0) -> httpx.Client:
    """Sync variant, only for code that cannot be async."""
    return httpx.Client(base_url=BACKEND_URL, timeout=timeout, headers=_identity_headers())


def describe_http_error(exc: Exception) -> Dict[str, Any]:
    """Compact, loggable description of a failed backend call."""
    info: Dict[str, Any] = {"type": type(exc).__name__, "message": obs.trunc(str(exc), 300)}
    if isinstance(exc, httpx.HTTPStatusError):
        info["status"] = exc.response.status_code
        info["body"] = obs.trunc(exc.response.text, 300)
    return info


# ---------------------------------------------------------------- sessions

_session_started: Dict[str, float] = {}


def start_session(guest_mode: bool) -> None:
    """Call once per session from main.py."""
    fields = session_fields()
    sid = fields.get("session_id")
    if sid:
        _session_started[sid] = time.time()
    obs.log_event("session.start", guest=guest_mode, **fields)
    try:
        pn.state.on_session_destroyed(_end_session)
    except Exception as exc:
        obs.log_event("session.hook_failed", outcome="error", error={"message": str(exc)}, **fields)


def _end_session(session_context) -> None:
    sid: Optional[str] = getattr(session_context, "id", None)
    started = _session_started.pop(sid, None) if sid else None
    obs.log_event(
        "session.end",
        session_id=sid,
        session_seconds=round(time.time() - started, 1) if started else None,
    )
