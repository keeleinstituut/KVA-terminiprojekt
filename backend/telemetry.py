"""
Backend observability wiring: one wide event per HTTP request, rejection
reasons for 4xx responses, and the /telemetry endpoint for browser events.
"""
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from utils import observability as obs

# Requests whose successful responses are not worth a log line.
_SAMPLED_OUT_PATHS = {"/health", "/telemetry"}

MAX_TELEMETRY_BODY = 32 * 1024
MAX_DETAILS_CHARS = 2000


def _header(headers: Dict[str, str], name: str) -> Optional[str]:
    value = headers.get(name)
    return obs.trunc(value, 100) if value else None


class WideEventMiddleware:
    """Pure ASGI middleware: emits one `http.request` event per request.

    Runs inside the OpenTelemetry server span, so the event carries its
    trace_id. Streaming responses (SSE upload) are covered because the event
    is emitted only after the last body chunk has been sent.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("utf-8", "replace") for k, v in scope["headers"]}
        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        fields: Dict[str, Any] = {
            "http": {"method": scope["method"], "path": scope["path"]},
        }
        for key, header in (("user", "x-kva-user"), ("session_id", "x-kva-session"),
                            ("client_version", "x-kva-client-version")):
            value = _header(headers, header)
            if value:
                fields[key] = value
        if "x-forwarded-for" in headers:
            fields["via_proxy"] = True

        with obs.event("http.request", span=False, **fields) as ev:
            try:
                await self.app(scope, receive, send_wrapper)
            finally:
                route = scope.get("route")
                code = status["code"]
                ev.add(http={"status": code, "route": getattr(route, "path", scope["path"])})
                if ev.outcome is None:
                    if code >= 500:
                        ev.set_outcome("error")
                    elif code >= 400:
                        ev.set_outcome("rejected")
                if scope["path"] in _SAMPLED_OUT_PATHS and code < 400 and ev.outcome in (None, "success"):
                    ev.discard()


def _validation_errors(exc: RequestValidationError) -> List[Dict[str, Any]]:
    # Never copy "input": for uploads it is the whole file content.
    return [
        {"loc": ".".join(str(p) for p in e.get("loc", ())), "type": e.get("type"),
         "msg": obs.trunc(e.get("msg", ""), 200)}
        for e in exc.errors()[:10]
    ]


async def _on_validation_error(request: Request, exc: RequestValidationError):
    obs.add(rejection={"kind": "validation", "errors": _validation_errors(exc)})
    return await request_validation_exception_handler(request, exc)


async def _on_http_error(request: Request, exc: StarletteHTTPException):
    obs.add(rejection={"kind": "http", "status": exc.status_code,
                       "detail": obs.trunc(str(exc.detail), 500)})
    return await http_exception_handler(request, exc)


# ---------------------------------------------------------------- browser telemetry

class BrowserEvent(BaseModel):
    type: str = Field(..., max_length=64, pattern=r"^[a-z0-9_.]+$")
    ts: Optional[str] = Field(None, max_length=40)
    session_id: Optional[str] = Field(None, max_length=100)
    user: Optional[str] = Field(None, max_length=100)
    page: Optional[str] = Field(None, max_length=300)
    details: Dict[str, Any] = Field(default_factory=dict)


class BrowserEventBatch(BaseModel):
    events: List[BrowserEvent] = Field(..., max_length=20)


router = APIRouter()


@router.post("/telemetry", status_code=204, tags=["Telemetry"])
async def telemetry(request: Request):
    """Receive browser-side events (sent with navigator.sendBeacon).

    sendBeacon posts text/plain, so the body is parsed by hand.
    """
    body = await request.body()
    if len(body) > MAX_TELEMETRY_BODY:
        obs.add(rejection={"kind": "telemetry", "detail": "body too large", "bytes": len(body)})
        return Response(status_code=413)
    try:
        batch = BrowserEventBatch.model_validate(json.loads(body))
    except (ValueError, ValidationError) as exc:
        obs.add(rejection={"kind": "telemetry", "detail": obs.trunc(str(exc), 300)})
        return Response(status_code=400)

    for e in batch.events:
        details = e.details
        if len(json.dumps(details, default=str)) > MAX_DETAILS_CHARS:
            details = {"truncated": True, "keys": list(details)[:20]}
        obs.log_event(
            f"browser.{e.type}",
            outcome="error" if e.type in ("js_error", "unhandled_rejection") else "success",
            source="browser",
            browser_ts=e.ts,
            session_id=e.session_id,
            user=e.user,
            page=e.page,
            details=details,
        )
    return Response(status_code=204)


def install(app: FastAPI) -> None:
    """Attach tracing, the wide-event middleware, handlers and /telemetry."""
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    app.add_middleware(WideEventMiddleware)
    app.add_exception_handler(RequestValidationError, _on_validation_error)
    app.add_exception_handler(StarletteHTTPException, _on_http_error)
    app.include_router(router)
    # Wraps the whole middleware stack, so the server span is outermost and
    # WideEventMiddleware sees its trace context.
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")
