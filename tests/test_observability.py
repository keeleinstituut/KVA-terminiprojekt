"""Tests for utils/observability.py and backend/telemetry.py wiring.

Run: python -m pytest tests/
"""
import io
import json
import logging
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import observability as obs  # noqa: E402

obs.setup("test")


@pytest.fixture
def lines():
    """Capture JSON log lines emitted during a test."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(obs.JsonFormatter())
    root = logging.getLogger()
    root.addHandler(handler)

    def read():
        return [json.loads(l) for l in buf.getvalue().splitlines() if l.strip()]

    yield read
    root.removeHandler(handler)


def _events(lines, name):
    return [l for l in lines() if l.get("event") == name]


def test_event_fields_and_error(lines):
    with obs.event("unit.ok", a=1) as ev:
        ev.add(nested={"x": 1})
        obs.add(nested={"y": 2})
    with pytest.raises(ValueError):
        with obs.event("unit.fail"):
            raise ValueError("boom")

    ok = _events(lines, "unit.ok")[0]
    assert ok["a"] == 1 and ok["nested"] == {"x": 1, "y": 2}
    assert ok["outcome"] == "success" and ok["level"] == "info"
    assert ok["service"] == "test" and len(ok["trace_id"]) == 32 and "duration_ms" in ok

    fail = _events(lines, "unit.fail")[0]
    assert fail["outcome"] == "error" and fail["level"] == "error"
    assert fail["error"]["type"] == "ValueError" and "boom" in fail["error"]["stack"]


def test_warning_attaches_to_event_and_thread_context(lines):
    def worker():
        obs.add(from_thread=True)
        logging.getLogger("backend").warning("something odd")

    with obs.event("unit.thread"):
        t = threading.Thread(target=obs.run_in_context(worker))
        t.start()
        t.join()

    ev = _events(lines, "unit.thread")[0]
    assert ev["from_thread"] is True
    assert ev["logs"][0]["msg"] == "something odd"


def _app():
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import StreamingResponse

    from backend import telemetry

    app = FastAPI()

    @app.post("/up")
    async def up(file: UploadFile = File(...), title: str = Form("")):
        obs.add(upload={"filename": file.filename})
        return {"ok": True}

    @app.get("/stream")
    async def stream():
        def gen():
            obs.add(streamed=True)  # runs in a threadpool
            yield "data: 1\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaboom")

    telemetry.install(app)
    return app


def test_backend_request_events(lines):
    from fastapi.testclient import TestClient

    client = TestClient(_app(), raise_server_exceptions=False)
    headers = {"X-KVA-User": "admin", "X-KVA-Session": "sess-1"}

    assert client.post("/up", files={"file": ("a.pdf", b"%PDF", "application/pdf")}, headers=headers).status_code == 200
    assert client.post("/up", files={"file": (None, b"%PDF", "application/pdf")}, headers=headers).status_code == 422
    assert client.post("/up", files={"file": (None, b"x" * 1_100_000, "application/pdf")}).status_code == 400
    assert client.get("/stream").status_code == 200
    assert client.get("/boom").status_code == 500

    reqs = {(e["http"]["path"], e["http"]["status"]): e for e in _events(lines, "http.request")}
    ok = reqs[("/up", 200)]
    assert ok["user"] == "admin" and ok["session_id"] == "sess-1"
    assert ok["upload"]["filename"] == "a.pdf" and ok["http"]["route"] == "/up"
    assert len(ok["trace_id"]) == 32

    v = reqs[("/up", 422)]
    assert v["outcome"] == "rejected" and v["rejection"]["errors"][0]["loc"] == "body.file"
    assert "input" not in v["rejection"]["errors"][0]

    big = reqs[("/up", 400)]
    assert "Part exceeded maximum size" in big["rejection"]["detail"]

    assert reqs[("/stream", 200)]["streamed"] is True
    boom = reqs[("/boom", 500)]
    assert boom["outcome"] == "error" and boom["error"]["type"] == "RuntimeError"


def test_trace_propagates_from_client_event_to_backend(lines):
    """A frontend event's trace id must reach the backend via traceparent."""
    import httpx
    from fastapi.testclient import TestClient

    client = TestClient(_app())
    with obs.event("ui.action") as ev:
        # TestClient is an httpx.Client, so the httpx instrumentation applies.
        client.post("/up", files={"file": ("b.pdf", b"%PDF", "application/pdf")})

    ui = _events(lines, "ui.action")[0]
    backend = [e for e in _events(lines, "http.request") if e.get("upload", {}).get("filename") == "b.pdf"][0]
    assert ui["trace_id"] == backend["trace_id"]


def test_telemetry_endpoint(lines):
    from fastapi.testclient import TestClient

    client = TestClient(_app())
    body = json.dumps({"events": [
        {"type": "ws_close", "session_id": "s1", "user": "admin", "details": {"code": 1006}},
        {"type": "js_error", "details": {"message": "x" * 5000}},
    ]})
    r = client.post("/telemetry", content=body, headers={"Content-Type": "text/plain"})
    assert r.status_code == 204
    assert client.post("/telemetry", content="not json").status_code == 400
    assert client.post("/telemetry", content=json.dumps({"events": [{"type": "Bad Type!"}]})).status_code == 400

    ws = _events(lines, "browser.ws_close")[0]
    assert ws["session_id"] == "s1" and ws["details"] == {"code": 1006} and ws["source"] == "browser"
    err = _events(lines, "browser.js_error")[0]
    assert err["outcome"] == "error" and err["details"]["truncated"] is True
    # successful telemetry posts themselves are sampled out
    assert not [e for e in _events(lines, "http.request") if e["http"]["path"] == "/telemetry" and e["http"]["status"] == 204]
