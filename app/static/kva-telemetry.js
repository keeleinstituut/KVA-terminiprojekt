/*
 * KVA browser telemetry.
 *
 * Loaded in <head> before Bokeh connects. Reports what the server cannot
 * see: websocket drops (after which Bokeh does NOT reconnect and the page
 * silently stops working), JS errors, clicks on disabled buttons and files
 * the user picked. Events go to <prefix>/api/telemetry (nginx -> backend)
 * with navigator.sendBeacon, so they arrive even when the websocket is dead.
 *
 * Events carry the Bokeh session id (read from the websocket token), which
 * joins them to the server-side session.* / ui events.
 */
(function () {
  "use strict";
  if (window.__kvaTelemetry) return;

  var MAX_EVENTS = 300;      // per page load
  var FLUSH_MS = 5000;
  var prefix = location.pathname.replace(/[^/]*$/, "");   // "/" or "/siseturvalisus/"
  var endpoint = prefix + "api/telemetry";
  var pageStart = Date.now();
  var queue = [];
  var sent = 0;
  var sessionId = null;

  function now() { return new Date().toISOString(); }
  function clip(v, n) { v = String(v == null ? "" : v); return v.length > n ? v.slice(0, n) + "…" : v; }

  function record(type, details) {
    if (sent + queue.length >= MAX_EVENTS) return;
    queue.push({ type: type, ts: now(), session_id: sessionId, page: clip(location.pathname, 300),
                 details: details || {} });
    if (queue.length >= 10) flush();
  }

  function flush() {
    if (!queue.length) return;
    var batch = queue.splice(0, 20);
    batch.forEach(function (e) { if (!e.session_id) e.session_id = sessionId; });
    sent += batch.length;
    var body = JSON.stringify({ events: batch });
    try {
      if (!(navigator.sendBeacon && navigator.sendBeacon(endpoint, new Blob([body], { type: "text/plain" })))) {
        fetch(endpoint, { method: "POST", body: body, keepalive: true, headers: { "Content-Type": "text/plain" } })
          .catch(function () {});
      }
    } catch (e) { /* never break the page */ }
    if (queue.length) flush();
  }

  function sessionFromToken(token) {
    try {
      var p = String(token).split(".")[0].replace(/-/g, "+").replace(/_/g, "/");
      while (p.length % 4) p += "=";
      return JSON.parse(atob(p)).session_id || null;
    } catch (e) { return null; }
  }

  // --- websocket lifecycle (Bokeh: new WebSocket(url, ["bokeh", token])) ---
  var NativeWS = window.WebSocket;
  if (NativeWS) {
    var Wrapped = function (url, protocols) {
      var ws = protocols === undefined ? new NativeWS(url) : new NativeWS(url, protocols);
      if (Array.isArray(protocols) && protocols[0] === "bokeh") {
        var sid = sessionFromToken(protocols[1]);
        if (sid) sessionId = sid;
        var opened = null;
        ws.addEventListener("open", function () {
          opened = Date.now();
          record("ws_open", { connect_ms: opened - pageStart });
        });
        ws.addEventListener("close", function (ev) {
          record("ws_close", {
            code: ev.code, reason: clip(ev.reason, 200), clean: ev.wasClean,
            open_ms: opened ? Date.now() - opened : null, online: navigator.onLine,
            visible: document.visibilityState
          });
          flush();
        });
        ws.addEventListener("error", function () { record("ws_error", { online: navigator.onLine }); });
      }
      return ws;
    };
    Wrapped.prototype = NativeWS.prototype;
    ["CONNECTING", "OPEN", "CLOSING", "CLOSED"].forEach(function (k) { Wrapped[k] = NativeWS[k]; });
    window.WebSocket = Wrapped;
  }

  // --- errors ---
  window.addEventListener("error", function (ev) {
    record("js_error", { message: clip(ev.message, 500), source: clip(ev.filename, 200),
                         line: ev.lineno, col: ev.colno,
                         stack: clip(ev.error && ev.error.stack, 1500) });
  });
  window.addEventListener("unhandledrejection", function (ev) {
    var r = ev.reason;
    record("unhandled_rejection", { message: clip(r && r.message ? r.message : r, 500),
                                    stack: clip(r && r.stack, 1500) });
  });

  // --- clicks on disabled buttons (pointer events cross shadow DOM) ---
  function buttonLabel(el) { return clip((el.innerText || el.textContent || el.title || "").trim(), 80); }
  document.addEventListener("pointerdown", function (ev) {
    var path = ev.composedPath ? ev.composedPath() : [];
    for (var i = 0; i < path.length && i < 15; i++) {
      var el = path[i];
      if (el && el.tagName === "BUTTON") {
        if (el.disabled) record("disabled_click", { label: buttonLabel(el) });
        return;
      }
    }
  }, true);

  // --- files picked (inputs live in shadow roots; change does not bubble out) ---
  function eachFileInput(root, fn) {
    var all = root.querySelectorAll("*");
    for (var i = 0; i < all.length; i++) {
      var el = all[i];
      if (el.tagName === "INPUT" && el.type === "file") fn(el);
      if (el.shadowRoot) eachFileInput(el.shadowRoot, fn);
    }
  }
  function watchFileInputs() {
    eachFileInput(document, function (input) {
      if (input.__kvaWatched) return;
      input.__kvaWatched = true;
      input.addEventListener("change", function () {
        var f = input.files && input.files[0];
        record("file_chosen", f ? { name: clip(f.name, 200), size: f.size, type: f.type }
                                : { cleared: true });
      });
    });
  }
  setInterval(watchFileInputs, 3000);

  // --- page lifecycle ---
  record("page_load", { ua: clip(navigator.userAgent, 200), lang: navigator.language,
                        viewport: window.innerWidth + "x" + window.innerHeight });
  window.addEventListener("online", function () { record("online", {}); });
  window.addEventListener("offline", function () { record("offline", {}); });
  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden") flush();
  });
  window.addEventListener("pagehide", function () {
    record("page_hide", { page_ms: Date.now() - pageStart });
    flush();
  });
  setInterval(flush, FLUSH_MS);

  window.__kvaTelemetry = { record: record, flush: flush };
})();
