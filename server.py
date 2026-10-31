#!/usr/bin/env python3
"""A2A 1.0 (HTTP+JSON) invoice action agent. Standard library only.

Layers:  protocol (this file)  ->  store.py (SQLite)  ->  decide.py (AI)
Run ONE process (locks are in-process).  See README.md for configuration.
"""
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

from decide import Decider, canonical, llm_from_env
from store import Store

log = logging.getLogger("a2a")

A2A_JSON = "application/a2a+json"
MT_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
MT_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
MT_RESULTS = "application/vnd.ga5.invoice-action-results+json"
MT_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"

SUBMITTED = "TASK_STATE_SUBMITTED"
WORKING = "TASK_STATE_WORKING"
INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
COMPLETED = "TASK_STATE_COMPLETED"
CANCELED = "TASK_STATE_CANCELED"
FAILED = "TASK_STATE_FAILED"
REJECTED = "TASK_STATE_REJECTED"
TERMINAL = {COMPLETED, CANCELED, FAILED, REJECTED}

MAX_BODY = 16 * 1024 * 1024
MAX_RESPONSE = 500 * 1024  # contract limit is 512 KiB
REQUEST_BUDGET = 38.0  # seconds of the 45s per-request limit we allow ourselves


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class ApiError(Exception):
    def __init__(self, http, status, reason, message):
        super().__init__(message)
        self.http, self.status, self.reason, self.message = http, status, reason, message


def bad(msg):
    return ApiError(400, "INVALID_ARGUMENT", "INVALID_ARGUMENT", msg)


def not_found():
    # identical for "does not exist" and "belongs to someone else"
    return ApiError(404, "NOT_FOUND", "TASK_NOT_FOUND", "Task not found.")


class KeyedLocks:
    def __init__(self):
        self._m = threading.Lock()
        self._d = {}

    @contextmanager
    def hold(self, key):
        with self._m:
            ent = self._d.get(key)
            if ent is None:
                ent = self._d[key] = [threading.Lock(), 0]
            ent[1] += 1
        ent[0].acquire()
        try:
            yield
        finally:
            ent[0].release()
            with self._m:
                ent[1] -= 1
                if ent[1] == 0:
                    self._d.pop(key, None)


# ----------------------------------------------------------------------------
# application logic
# ----------------------------------------------------------------------------
class App:
    def __init__(self, store, decider, base_url=None):
        self.store, self.decider = store, decider
        self.base_url = base_url
        self.prefix = ""
        if base_url:
            self.prefix = urlsplit(base_url).path.rstrip("/")
        self.locks = KeyedLocks()

    # ---- agent card ----------------------------------------------------
    def agent_card(self, fallback_base):
        base = self.base_url or fallback_base
        return {
            "name": "Invoice Action Agent",
            "description": "Reads invoice packages, chooses one evidence-backed business action per invoice "
                           "(settle, request approval, hold, reject duplicate, open exception) and only "
                           "executes proposals the grader accepts.",
            "version": "1.0.0",
            "supportedInterfaces": [{"url": base, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
            "capabilities": {"streaming": False, "pushNotifications": False, "extendedAgentCard": False},
            "securitySchemes": {"bearerAuth": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}},
            "securityRequirements": [{"schemes": {"bearerAuth": {"list": []}}}],
            "defaultInputModes": [MT_BATCH],
            "defaultOutputModes": [MT_PROPOSALS, MT_RECEIPTS],
            "skills": [{
                "id": "invoice_action_agent",
                "name": "Invoice action decisions",
                "description": "Classifies each invoice package into exactly one typed business action with the "
                               "decisive evidence references, then records receipts for accepted actions.",
                "tags": ["invoice", "accounts-payable", "approvals", "duplicate-detection"],
                "examples": ["Decide the action for each package in an invoice claim batch"],
                "inputModes": [MT_BATCH, MT_RESULTS],
                "outputModes": [MT_PROPOSALS, MT_RECEIPTS],
            }],
        }

    # ---- rendering -------------------------------------------------------
    @staticmethod
    def render(task, history_length=None, include_artifacts=True):
        t = dict(task)
        if not include_artifacts:
            t.pop("artifacts", None)
        hist = t.get("history")
        if hist is not None:
            if history_length is not None:
                if history_length <= 0:
                    t.pop("history", None)
                else:
                    t["history"] = hist[-history_length:]
        return t

    # ---- message:send ----------------------------------------------------
    def send_message(self, principal, body):
        started = time.monotonic()
        if not isinstance(body, dict) or not isinstance(body.get("message"), dict):
            raise bad("message is required")
        msg = body["message"]
        mid = msg.get("messageId")
        if not isinstance(mid, str) or not mid.strip() or len(mid) > 256:
            raise bad("message.messageId is required")
        if msg.get("role") != "ROLE_USER":
            raise bad("message.role must be ROLE_USER")
        parts = msg.get("parts")
        if not isinstance(parts, list) or not parts or not all(isinstance(p, dict) for p in parts):
            raise bad("message.parts must be a non-empty array")
        for f in ("taskId", "contextId"):
            if f in msg and msg[f] is not None and not (isinstance(msg[f], str) and msg[f]):
                raise bad(f"message.{f} must be a string")
        cfg = body.get("configuration")
        if cfg is not None and not isinstance(cfg, dict):
            raise bad("configuration must be an object")
        hl = (cfg or {}).get("historyLength")
        if hl is not None and (isinstance(hl, bool) or not isinstance(hl, int) or hl < 0):
            raise bad("configuration.historyLength must be a non-negative integer")

        # Hash of the message only (configuration is ignored), keys sorted recursively.
        mhash = hashlib.sha256(canonical(msg).encode("utf-8")).hexdigest()
        deadline = started + REQUEST_BUDGET

        with self.locks.hold((principal, mid)):
            rec = self.store.get_message(principal, mid)
            if rec:
                if rec["hash"] != mhash:
                    raise ApiError(409, "IDEMPOTENCY_CONFLICT", "IDEMPOTENCY_CONFLICT",
                                   "IDEMPOTENCY_CONFLICT: this messageId was already used with different content.")
                task = self.store.get_task(principal, rec["task_id"])
                if task is None:
                    raise ApiError(500, "INTERNAL", "INTERNAL", "Internal error.")
                if task["status"]["state"] in (SUBMITTED, WORKING):  # crash recovery
                    task = self._finish_decision(principal, task, deadline)
                return {"task": self.render(task, hl)}
            if msg.get("taskId"):
                task = self._continue_task(principal, msg, mhash)
            else:
                task = self._start_task(principal, msg, mhash, deadline)
            return {"task": self.render(task, hl)}

    @staticmethod
    def _single_part(msg, media_type):
        parts = msg["parts"]
        for p in parts:
            mt = p.get("mediaType")
            if mt not in (MT_BATCH, MT_RESULTS):
                raise ApiError(415, "INVALID_ARGUMENT", "CONTENT_TYPE_NOT_SUPPORTED",
                               "Unsupported media type in message parts.")
        if len(parts) != 1 or parts[0].get("mediaType") != media_type:
            raise bad(f"message must contain exactly one {media_type} part")
        data = parts[0].get("data")
        if not isinstance(data, dict):
            raise bad("part.data must be a JSON object")
        return data

    @staticmethod
    def _validate_batch(data):
        bid = data.get("batchId")
        if not isinstance(bid, str) or not bid:
            raise bad("batchId is required")
        pkgs = data.get("packages")
        if not isinstance(pkgs, list) or not pkgs or len(pkgs) > 500:
            raise bad("packages must be a non-empty array")
        seen = set()
        for p in pkgs:
            if not isinstance(p, dict) or not isinstance(p.get("packageId"), str) or not p["packageId"]:
                raise bad("every package needs a packageId")
            if p["packageId"] in seen:
                raise bad("duplicate packageId in batch")
            seen.add(p["packageId"])
        return data

    def _start_task(self, principal, msg, mhash, deadline):
        batch = self._validate_batch(self._single_part(msg, MT_BATCH))
        task_id = str(uuid.uuid4())
        ctx = msg.get("contextId") or str(uuid.uuid4())
        stored = dict(msg)
        stored["taskId"], stored["contextId"] = task_id, ctx
        task = {
            "id": task_id,
            "contextId": ctx,
            "status": {"state": SUBMITTED, "timestamp": now_iso()},
            "history": [stored],
        }
        with self.store.tx():  # state is durable before any model work or response
            self.store.put_task(principal, task)
            self.store.put_message(principal, msg["messageId"], mhash, task_id)
        return self._finish_decision(principal, task, deadline)

    def _finish_decision(self, principal, task, deadline):
        batch = None
        for p in task["history"][0]["parts"]:
            if p.get("mediaType") == MT_BATCH:
                batch = p["data"]
        decisions = self.decider.decide(batch, deadline)
        proposals = []
        for pkg, d in zip(batch["packages"], decisions):
            proposals.append({
                "packageId": pkg["packageId"],
                "actionId": "act-" + uuid.uuid4().hex,
                "action": d["action"],
                "facts": d["facts"],
                "evidenceRefs": d["evidenceRefs"],
                "rationale": d["rationale"],
            })
        with self.store.tx():
            cur = self.store.get_task(principal, task["id"])
            if cur["status"]["state"] not in (SUBMITTED, WORKING):
                return cur  # e.g. canceled while the model was thinking
            cur["artifacts"] = [{
                "artifactId": "proposals-" + cur["id"],
                "name": "invoice-action-proposals",
                "parts": [{"mediaType": MT_PROPOSALS, "data": {"batchId": batch["batchId"], "proposals": proposals}}],
            }]
            cur["status"] = {
                "state": INPUT_REQUIRED,
                "timestamp": now_iso(),
                "message": {
                    "messageId": str(uuid.uuid4()), "role": "ROLE_AGENT",
                    "taskId": cur["id"], "contextId": cur["contextId"],
                    "parts": [{"text": "Proposals ready. Waiting for the grader's results before acting."}],
                },
            }
            self.store.put_task(principal, cur)
        return cur

    def _continue_task(self, principal, msg, mhash):
        tid = msg["taskId"]
        data = self._single_part(msg, MT_RESULTS)
        with self.store.tx():  # state check + change are one atomic step (cancel race)
            task = self.store.get_task(principal, tid)
            if task is None:
                raise not_found()
            state = task["status"]["state"]
            if state in TERMINAL:
                raise ApiError(409, "FAILED_PRECONDITION", "TASK_TERMINAL", "Task is already in a final state.")
            if state != INPUT_REQUIRED:
                raise ApiError(409, "FAILED_PRECONDITION", "TASK_NOT_READY", "Task is not waiting for results.")
            if msg.get("contextId") not in (None, task["contextId"]):
                raise bad("contextId does not match the task")
            prop = None
            for a in task.get("artifacts", []):
                for p in a["parts"]:
                    if p.get("mediaType") == MT_PROPOSALS:
                        prop = p["data"]
            if prop is None:
                raise ApiError(409, "FAILED_PRECONDITION", "TASK_NOT_READY", "Task has no proposals.")
            if data.get("batchId") != prop["batchId"]:
                raise bad("batchId does not match the task")
            results = data.get("results")
            if not isinstance(results, list):
                raise bad("results must be an array")
            by_pkg = {p["packageId"]: p for p in prop["proposals"]}
            seen, executions = set(), []
            for r in results:
                if not isinstance(r, dict):
                    raise bad("each result must be an object")
                pid = r.get("packageId")
                p = by_pkg.get(pid) if isinstance(pid, str) else None
                if p is None:
                    raise bad("result references an unknown package")
                if pid in seen:
                    raise bad("duplicate package in results")
                seen.add(pid)
                if r.get("actionId") != p["actionId"]:
                    raise bad("actionId does not match the stored proposal")
                if r.get("action") != p["action"]:
                    raise bad("action does not match the stored proposal")
                outcome = r.get("outcome")
                if outcome not in ("ACCEPTED", "REJECTED"):
                    raise bad("outcome must be ACCEPTED or REJECTED")
                if outcome == "ACCEPTED":
                    nonce = r.get("receiptNonce")
                    if not isinstance(nonce, str) or not nonce or len(nonce) > 1024:
                        raise bad("accepted results need a receiptNonce")
                    executions.append({
                        "packageId": pid, "actionId": p["actionId"], "action": p["action"],
                        "receiptNonce": nonce, "facts": p["facts"], "evidenceRefs": p["evidenceRefs"],
                    })
            stored = dict(msg)
            stored["taskId"], stored["contextId"] = task["id"], task["contextId"]
            task["history"].append(stored)
            task["artifacts"].append({
                "artifactId": "receipts-" + task["id"],
                "name": "invoice-action-receipts",
                "parts": [{"mediaType": MT_RECEIPTS, "data": {"batchId": prop["batchId"], "executions": executions}}],
            })
            task["status"] = {"state": COMPLETED, "timestamp": now_iso()}
            self.store.put_task(principal, task)
            self.store.put_message(principal, msg["messageId"], mhash, task["id"])
        log.info("task completed: %d accepted executions", len(executions))
        return task

    # ---- reads / cancel ------------------------------------------------------
    def get_task(self, principal, tid, query):
        hl = parse_hl(query)
        task = self.store.get_task(principal, tid)
        if task is None:
            raise not_found()
        return self.render(task, hl)

    def list_tasks(self, principal, query):
        def one(name):
            v = query.get(name)
            return v[0] if v else None

        try:
            page_size = int(one("pageSize")) if one("pageSize") is not None else 50
            offset = int(one("pageToken")) if one("pageToken") else 0
        except ValueError:
            raise bad("pageSize/pageToken must be integers")
        if not 1 <= page_size <= 100 or offset < 0:
            raise bad("pageSize must be between 1 and 100")
        hl = parse_hl(query)
        include_artifacts = (one("includeArtifacts") or "").lower() == "true"
        tasks = self.store.list_tasks(principal)
        ctx, status, after = one("contextId"), one("status"), one("statusTimestampAfter")
        if ctx:
            tasks = [t for t in tasks if t["contextId"] == ctx]
        if status:
            tasks = [t for t in tasks if t["status"]["state"] == status]
        if after:
            tasks = [t for t in tasks if t["status"].get("timestamp", "") >= after]
        total = len(tasks)
        page = tasks[offset: offset + page_size]
        views = [self.render(t, hl if hl is not None else 0, include_artifacts) for t in page]
        # keep within the response-size limit
        while len(views) > 1 and len(json.dumps(views, separators=(",", ":"))) > MAX_RESPONSE - 2048:
            views.pop()
        nxt = offset + len(views)
        return {
            "tasks": views,
            "nextPageToken": str(nxt) if nxt < total else "",
            "pageSize": page_size,
            "totalSize": total,
        }

    def cancel_task(self, principal, tid):
        with self.store.tx():
            task = self.store.get_task(principal, tid)
            if task is None:
                raise not_found()
            if task["status"]["state"] in TERMINAL:
                raise ApiError(409, "FAILED_PRECONDITION", "TASK_NOT_CANCELABLE", "Task cannot be canceled.")
            task["status"] = {"state": CANCELED, "timestamp": now_iso()}
            self.store.put_task(principal, task)
        return task

    # ---- startup recovery ---------------------------------------------------
    def resume_unfinished(self):
        for principal, task in self.store.submitted_tasks():
            def run(principal=principal, task=task):
                first = task["history"][0]["messageId"]
                with self.locks.hold((principal, first)):
                    cur = self.store.get_task(principal, task["id"])
                    if cur and cur["status"]["state"] in (SUBMITTED, WORKING):
                        try:
                            self._finish_decision(principal, cur, time.monotonic() + REQUEST_BUDGET)
                        except Exception:
                            log.exception("resume failed")
            threading.Thread(target=run, daemon=True).start()


def parse_hl(query):
    v = query.get("historyLength")
    if not v:
        return None
    try:
        n = int(v[0])
    except ValueError:
        raise bad("historyLength must be an integer")
    if n < 0:
        raise bad("historyLength must be >= 0")
    return n


# ----------------------------------------------------------------------------
# HTTP layer
# ----------------------------------------------------------------------------
TASK_RE = re.compile(r"^/tasks/([^/:]+)(:cancel)?$")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "InvoiceAgent/1.0"
    timeout = 120

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    # -- verbs
    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_PUT(self):
        self._handle("PUT")

    def do_DELETE(self):
        self._handle("DELETE")

    def do_PATCH(self):
        self._handle("PATCH")

    # -- plumbing
    def _send(self, status, obj, content_type=A2A_JSON, extra=None):
        data = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(data) > 512 * 1024 - 1024 and status == 200 and isinstance(obj, dict) and "task" in obj:
            t = dict(obj["task"])
            hist = t.get("history") or []
            if len(hist) > 1:
                t["history"] = hist[-1:]
            data = json.dumps({"task": t}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _error(self, e, extra=None):
        body = {"error": {
            "code": e.http, "status": e.status, "message": e.message,
            "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo",
                         "reason": e.reason, "domain": "a2a-protocol.org"}],
        }}
        self._send(e.http, body, extra=extra)

    def _read_body(self):
        te = (self.headers.get("Transfer-Encoding") or "").lower()
        if "chunked" in te:
            chunks, total = [], 0
            while True:
                line = self.rfile.readline(65537).strip()
                size = int(line.split(b";")[0] or b"0", 16)
                if size == 0:
                    while self.rfile.readline(65537).strip():
                        pass
                    break
                total += size
                if total > MAX_BODY:
                    raise ApiError(413, "INVALID_ARGUMENT", "PAYLOAD_TOO_LARGE", "Request body too large.")
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
            return b"".join(chunks)
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            self.close_connection = True
            raise ApiError(413, "INVALID_ARGUMENT", "PAYLOAD_TOO_LARGE", "Request body too large.")
        return self.rfile.read(n) if n else b""

    def _handle(self, method):
        app = self.server.app
        try:
            body_bytes = self._read_body() if method in ("POST", "PUT", "PATCH") else b""
            if method in ("GET", "DELETE"):
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
            parts = urlsplit(self.path)
            path = parts.path
            query = parse_qs(parts.query)

            # public: agent card
            if path.endswith("/.well-known/agent-card.json"):
                if method != "GET":
                    raise ApiError(405, "INVALID_ARGUMENT", "METHOD_NOT_ALLOWED", "Method not allowed.")
                host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host") or "localhost"
                proto = self.headers.get("X-Forwarded-Proto") or "https"
                return self._send(200, app.agent_card(f"{proto}://{host}/a2a/"), "application/json",
                                  {"Cache-Control": "no-cache"})

            prefix = app.prefix if app.base_url else "/a2a"
            if not path.startswith(prefix + "/"):
                raise ApiError(404, "NOT_FOUND", "NOT_FOUND", "Not found.")
            rel = path[len(prefix):]

            principal = self._authenticate()
            self._check_version(query)

            if rel == "/message:send":
                if method != "POST":
                    raise ApiError(405, "INVALID_ARGUMENT", "METHOD_NOT_ALLOWED", "Method not allowed.")
                self._check_content_type(body_bytes, required=True)
                try:
                    body = json.loads(body_bytes.decode("utf-8"))
                except Exception:
                    raise bad("Request body is not valid JSON.")
                return self._send(200, app.send_message(principal, body))
            if rel == "/tasks":
                if method != "GET":
                    raise ApiError(405, "INVALID_ARGUMENT", "METHOD_NOT_ALLOWED", "Method not allowed.")
                return self._send(200, app.list_tasks(principal, query))
            m = TASK_RE.match(rel)
            if m:
                tid = unquote(m.group(1))
                if m.group(2):
                    if method != "POST":
                        raise ApiError(405, "INVALID_ARGUMENT", "METHOD_NOT_ALLOWED", "Method not allowed.")
                    self._check_content_type(body_bytes, required=False)
                    return self._send(200, app.cancel_task(principal, tid))
                if method != "GET":
                    raise ApiError(405, "INVALID_ARGUMENT", "METHOD_NOT_ALLOWED", "Method not allowed.")
                return self._send(200, app.get_task(principal, tid, query))
            raise ApiError(404, "NOT_FOUND", "NOT_FOUND", "Not found.")
        except ApiError as e:
            extra = {"WWW-Authenticate": "Bearer"} if e.http == 401 else None
            self._error(e, extra)
        except Exception:
            log.exception("unhandled error")
            self._error(ApiError(500, "INTERNAL", "INTERNAL", "Internal error."))

    def _authenticate(self):
        h = self.headers.get("Authorization") or ""
        m = re.fullmatch(r"(?i)Bearer ([^\s]+)", h.strip())
        if not m:
            raise ApiError(401, "UNAUTHENTICATED", "UNAUTHENTICATED", "Authentication required.")
        allowed = self.server.allowed_tokens
        if allowed is not None and m.group(1) not in allowed:
            raise ApiError(401, "UNAUTHENTICATED", "UNAUTHENTICATED", "Authentication required.")
        return hashlib.sha256(m.group(1).encode("utf-8")).hexdigest()

    def _check_version(self, query):
        v = self.headers.get("A2A-Version") or (query.get("A2A-Version") or [None])[0]
        if v != "1.0":
            raise ApiError(400, "INVALID_ARGUMENT", "VERSION_NOT_SUPPORTED",
                           "Unsupported A2A version. This interface supports 1.0.")

    def _check_content_type(self, body, required):
        if not body and not required:
            return
        ct = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ct != A2A_JSON:
            raise ApiError(415, "INVALID_ARGUMENT", "CONTENT_TYPE_NOT_SUPPORTED",
                           "Content-Type must be application/a2a+json.")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, addr, app, allowed_tokens=None):
        super().__init__(addr, Handler)
        self.app = app
        self.allowed_tokens = allowed_tokens


def build_app(env=os.environ):
    store = Store(env.get("DB_PATH", "data/agent.sqlite"))
    llm = llm_from_env(env)
    decider = Decider(store, llm, env.get("EVIDENCE_REF_STYLE", "verbatim"), env.get("CACHE_SALT", ""))
    return App(store, decider, env.get("BASE_URL") or None), llm


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    app, llm = build_app()
    if llm is None:
        log.warning("LLM_API_KEY / LLM_MODEL not set: every package will get the conservative fallback decision!")
    if not app.base_url:
        log.warning("BASE_URL not set: the agent card URL is derived from request headers and routes use /a2a")
    tokens = os.environ.get("ALLOWED_TOKENS")
    allowed = set(t.strip() for t in tokens.split(",") if t.strip()) if tokens else None
    port = int(os.environ.get("PORT", "8080"))
    srv = Server((os.environ.get("HOST", "0.0.0.0"), port), app, allowed)
    app.resume_unfinished()
    log.info("listening on :%d  base=%s", port, app.base_url or "(derived)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
