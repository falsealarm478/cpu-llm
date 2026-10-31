"""Allowlisted JSON logging and body-transparent ASGI request auditing."""
import json
import logging
import math
import re
import time
import uuid
from datetime import datetime, timezone

METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
ENDPOINTS = {
    "/v1/chat/completions": "chat_completions",
    "/v1/completions": "completions",
    "/v1/models": "models",
    "/health": "health",
    "/metrics": "metrics",
}
EVENTS = {"request_received", "request_finished", "server_ready", "server_stopped"}
OUTCOMES = {"complete", "incomplete", "error", "cancelled"}


class SafeJSONFormatter(logging.Formatter):
    def format(self, record):
        # Never call getMessage()/formatException(): either can expose payloads.
        result = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": logging.getLevelName(record.levelno),
        }
        audit = getattr(record, "audit", None)
        if (record.name == "privacy.audit" and isinstance(audit, dict)
                and audit.get("event") in EVENTS):
            result["event"] = audit["event"]
            for key, allowed in (
                ("method", METHODS | {"OTHER"}),
                ("endpoint", set(ENDPOINTS.values()) | {"other"}),
                ("outcome", OUTCOMES),
            ):
                value = audit.get(key)
                if isinstance(value, str) and value in allowed:
                    result[key] = value
            request_id = audit.get("request_id", "")
            if isinstance(request_id, str) and re.fullmatch(r"[0-9a-f]{32}", request_id):
                result["request_id"] = request_id
            status = audit.get("status")
            if type(status) is int and 100 <= status <= 599:
                result["status"] = status
            duration = audit.get("duration_ms")
            if type(duration) in (float, int) and math.isfinite(duration) and duration >= 0:
                result["duration_ms"] = round(duration, 2)
        else:
            result.update(event="diagnostic", message_suppressed=True)
            # Fixed categories only; no arbitrary logger names or file paths.
            result["component"] = (
                "vllm" if record.name.startswith("vllm") else
                "uvicorn" if record.name.startswith("uvicorn") else "other"
            )
            result["line"] = record.lineno
        return json.dumps(result, ensure_ascii=True, allow_nan=False)


def emit(event, **fields):
    logging.getLogger("privacy.audit").info("audit", extra={"audit": {"event": event, **fields}})


class RequestAuditMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            async def lifespan_send(message):
                await send(message)
                if message["type"] == "lifespan.startup.complete":
                    emit("server_ready")
                elif message["type"] == "lifespan.shutdown.complete":
                    emit("server_stopped")
            return await self.app(scope, receive, lifespan_send)
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Do not read body, response contents, query strings, or any headers.
        fields = {
            "request_id": uuid.uuid4().hex,
            "method": scope.get("method") if scope.get("method") in METHODS else "OTHER",
            "endpoint": ENDPOINTS.get(scope.get("path"), "other"),
        }
        started = time.perf_counter()
        status = 500
        outcome = "incomplete"
        emit("request_received", **fields)

        async def audit_send(message):
            nonlocal status, outcome
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                outcome = "complete"

        try:
            await self.app(scope, receive, audit_send)
        except BaseException as exc:
            outcome = "error" if isinstance(exc, Exception) else "cancelled"
            raise
        finally:
            emit("request_finished", **fields, status=status, outcome=outcome,
                 duration_ms=(time.perf_counter() - started) * 1000)
