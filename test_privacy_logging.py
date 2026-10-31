import asyncio
import io
import json
import logging
import logging.config
import unittest
from pathlib import Path
from privacy_logging import RequestAuditMiddleware, SafeJSONFormatter


class PrivacyTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.logger = logging.getLogger("privacy.audit")
        self.logger.disabled = False
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = logging.StreamHandler(self.output)
        handler.setFormatter(SafeJSONFormatter())
        self.logger.handlers = [handler]

    def events(self):
        return [json.loads(line) for line in self.output.getvalue().splitlines()]

    def test_diagnostic_message_and_traceback_are_not_formatted(self):
        class Unsafe:
            def __str__(self):
                raise AssertionError("Log arguments must never be formatted")
        record = logging.LogRecord("vllm.worker", 40, "path", 42, "%s", (Unsafe(),), None)
        try:
            raise ValueError("PRIVATE_EXCEPTION_MARKER")
        except ValueError:
            import sys
            record.exc_info = sys.exc_info()
        record.stack_info = "PRIVATE_STACK"
        result = SafeJSONFormatter().format(record)
        self.assertNotIn("PRIVATE", result)
        self.assertTrue(json.loads(result)["message_suppressed"])

    def test_streaming_is_forwarded_unchanged_and_logged_once(self):
        sent = []
        payload = b'PRIVATE_RESPONSE'
        async def app(scope, receive, send):
            self.assertEqual((await receive())["body"], b"PRIVATE_PROMPT")
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": payload, "more_body": True})
            await send({"type": "http.response.body", "body": b"END", "more_body": False})
        async def receive():
            return {"type": "http.request", "body": b"PRIVATE_PROMPT"}
        async def send(message):
            sent.append(message)
        scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
                 "query_string": b"PRIVATE_QUERY", "headers": [(b"authorization", b"PRIVATE_TOKEN")]}
        asyncio.run(RequestAuditMiddleware(app)(scope, receive, send))
        self.assertEqual(sent[1]["body"], payload)
        events = self.events()
        self.assertEqual([e["event"] for e in events], ["request_received", "request_finished"])
        self.assertEqual(events[-1]["status"], 200)
        self.assertEqual(events[-1]["outcome"], "complete")
        self.assertEqual(events[0]["request_id"], events[1]["request_id"])
        self.assertNotIn("PRIVATE", self.output.getvalue())

    def test_unauthorized_does_not_read_body(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 401})
            await send({"type": "http.response.body", "body": b"Unauthorized"})
        async def receive():
            raise AssertionError("Audit must not read the body")
        async def send(message):
            pass
        asyncio.run(RequestAuditMiddleware(app)(
            {"type": "http", "method": "POST", "path": "/PRIVATE_PATH"}, receive, send))
        self.assertEqual(self.events()[-1]["status"], 401)
        self.assertEqual(self.events()[-1]["endpoint"], "other")
        self.assertNotIn("PRIVATE", self.output.getvalue())

    def test_exception_is_reraised_without_logging_its_text(self):
        async def app(scope, receive, send):
            raise ValueError("PRIVATE_ERROR")
        async def noop(*args):
            pass
        with self.assertRaises(ValueError):
            asyncio.run(RequestAuditMiddleware(app)(
                {"type": "http", "method": "POST", "path": "/v1/chat/completions"}, noop, noop))
        self.assertEqual(self.events()[-1]["outcome"], "error")
        self.assertEqual(self.events()[-1]["status"], 500)
        self.assertNotIn("PRIVATE", self.output.getvalue())

    def test_lifespan_readiness(self):
        async def app(scope, receive, send):
            await send({"type": "lifespan.startup.complete"})
        async def noop(*args):
            pass
        asyncio.run(RequestAuditMiddleware(app)({"type": "lifespan"}, noop, noop))
        self.assertEqual(self.events()[0]["event"], "server_ready")

    def test_forged_fields_are_not_emitted(self):
        self.logger.info("PRIVATE_MESSAGE", extra={"audit": {
            "event": "request_finished", "method": "PRIVATE_METHOD", "endpoint": "PRIVATE_ENDPOINT",
            "request_id": "PRIVATE_ID", "status": "PRIVATE_STATUS", "duration_ms": float("nan"),
            "prompt": "PRIVATE_PROMPT", "outcome": "PRIVATE_OUTCOME"}})
        self.assertNotIn("PRIVATE", self.output.getvalue())

    def test_configuration_covers_vllm_uvicorn_and_root(self):
        logging.config.dictConfig(json.loads(Path(__file__).with_name("logging.json").read_text()))
        for name in ("vllm.worker", "uvicorn.error", "uvicorn.access", "privacy.audit", "new_library"):
            logger = logging.getLogger(name)
            handlers = []
            while logger:
                handlers.extend(logger.handlers)
                if not logger.propagate:
                    break
                logger = logger.parent
            self.assertTrue(handlers, name)
            self.assertTrue(all(isinstance(h.formatter, SafeJSONFormatter) for h in handlers), name)


if __name__ == "__main__":
    unittest.main()
