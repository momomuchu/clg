#!/usr/bin/env python3
"""Behavior locks for clg's concurrency-aware local router."""

from __future__ import annotations

import concurrent.futures
import http.client
import importlib.machinery
import importlib.util
import json
import pathlib
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


CLG_PATH = pathlib.Path(__file__).parents[1] / "bin" / "clg"
LOADER = importlib.machinery.SourceFileLoader("clg_launcher", str(CLG_PATH))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
clg = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = clg
LOADER.exec_module(clg)


class QuietHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        if getattr(self.server, "suppress_logs", False):
            return
        super().log_message(_format, *_args)


class StubServer:
    def __init__(self, handler: type[BaseHTTPRequestHandler]) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.server.suppress_logs = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def __enter__(self) -> "StubServer":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class RunningRouter:
    def __init__(
        self,
        upstreams: list[dict[str, object]],
        *,
        initial_permits: int = 6,
        anthropic_port: int | None = None,
    ) -> None:
        anthropic_factory = None
        if anthropic_port is not None:
            anthropic_factory = lambda: http.client.HTTPConnection(
                "127.0.0.1", anthropic_port, timeout=10
            )
        self.server = clg.RouterServer(
            ("127.0.0.1", 0),
            upstreams,
            initial_permits=initial_permits,
            anthropic_connection_factory=anthropic_factory,
            retry_sleeper=lambda _delay: None,
            jitter_source=lambda: 0.0,
            route_logger=lambda _message: None,
        )
        self.server.suppress_logs = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def __enter__(self) -> "RunningRouter":
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def request(port: int, payload: dict[str, object]) -> tuple[int, bytes]:
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(
            "POST",
            "/v1/messages",
            body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
        )
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def proxy_payload(model: str = "claude-sonnet-4-5") -> dict[str, object]:
    return {"model": model, "system": "You are an agent for a bounded task", "messages": []}


def main_payload() -> dict[str, object]:
    return {
        "model": "claude-opus-5[1m]",
        "system": "You are an interactive agent running Claude Code",
        "messages": [],
    }


class RouterTests(unittest.TestCase):
    def test_cap_enforced(self) -> None:
        class CappedStub(QuietHandler):
            lock = threading.Lock()
            inflight = 0
            max_inflight = 0

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                with self.lock:
                    type(self).inflight += 1
                    type(self).max_inflight = max(type(self).max_inflight, type(self).inflight)
                try:
                    time.sleep(0.05)
                    body = b"ok"
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                finally:
                    with self.lock:
                        type(self).inflight -= 1

        with StubServer(CappedStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}], initial_permits=3
        ) as router:
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                results = list(pool.map(lambda _i: request(router.port, proxy_payload()), range(12)))

        self.assertTrue(all(status == 200 for status, _body in results))
        self.assertLessEqual(CappedStub.max_inflight, 3)

    def test_websocket_403_retries_on_another_upstream_and_decreases_permits(self) -> None:
        class RejectingStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = b'WebSocket upgrade was rejected'
                self.send_response(403)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class SuccessStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = b"retried"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with StubServer(RejectingStub) as first, StubServer(SuccessStub) as second, RunningRouter(
            [{"name": "a", "port": first.port}, {"name": "b", "port": second.port}]
        ) as router:
            status, body = request(router.port, proxy_payload())
            states = {state["name"]: state for state in router.server.scheduler.snapshot()}

        self.assertEqual((status, body), (200, b"retried"))
        self.assertEqual(RejectingStub.requests, 1)
        self.assertEqual(SuccessStub.requests, 1)
        self.assertLess(states["a"]["permits"], 6)

    def test_non_retry_statuses_pass_through_once(self) -> None:
        for expected_status in (500, 413):
            with self.subTest(status=expected_status):
                class StatusStub(QuietHandler):
                    requests = 0

                    def do_POST(self) -> None:
                        type(self).requests += 1
                        self.rfile.read(int(self.headers.get("Content-Length", "0")))
                        body = f"status-{expected_status}".encode()
                        self.send_response(expected_status)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)

                with StubServer(StatusStub) as upstream, RunningRouter(
                    [{"name": "a", "port": upstream.port}]
                ) as router:
                    status, body = request(router.port, proxy_payload())

                self.assertEqual(status, expected_status)
                self.assertEqual(body, f"status-{expected_status}".encode())
                self.assertEqual(StatusStub.requests, 1)

    def test_non_403_websocket_text_passes_through_once(self) -> None:
        for expected_status in (500, 429):
            with self.subTest(status=expected_status):
                class StatusStub(QuietHandler):
                    requests = 0
                    body = b"WebSocket upgrade was rejected"

                    def do_POST(self) -> None:
                        type(self).requests += 1
                        self.rfile.read(int(self.headers.get("Content-Length", "0")))
                        self.send_response(expected_status)
                        self.send_header("Content-Length", str(len(type(self).body)))
                        self.end_headers()
                        self.wfile.write(type(self).body)

                with StubServer(StatusStub) as upstream, RunningRouter(
                    [{"name": "a", "port": upstream.port}]
                ) as router:
                    status, body = request(router.port, proxy_payload())

                self.assertEqual(status, expected_status)
                self.assertEqual(body, StatusStub.body)
                self.assertEqual(StatusStub.requests, 1)

    def test_large_403_is_not_inspected_or_retried(self) -> None:
        class LargeRejectingStub(QuietHandler):
            requests = 0
            body = b"WebSocket upgrade" + b"x" * clg.RETRY_BODY_LIMIT

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(403)
                self.send_header("Content-Length", str(len(type(self).body)))
                self.end_headers()
                self.wfile.write(type(self).body)

        with StubServer(LargeRejectingStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}]
        ) as router:
            status, body = request(router.port, proxy_payload())

        self.assertEqual(LargeRejectingStub.requests, 1)
        self.assertEqual(status, 403)
        self.assertEqual(body, LargeRejectingStub.body)

    def test_anthropic_main_chat_bypasses_proxy_permits(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class AnthropicStub(QuietHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                started.set()
                release.wait(timeout=5)
                body = b"anthropic"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class ProxyStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()

        original_oauth_token = clg.oauth_token
        clg.oauth_token = lambda: "test-token"
        try:
            with StubServer(ProxyStub) as proxy, StubServer(AnthropicStub) as anthropic, RunningRouter(
                [{"name": "a", "port": proxy.port}], anthropic_port=anthropic.port
            ) as router:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(request, router.port, main_payload())
                    self.assertTrue(started.wait(timeout=2))
                    state_during_request = router.server.scheduler.snapshot()[0]
                    release.set()
                    result = future.result(timeout=5)
        finally:
            clg.oauth_token = original_oauth_token

        self.assertEqual(result, (200, b"anthropic"))
        self.assertEqual(state_during_request["inflight"], 0)
        self.assertEqual(ProxyStub.requests, 0)

    def test_routing_table(self) -> None:
        inherited = proxy_payload("claude-opus-5")
        self.assertEqual(clg.route_model("claude-opus-5", inherited), ("proxy", clg.TERRA))
        self.assertEqual(
            clg.route_model("claude-fable-1", proxy_payload("claude-fable-1")),
            ("proxy", clg.SOL),
        )
        self.assertEqual(
            clg.route_model("claude-opus-5[1m]", main_payload()),
            ("anthropic", "claude-opus-5"),
        )

    def test_health_reports_upstream_capacity(self) -> None:
        class UnusedStub(QuietHandler):
            pass

        with StubServer(UnusedStub) as first, StubServer(UnusedStub) as second, RunningRouter(
            [{"name": "a", "port": first.port}, {"name": "b", "port": second.port}],
            initial_permits=3,
        ) as router:
            conn = http.client.HTTPConnection("127.0.0.1", router.port, timeout=10)
            try:
                conn.request("GET", "/__clg")
                response = conn.getresponse()
                payload = json.loads(response.read())
            finally:
                conn.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(
            payload["upstreams"],
            [
                {"free": 3, "inflight": 0, "name": "a", "permits": 3, "port": first.port, "successes": 0},
                {"free": 3, "inflight": 0, "name": "b", "permits": 3, "port": second.port, "successes": 0},
            ],
        )

    def test_all_websocket_retries_fail_with_last_response_unchanged(self) -> None:
        class RejectingStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = f"WebSocket upgrade was rejected attempt-{type(self).requests}".encode()
                self.send_response(403)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with StubServer(RejectingStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}]
        ) as router:
            status, body = request(router.port, proxy_payload())

        self.assertEqual(RejectingStub.requests, clg.MAX_PROXY_ATTEMPTS)
        self.assertEqual(status, 403)
        self.assertEqual(body, b"WebSocket upgrade was rejected attempt-4")

    def test_aimd_additive_increase_is_bounded(self) -> None:
        scheduler = clg.UpstreamScheduler([{"name": "a", "port": 1}], initial_permits=3)
        upstream = scheduler.acquire()
        scheduler.release(upstream)
        for _ in range(clg.SUCCESS_WINDOW):
            scheduler.succeeded(upstream)
        self.assertEqual(scheduler.snapshot()[0]["permits"], 4)

        upstream.permits = clg.MAX_PERMITS
        for _ in range(clg.SUCCESS_WINDOW):
            scheduler.succeeded(upstream)
        self.assertEqual(scheduler.snapshot()[0]["permits"], clg.MAX_PERMITS)

    def test_chunked_streaming_is_byte_identical(self) -> None:
        expected = b"first chunk\nsecond chunk with binary: \x00\xff\nlast"

        class ChunkedStub(QuietHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                pieces = (expected[:7], expected[7:29], expected[29:])
                for piece in pieces:
                    self.wfile.write(f"{len(piece):X}\r\n".encode() + piece + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        with StubServer(ChunkedStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}]
        ) as router:
            status, body = request(router.port, proxy_payload())

        self.assertEqual(status, 200)
        self.assertEqual(body, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
