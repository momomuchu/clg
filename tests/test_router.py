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
        route_logger: object | None = None,
        proxy_connection_factory: object | None = None,
        retry_sleeper: object | None = None,
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
            retry_sleeper=(
                retry_sleeper if retry_sleeper is not None else lambda _delay: None
            ),
            jitter_source=lambda: 0.0,
            route_logger=route_logger or (lambda _message: None),
            proxy_connection_factory=proxy_connection_factory,
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


def request(port: int, payload: dict[str, object], path: str = "/v1/messages", method: str = "POST") -> tuple[int, bytes]:
    body = json.dumps(payload).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request(
            method, path, body=body if method != "GET" else None,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))} if method != "GET" else {},
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

    def test_websocket_403_retries_on_same_upstream_after_backoff(self) -> None:
        class TransientRejectingStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if type(self).requests == 1:
                    body = b"WebSocket upgrade was rejected"
                    self.send_response(403)
                else:
                    body = b"retried-same-upstream"
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        delays: list[float] = []
        with StubServer(TransientRejectingStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}], retry_sleeper=delays.append
        ) as router:
            status, body = request(router.port, proxy_payload())

        self.assertEqual((status, body), (200, b"retried-same-upstream"))
        self.assertEqual(TransientRejectingStub.requests, 2)
        self.assertEqual(delays, [clg.RETRY_BASE_SECONDS])

    def test_websocket_retry_releases_permit_before_backoff(self) -> None:
        class TransientRejectingStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = b"WebSocket upgrade was rejected" if type(self).requests == 1 else b"ok"
                self.send_response(403 if type(self).requests == 1 else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        inflight_during_backoff: list[int] = []
        with StubServer(TransientRejectingStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}]
        ) as router:
            def inspect_backoff(_delay: float) -> None:
                inflight_during_backoff.append(router.server.scheduler.snapshot()[0]["inflight"])

            router.server.retry_sleeper = inspect_backoff
            self.assertEqual(request(router.port, proxy_payload()), (200, b"ok"))

        self.assertEqual(inflight_during_backoff, [0])

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

    def test_non_websocket_403_passes_through_once(self) -> None:
        class AuthFailureStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = b"authentication rejected"
                self.send_response(403)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        delays: list[float] = []
        with StubServer(AuthFailureStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}], retry_sleeper=delays.append
        ) as router:
            status, body = request(router.port, proxy_payload())

        self.assertEqual((status, body), (403, b"authentication rejected"))
        self.assertEqual(AuthFailureStub.requests, 1)
        self.assertEqual(delays, [])

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

    def test_persistent_websocket_403_gives_up_after_retry_budget(self) -> None:
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

        delays: list[float] = []
        with StubServer(RejectingStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}], retry_sleeper=delays.append
        ) as router:
            status, body = request(router.port, proxy_payload())

        self.assertEqual(RejectingStub.requests, clg.MAX_PROXY_ATTEMPTS)
        self.assertEqual(status, 403)
        self.assertEqual(body, f"WebSocket upgrade was rejected attempt-{clg.MAX_PROXY_ATTEMPTS}".encode())
        self.assertEqual(len(delays), clg.MAX_PROXY_ATTEMPTS - 1)

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


    def test_permit_is_released_when_logger_or_close_fails(self) -> None:
        class SuccessStub(QuietHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

        with StubServer(SuccessStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}], route_logger=lambda _msg: (_ for _ in ()).throw(OSError("log closed"))
        ) as router:
            status, _body = request(router.port, proxy_payload())
            self.assertEqual(status, 502)
            self.assertEqual(router.server.scheduler.snapshot()[0]["inflight"], 0)

        class CloseFails:
            def __init__(self, port: int) -> None:
                self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            def request(self, *args: object, **kwargs: object) -> None:
                self.conn.request(*args, **kwargs)
            def getresponse(self) -> http.client.HTTPResponse:
                return self.conn.getresponse()
            def close(self) -> None:
                self.conn.close()
                raise OSError("close failed")

        with StubServer(SuccessStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}], proxy_connection_factory=CloseFails
        ) as router:
            self.assertEqual(request(router.port, proxy_payload())[0], 200)
            self.assertEqual(router.server.scheduler.snapshot()[0]["inflight"], 0)

    def test_fifo_timeout_and_monotonic_aimd(self) -> None:
        scheduler = clg.UpstreamScheduler([{"name": "a", "port": 1}], initial_permits=1)
        held = scheduler.acquire()
        order: list[str] = []
        acquired: list[object] = []
        def wait(name: str) -> None:
            upstream = scheduler.acquire()
            acquired.append(upstream)
            order.append(name)
        first = threading.Thread(target=wait, args=("first",))
        second = threading.Thread(target=wait, args=("second",))
        first.start(); second.start(); time.sleep(0.05)
        scheduler.release(held)
        first.join(1)
        self.assertEqual(order, ["first"])
        self.assertRaises(TimeoutError, scheduler.acquire, timeout=0)
        self.assertEqual(scheduler.snapshot()[0]["inflight"], 1)
        lone = clg.UpstreamScheduler([{"name": "only", "port": 2}], initial_permits=1)
        retry_outcome: list[str] = []
        def reject_same_upstream() -> None:
            try:
                retried = lone.acquire(avoid_name="only", timeout=0.05)
            except TimeoutError:
                retry_outcome.append("timeout")
            else:
                retry_outcome.append("same-upstream")
                lone.release(retried)
        retry_thread = threading.Thread(target=reject_same_upstream)
        retry_thread.start(); retry_thread.join(1)
        self.assertEqual(retry_outcome, ["timeout"])
        scheduler.release(acquired.pop())
        second.join(1)
        self.assertEqual(order, ["first", "second"])
        scheduler.release(acquired.pop())
        scheduler._upstreams[0].permits = clg.MAX_PERMITS + 3
        for _ in range(clg.SUCCESS_WINDOW):
            scheduler.succeeded(scheduler._upstreams[0])
        self.assertEqual(scheduler.snapshot()[0]["permits"], clg.MAX_PERMITS + 3)

    def test_wire_routing_fails_safe_for_malformed_inputs(self) -> None:
        class ProxyStub(QuietHandler):
            requests = 0
            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
        class AnthropicStub(QuietHandler):
            requests = 0
            def do_POST(self) -> None:
                type(self).requests += 1
                self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"no")
        original = clg.oauth_token; clg.oauth_token = lambda: "token"
        try:
            with StubServer(ProxyStub) as proxy, StubServer(AnthropicStub) as anth, RunningRouter(
                [{"name": "a", "port": proxy.port}], anthropic_port=anth.port
            ) as router:
                cases = [
                    {"model": "claude-opus-5[1m][1m]", "system": "You are an interactive agent"},
                    {"system": "You are an interactive agent"},
                    {"model": "claude-opus-5", "system": [{"text": {"x": 1}}]},
                    {"model": "claude-opus-5", "system": "plain string"},
                ]
                for payload in cases:
                    self.assertEqual(request(router.port, payload)[0], 200)
        finally:
            clg.oauth_token = original
        self.assertEqual(ProxyStub.requests, 4)
        self.assertEqual(AnthropicStub.requests, 0)

    def test_content_length_limits_and_socket_timeout_are_configured(self) -> None:
        class Unused(QuietHandler): pass
        with StubServer(Unused) as upstream, RunningRouter([{"name": "a", "port": upstream.port}]) as router:
            sock = __import__("socket").create_connection(("127.0.0.1", router.port))
            try:
                sock.sendall(b"POST /v1/messages HTTP/1.0\r\nContent-Length: nope\r\n\r\n")
                self.assertIn(b"400", sock.recv(1024))
            finally:
                sock.close()
            sock = __import__("socket").create_connection(("127.0.0.1", router.port))
            try:
                sock.sendall(f"POST /v1/messages HTTP/1.0\r\nContent-Length: {clg.MAX_REQUEST_BODY_BYTES + 1}\r\n\r\n".encode())
                self.assertIn(b"413", sock.recv(1024))
            finally:
                sock.close()
        self.assertEqual(clg.SOCKET_READ_TIMEOUT_SECONDS, 600)

    def test_non_generation_bypasses_permits_and_aimd(self) -> None:
        class Stub(QuietHandler):
            def do_GET(self) -> None:
                self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
        with StubServer(Stub) as upstream, RunningRouter([{"name": "a", "port": upstream.port}], initial_permits=1) as router:
            state = router.server.scheduler._upstreams[0]
            state.inflight = 1
            for _ in range(clg.SUCCESS_WINDOW):
                self.assertEqual(request(router.port, {}, "/v1/messages/count_tokens")[0], 200)
            self.assertEqual(request(router.port, {}, "/v1/models", "GET")[0], 200)
            self.assertEqual(router.server.scheduler.snapshot()[0]["inflight"], 1)
            self.assertEqual(router.server.scheduler.snapshot()[0]["successes"], 0)
            self.assertEqual(router.server.scheduler.snapshot()[0]["permits"], 1)

    def test_stream_truncation_and_client_disconnect_release_permit_without_aimd_success(self) -> None:
        class TruncatedResponse:
            status = 200
            def getheaders(self) -> list[tuple[str, str]]: return []
            def read1(self, _size: int) -> bytes: raise http.client.IncompleteRead(b"partial", 10)
        class TruncatedConnection:
            def __init__(self, _port: int) -> None: pass
            def request(self, *_args: object, **_kwargs: object) -> None: pass
            def getresponse(self) -> TruncatedResponse: return TruncatedResponse()
            def close(self) -> None: pass
        with StubServer(QuietHandler) as upstream, RunningRouter(
            [{"name":"a", "port": upstream.port}], proxy_connection_factory=TruncatedConnection
        ) as router:
            status, _body = request(router.port, proxy_payload())
            state = router.server.scheduler.snapshot()[0]
        self.assertEqual(status, 200)
        self.assertEqual(state["inflight"], 0)
        self.assertEqual(state["successes"], 0)

        class LargeStream(QuietHandler):
            def handle(self) -> None:
                try:
                    super().handle()
                except ConnectionResetError:
                    pass

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200); self.end_headers()
                for _ in range(30):
                    try:
                        self.wfile.write(b"x" * 65536); self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
        with StubServer(LargeStream) as upstream, RunningRouter([{"name":"a", "port":upstream.port}]) as router:
            conn = __import__("socket").create_connection(("127.0.0.1", router.port))
            conn.sendall(b"POST /v1/messages HTTP/1.0\r\nContent-Length: 2\r\n\r\n{}")
            conn.recv(128); conn.close(); time.sleep(0.1)
            state = router.server.scheduler.snapshot()[0]
        self.assertEqual(state["inflight"], 0)
        self.assertEqual(state["successes"], 0)

    def test_oauth_cache_invalidation_is_compare_and_swap(self) -> None:
        clg._token_cache = ("fresh", time.time())
        clg.invalidate_oauth_token("stale")
        self.assertEqual(clg._token_cache[0], "fresh")
        clg.invalidate_oauth_token("fresh")
        self.assertIsNone(clg._token_cache)

    def test_pinned_websocket_retry_stays_on_pinned_upstream(self) -> None:
        class UnpinnedStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200)
                self.send_header("Content-Length", "8")
                self.end_headers()
                self.wfile.write(b"unwanted")

        class PinnedStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if type(self).requests == 1:
                    body = b"WebSocket upgrade was rejected"
                    self.send_response(403)
                else:
                    body = b"pinned-ok"
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with StubServer(UnpinnedStub) as first, StubServer(PinnedStub) as second, RunningRouter(
            [{"name": "a", "port": first.port}, {"name": "b", "port": second.port}]
        ) as router:
            conn = http.client.HTTPConnection("127.0.0.1", router.port, timeout=10)
            try:
                body = json.dumps(proxy_payload()).encode()
                conn.request(
                    "POST", "/v1/messages", body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Content-Length": str(len(body)),
                        clg.PINNED_UPSTREAM_HEADER: "b",
                    },
                )
                response = conn.getresponse()
                status, payload = response.status, response.read()
            finally:
                conn.close()

        self.assertEqual((status, payload), (200, b"pinned-ok"))
        self.assertEqual(UnpinnedStub.requests, 0)
        self.assertEqual(PinnedStub.requests, 2)

    def test_retry_preference_falls_back_to_rejected_upstream_when_alternative_is_busy(self) -> None:
        scheduler = clg.UpstreamScheduler(
            [{"name": "a", "port": 1}, {"name": "b", "port": 2}], initial_permits=1
        )
        preferred = scheduler.acquire(prefer_not_name="a", timeout=0)
        self.assertEqual(preferred.name, "b")
        scheduler.release(preferred)

        occupied_alternative = scheduler.acquire(only_name="b", timeout=0)
        fallback = scheduler.acquire(prefer_not_name="a", timeout=0)
        self.assertEqual(fallback.name, "a")
        scheduler.release(fallback)
        scheduler.release(occupied_alternative)

    def test_pinned_compatibility_router_delegates_to_shared_scheduler(self) -> None:
        class A(QuietHandler):
            def do_POST(self) -> None: self.send_response(500); self.send_header("Content-Length", "0"); self.end_headers()
        class B(QuietHandler):
            requests = 0
            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
        with StubServer(A) as a, StubServer(B) as b, RunningRouter([{"name":"a","port":a.port},{"name":"b","port":b.port}]) as shared:
            relay = clg.RouterServer(("127.0.0.1", 0), [{"name":"a","port":a.port},{"name":"b","port":b.port}], delegate_port=shared.port, pinned_upstream="b")
            thread = threading.Thread(target=relay.serve_forever, daemon=True); thread.start()
            try:
                self.assertEqual(request(relay.server_address[1], proxy_payload())[0], 200)
                self.assertEqual(B.requests, 1)
                self.assertEqual(relay.scheduler.snapshot()[1]["inflight"], 0)
                self.assertEqual(shared.server.scheduler.snapshot()[1]["inflight"], 0)
            finally:
                relay.shutdown(); relay.server_close(); thread.join(2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
