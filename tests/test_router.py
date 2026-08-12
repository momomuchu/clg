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
    def setUp(self) -> None:
        # Le module lit ~/.config/clg/.../routing.json au chargement : sans ce
        # verrou, la config locale de la machine ferait passer ou échouer la
        # suite. Ces tests décrivent le comportement par défaut.
        self._main_chat = clg.MAIN_CHAT_UPSTREAM
        clg.MAIN_CHAT_UPSTREAM = "anthropic"

    def tearDown(self) -> None:
        clg.MAIN_CHAT_UPSTREAM = self._main_chat

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

    def test_every_anthropic_family_is_translated_to_gpt(self) -> None:
        """Régression : sonnet/haiku partaient au proxy sans traduction."""
        for model, expected in (
            ("claude-sonnet-5", clg.TERRA),
            ("claude-sonnet-4-5", clg.TERRA),
            ("sonnet", clg.TERRA),
            ("claude-haiku-4-5-20251001", clg.LUNA),
            ("claude-3-5-haiku-20241022", clg.LUNA),
            ("haiku", clg.LUNA),
            ("claude-opus-4-8", clg.SOL),
            ("opus", clg.SOL),
            ("fable", clg.SOL),
        ):
            with self.subTest(model=model):
                self.assertEqual(
                    clg.route_model(model, proxy_payload(model)), ("proxy", expected)
                )

    def test_gpt_models_pass_through_untouched(self) -> None:
        for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            with self.subTest(model=model):
                self.assertEqual(
                    clg.route_model(model, proxy_payload(model)), ("proxy", model)
                )

    def test_non_gpt_upstream_models_never_reach_the_proxy_verbatim(self) -> None:
        """Le proxy sert aussi grok/kimi/deepseek : rien de tout ça ne doit sortir."""
        for model in ("grok-4.5", "kimi-k3", "deepseek-v4-pro", "glm-5.2", "qwen3.7-max"):
            with self.subTest(model=model):
                target, effective = clg.route_model(model, proxy_payload(model))
                self.assertEqual(target, "proxy")
                self.assertTrue(effective.startswith(clg.GPT_PREFIX), effective)

    def test_compaction_reaches_anthropic_whatever_the_incoming_model(self) -> None:
        """Le prompt de compaction est le discriminant, pas le modèle."""
        compaction = {
            "model": "ignored",
            "system": [
                {
                    "type": "text",
                    "text": "You are a helpful AI assistant tasked with "
                    "summarizing conversations.",
                }
            ],
        }
        for incoming in ("claude-opus-5", "gpt-5.6-sol", "gpt-5.6-terra"):
            with self.subTest(incoming=incoming):
                self.assertEqual(
                    clg.route_model(incoming, compaction),
                    ("anthropic", clg.COMPACT_MODEL),
                )

    def test_main_chat_follows_the_main_chat_setting(self) -> None:
        original = clg.MAIN_CHAT_UPSTREAM
        try:
            clg.MAIN_CHAT_UPSTREAM = "gpt"
            self.assertEqual(
                clg.route_model(clg.MAIN_MODEL, main_payload()),
                ("proxy", clg.INHERITED),
            )
            clg.MAIN_CHAT_UPSTREAM = "anthropic"
            self.assertEqual(
                clg.route_model(clg.MAIN_MODEL, main_payload()),
                ("anthropic", clg.MAIN_MODEL),
            )
        finally:
            clg.MAIN_CHAT_UPSTREAM = original

    def test_main_model_reaches_anthropic_only_for_the_main_chat(self) -> None:
        self.assertEqual(
            clg.route_model(clg.MAIN_MODEL, proxy_payload(clg.MAIN_MODEL)),
            ("proxy", clg.INHERITED),
        )
        self.assertEqual(
            clg.route_model(clg.MAIN_MODEL, main_payload()),
            ("anthropic", clg.MAIN_MODEL),
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

        self.assertEqual(RejectingStub.requests, 1)
        self.assertEqual(status, 403)
        self.assertEqual(body, b"WebSocket upgrade was rejected attempt-1")

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


COMPACTION_TEXT = (
    "You are a helpful AI assistant tasked with summarizing conversations."
)


def compaction_payload(model: str = "claude-opus-5", blocks: list[str] | None = None) -> dict:
    system = [{"type": "text", "text": text} for text in (blocks or [COMPACTION_TEXT])]
    return {"model": model, "system": system, "messages": []}


class CompactionRoutingTests(unittest.TestCase):
    """La compaction est toujours servie par Claude, quel que soit le mode."""

    def _with_main_chat(self, value: str):
        original = clg.MAIN_CHAT_UPSTREAM
        clg.MAIN_CHAT_UPSTREAM = value
        self.addCleanup(lambda: setattr(clg, "MAIN_CHAT_UPSTREAM", original))

    def test_compaction_goes_to_anthropic_in_both_modes(self) -> None:
        for mode in ("anthropic", "gpt"):
            for model in ("claude-opus-5", "gpt-5.6-sol", "gpt-5.6-terra", "grok-4.5", "sonnet"):
                with self.subTest(main_chat=mode, model=model):
                    self._with_main_chat(mode)
                    self.assertEqual(
                        clg.route_model(model, compaction_payload(model)),
                        ("anthropic", clg.COMPACT_MODEL),
                    )

    def test_compaction_accepts_the_1m_suffix(self) -> None:
        self.assertEqual(
            clg.route_model("claude-opus-5[1m]", compaction_payload("claude-opus-5[1m]")),
            ("anthropic", clg.COMPACT_MODEL),
        )

    def test_compaction_detected_behind_a_billing_header_block(self) -> None:
        payload = compaction_payload(
            blocks=["x-anthropic-billing-header: cc_version=2.1.228", COMPACTION_TEXT]
        )
        self.assertEqual(
            clg.route_model("gpt-5.6-sol", payload), ("anthropic", clg.COMPACT_MODEL)
        )

    def test_compaction_as_a_plain_string_system(self) -> None:
        payload = {"model": "gpt-5.6-sol", "system": COMPACTION_TEXT, "messages": []}
        self.assertEqual(
            clg.route_model("gpt-5.6-sol", payload), ("anthropic", clg.COMPACT_MODEL)
        )

    def test_compaction_wins_over_the_main_chat_marker(self) -> None:
        """Si les deux marqueurs coexistent, la compaction prime."""
        self._with_main_chat("anthropic")
        payload = compaction_payload(
            blocks=[COMPACTION_TEXT, "You are an interactive agent that helps users"]
        )
        self.assertEqual(
            clg.route_model("claude-opus-5", payload), ("anthropic", clg.COMPACT_MODEL)
        )

    def test_marker_past_the_scan_window_falls_back_to_gpt(self) -> None:
        """Limite assumée : au-delà de 4 blocs ou 500 caractères, on ne voit rien."""
        deep = compaction_payload(blocks=["filler"] * 4 + [COMPACTION_TEXT])
        target, _ = clg.route_model("gpt-5.6-sol", deep)
        self.assertEqual(target, "proxy")
        buried = compaction_payload(blocks=["x" * 501 + COMPACTION_TEXT])
        target, _ = clg.route_model("gpt-5.6-sol", buried)
        self.assertEqual(target, "proxy")

    def test_subagents_never_reach_anthropic_in_either_mode(self) -> None:
        for mode in ("anthropic", "gpt"):
            for model in ("claude-opus-5", "claude-sonnet-5", "gpt-5.6-sol"):
                with self.subTest(main_chat=mode, model=model):
                    self._with_main_chat(mode)
                    target, effective = clg.route_model(model, proxy_payload(model))
                    self.assertEqual(target, "proxy")
                    self.assertTrue(effective.startswith(clg.GPT_PREFIX), effective)


class CompactionFallbackTests(unittest.TestCase):
    """Une compaction ne doit jamais échouer : Anthropic d'abord, GPT en dernier."""

    def _stubs(self, anthropic_status: int):
        class AnthropicStub(QuietHandler):
            calls = 0

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                type(self).calls += 1
                self.send_response(anthropic_status)
                self.send_header("Content-Length", "9")
                self.end_headers()
                self.wfile.write(b"anthropic")

        class ProxyStub(QuietHandler):
            calls = 0
            models: list[str] = []

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                type(self).calls += 1
                type(self).models.append(json.loads(raw)["model"])
                self.send_response(200)
                self.send_header("Content-Length", "5")
                self.end_headers()
                self.wfile.write(b"proxy")

        return AnthropicStub, ProxyStub

    def _run(self, anthropic_status: int, payload: dict, token: str | None = "t"):
        AnthropicStub, ProxyStub = self._stubs(anthropic_status)
        original = clg.oauth_token
        clg.oauth_token = lambda: token
        try:
            with StubServer(ProxyStub) as proxy, StubServer(AnthropicStub) as anthropic, RunningRouter(
                [{"name": "a", "port": proxy.port}], anthropic_port=anthropic.port
            ) as router:
                status, body = request(router.port, payload)
        finally:
            clg.oauth_token = original
        return status, body, AnthropicStub, ProxyStub

    def test_429_retries_anthropic_then_degrades_to_gpt(self) -> None:
        status, body, anthropic, proxy = self._run(429, compaction_payload("gpt-5.6-sol"))
        self.assertEqual((status, body), (200, b"proxy"))
        self.assertEqual(anthropic.calls, clg.COMPACTION_ANTHROPIC_ATTEMPTS)
        self.assertEqual(proxy.models, ["gpt-5.6-sol"])

    def test_529_overloaded_also_degrades(self) -> None:
        status, body, anthropic, proxy = self._run(529, compaction_payload("claude-opus-5"))
        self.assertEqual((status, body), (200, b"proxy"))
        self.assertEqual(anthropic.calls, clg.COMPACTION_ANTHROPIC_ATTEMPTS)
        self.assertEqual(proxy.models, [clg.INHERITED])

    def test_401_is_never_masked_by_a_fallback(self) -> None:
        status, _, anthropic, proxy = self._run(401, compaction_payload("gpt-5.6-sol"))
        self.assertEqual(status, 401)
        self.assertEqual(anthropic.calls, 1)
        self.assertEqual(proxy.calls, 0)

    def test_successful_compaction_never_touches_the_proxy(self) -> None:
        status, body, anthropic, proxy = self._run(200, compaction_payload("gpt-5.6-sol"))
        self.assertEqual((status, body), (200, b"anthropic"))
        self.assertEqual(anthropic.calls, 1)
        self.assertEqual(proxy.calls, 0)

    def test_missing_oauth_degrades_instead_of_502(self) -> None:
        status, body, anthropic, proxy = self._run(
            200, compaction_payload("gpt-5.6-sol"), token=None
        )
        self.assertEqual((status, body), (200, b"proxy"))
        self.assertEqual(anthropic.calls, 0)
        self.assertEqual(proxy.models, ["gpt-5.6-sol"])

    def test_main_chat_429_is_passed_through_not_degraded(self) -> None:
        """Le repli est réservé à la compaction ; le chat principal voit son 429."""
        original = clg.MAIN_CHAT_UPSTREAM
        clg.MAIN_CHAT_UPSTREAM = "anthropic"
        try:
            status, _, anthropic, proxy = self._run(429, main_payload())
        finally:
            clg.MAIN_CHAT_UPSTREAM = original
        self.assertEqual(status, 429)
        self.assertEqual(anthropic.calls, 1)
        self.assertEqual(proxy.calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
