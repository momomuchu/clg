#!/usr/bin/env python3
"""Behavior locks for clg's concurrency-aware local router."""

from __future__ import annotations

import concurrent.futures
import http.client
import importlib.machinery
import importlib.util
import json
import pathlib
import socket
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


class FakeMonotonicClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


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
        monotonic_clock: object | None = None,
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
            monotonic_clock=monotonic_clock,
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


# Forme réelle observée dans clg-router-shared.log : un bloc de transport, puis le bloc
# d'identité, puis les instructions communes. La phrase "You are an interactive agent"
# apparaît dans les DEUX familles et ne peut pas servir de discriminant.
MAIN_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
SUBAGENT_IDENTITY = "You are a Claude agent, built on Anthropic's Claude Code."
SHARED_TAIL = "\nYou are an interactive agent that helps users with software engineering tasks."
BILLING_BLOCK = "x-anthropic-billing-header: cc_version=2.1.238.abc"
COMPACTION_IDENTITY = "You are a helpful AI assistant tasked with summarizing conversations."


def blocks(*texts: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text} for text in texts]


def main_payload(model: str = "claude-opus-5[1m]") -> dict[str, object]:
    return {
        "model": model,
        "system": blocks(BILLING_BLOCK, MAIN_IDENTITY, SHARED_TAIL),
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

    def test_generation_classifier_uses_path_component_only(self) -> None:
        handler = object.__new__(clg.RouterHandler)
        handler.command = "POST"
        handler.path = "/v1/messages?beta=true"
        self.assertTrue(handler._is_generation_request())

        handler.path = "/v1/messages"
        self.assertTrue(handler._is_generation_request())

        handler.command = "GET"
        handler.path = "/v1/models"
        self.assertFalse(handler._is_generation_request())

    def test_query_generation_websocket_403_retries_and_preserves_upstream_path(self) -> None:
        class Response:
            def __init__(self, status: int, body: bytes) -> None:
                self.status = status
                self.body = body

            def getheader(self, name: str) -> str | None:
                return str(len(self.body)) if name == "Content-Length" else None

            def getheaders(self) -> list[tuple[str, str]]:
                return [("Content-Length", str(len(self.body)))]

            def read(self, _amount: int | None = None) -> bytes:
                body, self.body = self.body, b""
                return body

            def read1(self, _amount: int) -> bytes:
                return self.read()

        class RecordingConnection:
            requests: list[tuple[str, str]] = []
            attempts = 0

            def __init__(self, _port: int) -> None:
                type(self).attempts += 1
                self.attempt = type(self).attempts

            def request(self, method: str, path: str, **_kwargs: object) -> None:
                type(self).requests.append((method, path))

            def getresponse(self) -> Response:
                if self.attempt == 1:
                    return Response(403, b"WebSocket upgrade was rejected")
                return Response(200, b"retried-query-request")

            def close(self) -> None:
                pass

        clock = FakeMonotonicClock()
        with RunningRouter(
            [{"name": "a", "port": 1}],
            proxy_connection_factory=RecordingConnection,
            retry_sleeper=lambda _delay: clock.advance(clg.COOLDOWN_SECONDS),
            monotonic_clock=clock,
        ) as router:
            status, body = request(router.port, proxy_payload(), "/v1/messages?beta=true")

        self.assertEqual((status, body), (200, b"retried-query-request"))
        self.assertEqual(RecordingConnection.requests, [
            ("POST", "/v1/messages?beta=true"),
            ("POST", "/v1/messages?beta=true"),
        ])

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
        clock = FakeMonotonicClock()

        def advance_cooldown(delay: float) -> None:
            delays.append(delay)
            clock.advance(clg.COOLDOWN_SECONDS)

        with StubServer(TransientRejectingStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}],
            retry_sleeper=advance_cooldown,
            monotonic_clock=clock,
        ) as router:
            status, body = request(router.port, proxy_payload())

        self.assertEqual((status, body), (200, b"retried-same-upstream"))
        self.assertEqual(TransientRejectingStub.requests, 2)
        self.assertEqual(delays, [clg.RETRY_MAX_SLEEP_SECONDS])

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
        clock = FakeMonotonicClock()
        with StubServer(TransientRejectingStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}], monotonic_clock=clock
        ) as router:
            def inspect_backoff(_delay: float) -> None:
                inflight_during_backoff.append(router.server.scheduler.snapshot()[0]["inflight"])
                clock.advance(clg.COOLDOWN_SECONDS)

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

    def test_websocket_403_cools_upstream_while_concurrent_request_uses_other_upstream(self) -> None:
        # Regression lock: after a WebSocket 403 on /v1/messages, an independent
        # generation request cannot immediately reuse the saturated account.
        class RejectingStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if type(self).requests == 1:
                    body = b"WebSocket upgrade was rejected"
                    self.send_response(403)
                else:
                    body = b"recovered"
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class SuccessStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        clock = FakeMonotonicClock()
        with StubServer(RejectingStub) as first, StubServer(SuccessStub) as second, RunningRouter(
            [{"name": "a", "port": first.port}, {"name": "b", "port": second.port}],
            initial_permits=1,
            monotonic_clock=clock,
        ) as router:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                rejected = pool.submit(request, router.port, proxy_payload())
                while RejectingStub.requests == 0:
                    time.sleep(0.001)
                status, body = request(router.port, proxy_payload())
                states = {state["name"]: state for state in router.server.scheduler.snapshot()}
                clock.advance(clg.COOLDOWN_SECONDS)
                with router.server.scheduler._condition:
                    router.server.scheduler._condition.notify_all()
                rejected.result(timeout=2)

        self.assertEqual((status, body), (200, b"ok"))
        self.assertEqual(RejectingStub.requests, 1)
        self.assertGreater(states["a"]["cooldown_remaining"], 0)
        self.assertGreaterEqual(SuccessStub.requests, 1)

    def test_all_cooling_waits_until_earliest_expiry(self) -> None:
        clock = FakeMonotonicClock()
        scheduler = clg.UpstreamScheduler(
            [{"name": "a", "port": 1}, {"name": "b", "port": 2}],
            initial_permits=1,
            monotonic_clock=clock,
        )
        scheduler._upstreams[0].cooldown_until = 3.0
        scheduler._upstreams[1].cooldown_until = 5.0
        selected: list[object] = []
        started = threading.Event()

        def wait_for_capacity() -> None:
            started.set()
            selected.append(scheduler.acquire(timeout=1))

        thread = threading.Thread(target=wait_for_capacity)
        thread.start()
        self.assertTrue(started.wait(timeout=1))
        time.sleep(0.02)
        self.assertTrue(thread.is_alive())
        clock.advance(3.0)
        with scheduler._condition:
            scheduler._condition.notify_all()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(selected[0].name, "a")
        scheduler.release(selected[0])

    def test_all_cooling_wakes_at_earliest_expiry_without_notification(self) -> None:
        scheduler = clg.UpstreamScheduler(
            [{"name": "a", "port": 1}, {"name": "b", "port": 2}], initial_permits=1
        )
        now = time.monotonic()
        scheduler._upstreams[0].cooldown_until = now + 0.05
        scheduler._upstreams[1].cooldown_until = now + 0.20

        started = time.monotonic()
        selected = scheduler.acquire(timeout=0.5)
        elapsed = time.monotonic() - started

        self.assertEqual(selected.name, "a")
        self.assertLess(elapsed, 0.15)
        scheduler.release(selected)

    def test_pinned_cooling_upstream_waits_for_its_expiry(self) -> None:
        clock = FakeMonotonicClock()
        scheduler = clg.UpstreamScheduler(
            [{"name": "a", "port": 1}, {"name": "b", "port": 2}],
            initial_permits=1,
            monotonic_clock=clock,
        )
        scheduler._upstreams[0].cooldown_until = 4.0
        selected: list[object] = []

        thread = threading.Thread(target=lambda: selected.append(scheduler.acquire(only_name="a", timeout=1)))
        thread.start()
        time.sleep(0.02)
        self.assertTrue(thread.is_alive())
        clock.advance(4.0)
        with scheduler._condition:
            scheduler._condition.notify_all()
        thread.join(timeout=1)

        self.assertFalse(thread.is_alive())
        self.assertEqual(selected[0].name, "a")
        scheduler.release(selected[0])

    def test_cooldown_expiry_restores_normal_selection(self) -> None:
        clock = FakeMonotonicClock()
        scheduler = clg.UpstreamScheduler(
            [{"name": "a", "port": 1}, {"name": "b", "port": 2}],
            initial_permits=1,
            monotonic_clock=clock,
        )
        scheduler._upstreams[0].cooldown_until = 2.0
        first = scheduler.acquire(timeout=0)
        self.assertEqual(first.name, "b")
        scheduler.release(first)
        clock.advance(2.0)
        restored = scheduler.acquire(only_name="a", timeout=0)
        self.assertEqual(restored.name, "a")
        scheduler.release(restored)

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
        # 429 is deliberately excluded: it means the account has no capacity, not that
        # the request is bad, so it switches upstream instead of passing through.
        # Its contract is locked by test_429_switches_to_another_upstream below.
        for expected_status in (500, 502):
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

    def _main_chat_with_anthropic_status(self, status: int, anthropic_body: bytes = b""):
        """Drive one main-chat request whose Anthropic upstream answers `status`."""
        class AnthropicStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(status)
                self.send_header("Content-Length", str(len(anthropic_body)))
                self.end_headers()
                if anthropic_body:
                    self.wfile.write(anthropic_body)

        class ProxyStub(QuietHandler):
            requests = 0
            seen_models: list = []

            def do_POST(self) -> None:
                type(self).requests += 1
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                try:
                    type(self).seen_models.append(json.loads(raw).get("model"))
                except Exception:
                    type(self).seen_models.append(None)
                body = b"served by proxy"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        ProxyStub.seen_models = []
        original = clg.oauth_token
        clg.oauth_token = lambda: "test-token"
        try:
            with StubServer(ProxyStub) as proxy, StubServer(AnthropicStub) as anthropic, RunningRouter(
                [{"name": "a", "port": proxy.port}], anthropic_port=anthropic.port
            ) as router:
                result = request(router.port, main_payload())
        finally:
            clg.oauth_token = original
        return result, AnthropicStub.requests, ProxyStub.requests, ProxyStub.seen_models

    def test_main_chat_falls_back_to_proxy_when_anthropic_is_unavailable(self) -> None:
        # The founder's actual need: when Anthropic is down or saturated he had to switch
        # the session model by hand. The router now replays the same main-chat request on
        # the proxy instead, and rewrites the model to one the proxy can serve.
        for status in (429, 500, 502, 503, 529):
            with self.subTest(status=status):
                result, anthropic_hits, proxy_hits, models = self._main_chat_with_anthropic_status(status)
                self.assertEqual(result, (200, b"served by proxy"))
                self.assertEqual(anthropic_hits, 1)
                self.assertEqual(proxy_hits, 1)
                self.assertEqual(models, [clg.SOL])

    def test_main_chat_does_not_fall_back_on_auth_or_client_error(self) -> None:
        # A dead OAuth token and a malformed request must stay visible. Falling back would
        # hide the re-auth prompt behind a silently different model.
        for status in (400, 401, 404, 413):
            with self.subTest(status=status):
                result, anthropic_hits, proxy_hits, _ = self._main_chat_with_anthropic_status(status)
                self.assertEqual(result[0], status)
                self.assertEqual(anthropic_hits, 1)
                self.assertEqual(proxy_hits, 0)

    def test_main_chat_falls_back_when_anthropic_is_unreachable(self) -> None:
        class ProxyStub(QuietHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = b"served by proxy"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        original = clg.oauth_token
        clg.oauth_token = lambda: "test-token"
        try:
            with StubServer(ProxyStub) as proxy:
                # Bind then immediately free a port so the connection is refused.
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", 0))
                    dead_port = probe.getsockname()[1]
                with RunningRouter(
                    [{"name": "a", "port": proxy.port}], anthropic_port=dead_port
                ) as router:
                    result = request(router.port, main_payload())
        finally:
            clg.oauth_token = original
        self.assertEqual(result, (200, b"served by proxy"))

    def test_main_chat_reaches_anthropic_on_every_claude_model(self) -> None:
        # THE leak, measured: ~62k main-chat requests reached GPT because the model
        # family was tested before the request identity. On any Claude model other than
        # claude-opus-5 the router never even asked whether this was the main chat.
        for model in ("claude-opus-5", "claude-opus-5[1m]", "claude-fable-5",
                      "claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5-20251001"):
            with self.subTest(model=model):
                target, effective = clg.route_model(model, main_payload(model))
                self.assertEqual(target, "anthropic")
                self.assertEqual(effective, model.removesuffix("[1m]"))

    def test_compaction_reaches_anthropic_on_every_claude_model(self) -> None:
        # Compaction ships the whole conversation in one request. Sent to GPT through
        # the proxy it is translated wholesale into a smaller window — the founder's
        # "extremely slow compaction". Its prompt is verified against the Claude Code
        # 2.1.238 binary.
        for model in ("claude-opus-5", "claude-fable-5", "claude-sonnet-5"):
            with self.subTest(model=model):
                payload = {
                    "model": model,
                    "system": blocks(BILLING_BLOCK, COMPACTION_IDENTITY),
                    "messages": [],
                }
                self.assertEqual(clg.route_model(model, payload), ("anthropic", model))

    def test_subagent_never_reaches_anthropic_even_carrying_main_chat_phrasing(self) -> None:
        # The reverse leak, measured: 164 subagent requests reached the Anthropic quota
        # because "You are an interactive agent" appears in BOTH families. Identity is
        # anchored at the first substantive block and the deny list wins.
        for identity in ("You are a Claude agent, built on Anthropic's Claude Code.",
                         "You are an agent for Claude Code, Anthropic's official CLI.",
                         "You are a subagent spawned by a workflow orchestrator.",
                         "A user kicked off a Claude Code agent to do a code review."):
            with self.subTest(identity=identity[:40]):
                payload = {
                    "model": "claude-opus-5",
                    "system": blocks(BILLING_BLOCK, identity, SHARED_TAIL),
                    "messages": [],
                }
                self.assertEqual(clg.route_model("claude-opus-5", payload), ("proxy", clg.SOL))

    def test_bare_alias_main_chat_stays_on_the_proxy_by_design(self) -> None:
        # Known, deliberate residual gap. A bare alias ("fable", "opus") cannot be sent
        # to api.anthropic.com, which needs a full model ID, and resolving one would mean
        # guessing a substitution the founder did not ask for. Measured over ~250k logged
        # requests: Claude Code never sends a bare alias, always a full ID. If that ever
        # changes, this test is the tripwire — flip it and add an explicit alias map.
        for alias in ("fable", "opus"):
            with self.subTest(alias=alias):
                payload = {"model": alias, "system": blocks(BILLING_BLOCK, MAIN_IDENTITY), "messages": []}
                self.assertEqual(clg.route_model(alias, payload), ("proxy", clg.SOL))

    def test_non_claude_model_never_reaches_anthropic(self) -> None:
        # Anthropic cannot serve a GPT model. Even with a perfect main-chat identity,
        # a non-Claude model must stay on the proxy rather than produce a 4xx upstream.
        payload = {"model": "gpt-5.6-sol", "system": blocks(BILLING_BLOCK, MAIN_IDENTITY), "messages": []}
        self.assertEqual(clg.route_model("gpt-5.6-sol", payload), ("proxy", "gpt-5.6-sol"))

    def test_retired_model_alias_routes_to_luna(self) -> None:
        self.assertEqual(
            clg.route_model("grok-4.5", proxy_payload("grok-4.5")),
            ("proxy", clg.LUNA),
        )

    def test_retired_model_alias_reaches_luna_only_upstream_through_acquire(self) -> None:
        class LunaOnlyStub(QuietHandler):
            received_models: list[str] = []

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                type(self).received_models.append(payload["model"])
                body = b"served by luna"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with StubServer(LunaOnlyStub) as upstream, RunningRouter([
            {"name": "luna", "port": upstream.port, "models": [clg.LUNA]},
        ]) as router:
            result = request(router.port, proxy_payload("grok-4.5"))

        # The only upstream is eligible only if RouterHandler rewrote the serialized
        # body before _relay_proxy passed its model into scheduler.acquire().
        self.assertEqual(result, (200, b"served by luna"))
        self.assertEqual(LunaOnlyStub.received_models, [clg.LUNA])

    def test_retired_model_alias_does_not_change_main_chat_routing(self) -> None:
        model = "claude-haiku-4-5-20251001"
        self.assertEqual(clg.route_model(model, main_payload(model)), ("anthropic", model))

    def test_retired_model_alias_does_not_override_agent_card_routing(self) -> None:
        original = clg.AGENT_ROUTING
        clg.AGENT_ROUTING = {"bb-quality": "deepseek-chat"}
        try:
            payload = {
                "model": "grok-4.5",
                "system": blocks(BILLING_BLOCK, SUBAGENT_IDENTITY, "# bb-quality\n\n## Mission\n\nReview it."),
                "messages": [],
            }
            self.assertEqual(clg.route_model("grok-4.5", payload), ("proxy", "deepseek-chat"))
        finally:
            clg.AGENT_ROUTING = original

    def test_unknown_model_passes_through_unchanged(self) -> None:
        self.assertEqual(
            clg.route_model("future-model", proxy_payload("future-model")),
            ("proxy", "future-model"),
        )

    def test_unrecognised_identity_fails_safe_to_the_proxy(self) -> None:
        # Fail-safe direction is preserved: an unknown or missing identity costs
        # quality, never the founder's Anthropic quota.
        for system in ([], blocks(BILLING_BLOCK), blocks(BILLING_BLOCK, "Some future prompt shape"),
                       blocks("", "   ")):
            with self.subTest(system=str(system)[:40]):
                payload = {"model": "claude-opus-5", "system": system, "messages": []}
                self.assertEqual(clg.route_model("claude-opus-5", payload), ("proxy", clg.SOL))

    def test_identity_block_skips_transport_blocks_only(self) -> None:
        self.assertEqual(clg.identity_block({"system": blocks(BILLING_BLOCK, MAIN_IDENTITY)}), MAIN_IDENTITY)
        self.assertEqual(clg.identity_block({"system": MAIN_IDENTITY}), MAIN_IDENTITY)
        self.assertEqual(clg.identity_block({"system": blocks(BILLING_BLOCK)}), "")
        self.assertEqual(clg.identity_block({}), "")

    def test_agent_card_routing_sends_one_role_to_another_provider(self) -> None:
        # The tier map is frozen into Claude Code's environment at launch, so a tier
        # cannot change mid-session. The agent card name arrives on every request, at
        # system block index 2 in all 58 885 logged occurrences. That makes it the only
        # place where one role can be sent to a different provider without a relaunch.
        original = clg.AGENT_ROUTING
        clg.AGENT_ROUTING = {"bb-js-mapper": "grok-4.5", "bb-quality": "deepseek-chat"}
        try:
            def payload(card, model="gpt-5.6-terra"):
                return {"model": model,
                        "system": blocks(BILLING_BLOCK, SUBAGENT_IDENTITY, f"# {card}\n\n## Mission\n\nDo the thing."),
                        "messages": []}
            self.assertEqual(clg.route_model("gpt-5.6-terra", payload("bb-js-mapper")),
                             ("proxy", "grok-4.5"))
            self.assertEqual(clg.route_model("gpt-5.6-terra", payload("bb-quality")),
                             ("proxy", "deepseek-chat"))
            # Unmapped card: untouched.
            self.assertEqual(clg.route_model("gpt-5.6-terra", payload("bb-exploit")),
                             ("proxy", "gpt-5.6-terra"))
            # The main chat is never rerouted by an agent map — identity wins.
            self.assertEqual(clg.route_model("claude-opus-5", main_payload("claude-opus-5")),
                             ("anthropic", "claude-opus-5"))
        finally:
            clg.AGENT_ROUTING = original

    def test_agent_card_name_extraction(self) -> None:
        self.assertEqual(clg.agent_card_name(
            {"system": blocks(BILLING_BLOCK, SUBAGENT_IDENTITY, "# bb-exploit\n\n## Mission\n\nx")}),
            "bb-exploit")
        # No card block at all.
        self.assertIsNone(clg.agent_card_name({"system": blocks(BILLING_BLOCK, MAIN_IDENTITY)}))
        # A markdown heading that is prose, not a card name, must not be read as one.
        self.assertIsNone(clg.agent_card_name({"system": blocks("# Some heading with spaces")}))
        self.assertIsNone(clg.agent_card_name({}))

    def test_health_check_uses_the_account_s_own_declared_models(self) -> None:
        # Before this, the check demanded SOL/TERRA/LUNA from EVERY account. Repointing a
        # tier at grok-4.5 therefore made all the GPT accounts unhealthy at once, which is
        # what kept a heterogeneous provider out of the fleet.
        class Catalog(QuietHandler):
            body = b'{"data":[{"id":"grok-4.5"},{"id":"grok-composer-2.5-fast"}]}'

            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(type(self).body)))
                self.end_headers()
                self.wfile.write(type(self).body)

        with StubServer(Catalog) as stub:
            # Declares grok: healthy on its own catalogue.
            self.assertTrue(clg.proxy_healthy(stub.port, ["grok"]))
            # Declares gpt: this account does not serve it, so not healthy.
            self.assertFalse(clg.proxy_healthy(stub.port, ["gpt-"]))
            # Declares nothing: falls back to the three tier models, absent here.
            self.assertFalse(clg.proxy_healthy(stub.port))

    def test_account_declaring_models_is_only_selected_for_those_models(self) -> None:
        # A Grok or DeepSeek account cannot serve GPT models, and /v1/models does not
        # say so: the catalogue advertises what the proxy SUPPORTS, not what the account
        # is entitled to. Measured 2026-08-21: a Grok-authenticated account advertises
        # gpt-5.6-sol and answers 429 for it. Without this filter the router would read
        # that refusal as saturation and park a healthy account.
        sched = clg.UpstreamScheduler([
            {"name": "gpt-a", "port": 1, "models": ["gpt-"]},
            {"name": "grok", "port": 2, "models": ["grok"]},
        ])
        self.assertEqual(sched.select_unlimited(model="gpt-5.6-sol").name, "gpt-a")
        self.assertEqual(sched.select_unlimited(model="grok-4.5").name, "grok")
        # Prefix matching: one declaration covers a family.
        self.assertEqual(sched.select_unlimited(model="grok-composer-2.5-fast").name, "grok")

    def test_account_without_declared_models_still_serves_everything(self) -> None:
        # Backward compatibility: the existing registry has no `models` key and must keep
        # behaving exactly as before this change.
        sched = clg.UpstreamScheduler([{"name": "a", "port": 1}, {"name": "b", "port": 2}])
        for model in ("gpt-5.6-sol", "grok-4.5", "anything-at-all", None):
            with self.subTest(model=model):
                self.assertIn(sched.select_unlimited(model=model).name, {"a", "b"})

    def test_no_account_for_a_model_is_a_named_error_not_a_random_attempt(self) -> None:
        # Configuration error, not saturation. Trying a random account and reading its
        # refusal as a rate limit is how a healthy account gets parked for nothing.
        sched = clg.UpstreamScheduler([{"name": "grok", "port": 1, "models": ["grok"]}])
        with self.assertRaises(ValueError) as ctx:
            sched.select_unlimited(model="gpt-5.6-sol")
        self.assertIn("gpt-5.6-sol", str(ctx.exception))

    def test_acquire_threads_the_model_filter_through(self) -> None:
        # Regression lock for a real defect: `model` was added to acquire()'s signature but
        # never passed to _eligible_upstreams INSIDE acquire. The unit test called
        # _eligible_upstreams directly, so it passed while the live path routed grok-4.5
        # onto GPT accounts. Test the door agents actually walk through, not the one next
        # to it.
        sched = clg.UpstreamScheduler([
            {"name": "gpt-a", "port": 1, "models": ["gpt-"]},
            {"name": "grok", "port": 2, "models": ["grok"]},
        ])
        got = sched.acquire(model="grok-4.5", timeout=2)
        try:
            self.assertEqual(got.name, "grok")
        finally:
            sched.release(got)
        got = sched.acquire(model="gpt-5.6-sol", timeout=2)
        try:
            self.assertEqual(got.name, "gpt-a")
        finally:
            sched.release(got)

    def test_failover_only_considers_accounts_serving_the_model(self) -> None:
        # The 429 failover must not "fail over" onto an account that cannot serve the
        # request at all — that would turn one exhausted account into two parked ones.
        sched = clg.UpstreamScheduler([
            {"name": "gpt-a", "port": 1, "models": ["gpt-"]},
            {"name": "gpt-b", "port": 2, "models": ["gpt-"]},
            {"name": "grok", "port": 3, "models": ["grok"]},
        ])
        eligible = sched._eligible_upstreams(avoid_name="gpt-a", prefer_not_name=None,
                                             only_name=None, model="gpt-5.6-sol")
        self.assertEqual({u.name for u in eligible}, {"gpt-b"})

    def test_429_switches_to_another_upstream_and_cools_the_exhausted_one(self) -> None:
        # A 429 means this account is out of capacity. The caller's work must survive
        # it: the router parks the exhausted account and serves the request from
        # another one. Before this lock, a 429 was returned to the caller and killed
        # the request even while a second account had capacity.
        class ExhaustedStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = b'{"error":"rate limit reached"}'
                self.send_response(429)
                self.send_header("Retry-After", "60")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class SuccessStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = b"served elsewhere"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with StubServer(ExhaustedStub) as first, StubServer(SuccessStub) as second, RunningRouter(
            [{"name": "a", "port": first.port}, {"name": "b", "port": second.port}]
        ) as router:
            status, body = request(router.port, proxy_payload())
            states = {state["name"]: state for state in router.server.scheduler.snapshot()}

        self.assertEqual((status, body), (200, b"served elsewhere"))
        self.assertEqual(ExhaustedStub.requests, 1)
        self.assertEqual(SuccessStub.requests, 1)
        self.assertLess(states["a"]["permits"], 6)

    def test_parse_retry_after_reads_seconds_and_ignores_unusable_values(self) -> None:
        self.assertEqual(clg.parse_retry_after("60"), 60.0)
        self.assertEqual(clg.parse_retry_after(" 1.5 "), 1.5)
        self.assertIsNone(clg.parse_retry_after(None))
        self.assertIsNone(clg.parse_retry_after(""))
        self.assertIsNone(clg.parse_retry_after("0"))
        self.assertIsNone(clg.parse_retry_after("-5"))
        # HTTP-date form is deliberately ignored so clock skew cannot park an account.
        self.assertIsNone(clg.parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT"))

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
        subagent_payload = {
            "model": "claude-opus-5",
            "system": [{"type": "text", "text": "<Agent_Prompt>\n<Role>\nYou are implementation-executor"}],
            "messages": [],
        }
        self.assertEqual(
            clg.route_model("claude-opus-5", subagent_payload),
            ("proxy", clg.SOL),
        )
        self.assertEqual(
            clg.route_model("claude-fable-1", proxy_payload("claude-fable-1")),
            ("proxy", clg.SOL),
        )
        main_chat_payload = {**subagent_payload, "system": blocks(BILLING_BLOCK, MAIN_IDENTITY)}
        self.assertEqual(
            clg.route_model("claude-opus-5", main_chat_payload),
            ("anthropic", "claude-opus-5"),
        )

    def test_main_preserves_account_model_eligibility_when_starting_router(self) -> None:
        original_argv = sys.argv
        original_load_registry = clg.load_registry
        original_has_auth = clg.has_auth
        original_start_proxy = clg.start_proxy
        original_start_router = clg.start_router
        original_read_oauth = clg.read_oauth
        original_load_routing = clg.load_routing
        original_resolve_claude_bin = clg.resolve_claude_bin
        original_execve = clg.os.execve
        started: list[tuple[str, int, list[dict[str, object]]]] = []

        def capture_router(
            name: str,
            port: int,
            upstreams: list[dict[str, object]],
            **_kwargs: object,
        ) -> bool:
            started.append((name, port, upstreams))
            return True

        def stop_before_launch(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("stop before Claude launch")

        clg.load_registry = lambda: {
            "luna": {
                "dir": "/tmp/ccp-luna",
                "port": 18765,
                "models": [clg.LUNA],
            },
        }
        clg.has_auth = lambda _account: True
        clg.start_proxy = lambda _name, _account: True
        clg.start_router = capture_router
        clg.read_oauth = lambda: {}
        clg.load_routing = lambda _account=None: {
            "main_model_env": "claude-opus-5[1m]",
            "tiers": {"opus": clg.SOL, "sonnet": clg.TERRA, "haiku": clg.LUNA},
        }
        clg.resolve_claude_bin = lambda: "/fake/claude"
        clg.os.execve = stop_before_launch
        sys.argv = ["clg"]
        try:
            with self.assertRaisesRegex(RuntimeError, "stop before Claude launch"):
                clg.main()
        finally:
            sys.argv = original_argv
            clg.load_registry = original_load_registry
            clg.has_auth = original_has_auth
            clg.start_proxy = original_start_proxy
            clg.start_router = original_start_router
            clg.read_oauth = original_read_oauth
            clg.load_routing = original_load_routing
            clg.resolve_claude_bin = original_resolve_claude_bin
            clg.os.execve = original_execve

        self.assertEqual(
            started,
            [("shared", clg.SHARED_ROUTER_PORT, [{"name": "luna", "port": 18765, "models": [clg.LUNA]}])],
        )

    def test_health_reports_upstream_capacity(self) -> None:
        class UnusedStub(QuietHandler):
            pass

        with StubServer(UnusedStub) as first, StubServer(UnusedStub) as second, RunningRouter(
            [
                {"name": "a", "port": first.port, "models": [clg.SOL]},
                {"name": "b", "port": second.port, "models": [clg.TERRA, clg.LUNA]},
            ],
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
                {
                    "cooldown_remaining": 0.0,
                    "cooldown_until": 0.0,
                    "free": 3,
                    "inflight": 0,
                    "models": [clg.SOL],
                    "name": "a",
                    "permits": 3,
                    "port": first.port,
                    "successes": 0,
                },
                {
                    "cooldown_remaining": 0.0,
                    "cooldown_until": 0.0,
                    "free": 3,
                    "inflight": 0,
                    "models": [clg.TERRA, clg.LUNA],
                    "name": "b",
                    "permits": 3,
                    "port": second.port,
                    "successes": 0,
                },
            ],
        )

    def test_router_healthy_rejects_stale_model_eligibility(self) -> None:
        original_router_status = clg.router_status
        clg.router_status = lambda _port: {
            "router": "ok",
            "main": clg.MAIN_MODEL,
            "delegate_port": None,
            "pinned_upstream": None,
            "upstreams": [{"name": "sol", "port": 18765, "models": [clg.TERRA]}],
        }
        try:
            self.assertFalse(
                clg.router_healthy(
                    9999,
                    [{"name": "sol", "port": 18765, "models": [clg.SOL]}],
                )
            )
        finally:
            clg.router_status = original_router_status

    def test_websocket_403_waits_through_sixty_seconds_of_saturation(self) -> None:
        class SaturatedThenHealthyStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if clock.now < 60.0:
                    body = b"WebSocket upgrade was rejected"
                    self.send_response(403)
                else:
                    body = b"recovered-after-saturation"
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        delays: list[float] = []
        clock = FakeMonotonicClock()

        def advance_time(delay: float) -> None:
            delays.append(delay)
            clock.advance(delay)

        # Isolate the attempt-budget regression; cooldown cooperation has its own
        # integration lock below.
        original_cooldown = clg.COOLDOWN_SECONDS
        clg.COOLDOWN_SECONDS = 0.0
        try:
            with StubServer(SaturatedThenHealthyStub) as upstream, RunningRouter(
                [{"name": "a", "port": upstream.port}],
                retry_sleeper=advance_time,
                monotonic_clock=clock,
            ) as router:
                status, body = request(router.port, proxy_payload())
        finally:
            clg.COOLDOWN_SECONDS = original_cooldown

        self.assertEqual((status, body), (200, b"recovered-after-saturation"))
        self.assertGreater(SaturatedThenHealthyStub.requests, clg.MAX_PROXY_ATTEMPTS)
        self.assertLess(clock.now, 180.0)
        self.assertLess(len(delays), 10)
        self.assertTrue(all(delay <= 15.0 for delay in delays))

    def test_retry_waits_for_earliest_cooldown_when_all_upstreams_cool(self) -> None:
        class RejectingStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = b"WebSocket upgrade was rejected"
                self.send_response(403)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        class SuccessStub(QuietHandler):
            requests = 0

            def do_POST(self) -> None:
                type(self).requests += 1
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = b"earliest-cooldown-won"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        clock = FakeMonotonicClock()
        delays: list[float] = []

        def advance_time(delay: float) -> None:
            delays.append(delay)
            clock.advance(delay)

        with StubServer(RejectingStub) as first, StubServer(SuccessStub) as second, RunningRouter(
            [{"name": "a", "port": first.port}, {"name": "b", "port": second.port}],
            retry_sleeper=advance_time,
            monotonic_clock=clock,
        ) as router:
            # The first selection is a. Its rejection cools it until t=20; b is
            # already cooling until t=3, so retry must wait exactly for b, not .4s.
            router.server.scheduler._upstreams[1].cooldown_until = 3.0
            status, body = request(router.port, proxy_payload())

        self.assertEqual((status, body), (200, b"earliest-cooldown-won"))
        self.assertEqual(RejectingStub.requests, 1)
        self.assertEqual(SuccessStub.requests, 1)
        self.assertEqual(delays, [3.0])

    def test_websocket_403_returns_once_after_wall_clock_budget(self) -> None:
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

        clock = FakeMonotonicClock()
        delays: list[float] = []

        def advance_time(delay: float) -> None:
            delays.append(delay)
            clock.advance(delay)

        with StubServer(RejectingStub) as upstream, RunningRouter(
            [{"name": "a", "port": upstream.port}],
            retry_sleeper=advance_time,
            monotonic_clock=clock,
        ) as router:
            status, body = request(router.port, proxy_payload())

        self.assertEqual(status, 403)
        self.assertEqual(body, f"WebSocket upgrade was rejected attempt-{RejectingStub.requests}".encode())
        self.assertGreater(RejectingStub.requests, clg.MAX_PROXY_ATTEMPTS)
        self.assertEqual(clock.now, 180.0)
        self.assertTrue(all(delay <= 15.0 for delay in delays))
        self.assertLess(len(delays), 25)

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

    def test_uncommitted_stream_error_retries_then_returns_success(self) -> None:
        # /v1/messages: a reset before response headers must transparently retry.
        class SuccessResponse:
            status = 200
            def getheaders(self) -> list[tuple[str, str]]: return []
            def read1(self, _size: int) -> bytes: return b""

        class FirstStreamErrorThenSuccess:
            attempts = 0

            def __init__(self, _port: int) -> None:
                type(self).attempts += 1
                self.attempt = type(self).attempts

            def request(self, *_args: object, **_kwargs: object) -> None:
                if self.attempt == 1:
                    raise clg.UpstreamStreamError("connection reset")

            def getresponse(self) -> SuccessResponse:
                return SuccessResponse()

            def close(self) -> None: pass

        delays: list[float] = []
        with RunningRouter(
            [{"name": "a", "port": 1}],
            proxy_connection_factory=FirstStreamErrorThenSuccess,
            retry_sleeper=delays.append,
        ) as router:
            status, body = request(router.port, proxy_payload())

        self.assertEqual((status, body), (200, b""))
        self.assertEqual(FirstStreamErrorThenSuccess.attempts, 2)
        self.assertEqual(delays, [clg.RETRY_BASE_SECONDS])

    def test_uncommitted_connection_error_retries_then_returns_success(self) -> None:
        class SuccessResponse:
            status = 200
            def getheaders(self) -> list[tuple[str, str]]: return []
            def read1(self, _size: int) -> bytes: return b""

        class FirstConnectionErrorThenSuccess:
            attempts = 0

            def __init__(self, _port: int) -> None:
                type(self).attempts += 1
                self.attempt = type(self).attempts

            def request(self, *_args: object, **_kwargs: object) -> None:
                if self.attempt == 1:
                    raise ConnectionResetError("connection reset")

            def getresponse(self) -> SuccessResponse:
                return SuccessResponse()

            def close(self) -> None: pass

        delays: list[float] = []
        with RunningRouter(
            [{"name": "a", "port": 1}],
            proxy_connection_factory=FirstConnectionErrorThenSuccess,
            retry_sleeper=delays.append,
        ) as router:
            status, body = request(router.port, proxy_payload())

        self.assertEqual((status, body), (200, b""))
        self.assertEqual(FirstConnectionErrorThenSuccess.attempts, 2)
        self.assertEqual(delays, [clg.RETRY_BASE_SECONDS])

    def test_committed_stream_error_is_not_retried(self) -> None:
        # Negative control: after /v1/messages headers/body commit, retrying corrupts the response.
        class PartialResponse:
            status = 200
            def getheaders(self) -> list[tuple[str, str]]: return []
            def read1(self, _size: int) -> bytes:
                raise ConnectionResetError("connection reset")

        class AlwaysStreamError:
            attempts = 0

            def __init__(self, _port: int) -> None:
                type(self).attempts += 1

            def request(self, *_args: object, **_kwargs: object) -> None: pass
            def getresponse(self) -> PartialResponse: return PartialResponse()
            def close(self) -> None: pass

        delays: list[float] = []
        with RunningRouter(
            [{"name": "a", "port": 1}],
            proxy_connection_factory=AlwaysStreamError,
            retry_sleeper=delays.append,
        ) as router:
            status, body = request(router.port, proxy_payload())

        self.assertEqual((status, body), (200, b""))
        self.assertEqual(AlwaysStreamError.attempts, 1)
        self.assertEqual(delays, [])

    def test_persistent_uncommitted_stream_errors_exhaust_wall_clock_budget(self) -> None:
        class AlwaysStreamError:
            attempts = 0

            def __init__(self, _port: int) -> None:
                type(self).attempts += 1

            def request(self, *_args: object, **_kwargs: object) -> None:
                raise clg.UpstreamStreamError("connection reset")

            def close(self) -> None: pass

        clock = FakeMonotonicClock()
        delays: list[float] = []

        def advance_time(delay: float) -> None:
            delays.append(delay)
            clock.advance(delay)

        with RunningRouter(
            [{"name": "a", "port": 1}],
            proxy_connection_factory=AlwaysStreamError,
            retry_sleeper=advance_time,
            monotonic_clock=clock,
        ) as router:
            status, _body = request(router.port, proxy_payload())

        self.assertEqual(status, 502)
        self.assertGreater(AlwaysStreamError.attempts, clg.MAX_PROXY_ATTEMPTS)
        self.assertEqual(clock.now, 180.0)
        self.assertTrue(all(delay <= 15.0 for delay in delays))

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

        clock = FakeMonotonicClock()

        def advance_cooldown(_delay: float) -> None:
            clock.advance(clg.COOLDOWN_SECONDS)

        with StubServer(UnpinnedStub) as first, StubServer(PinnedStub) as second, RunningRouter(
            [{"name": "a", "port": first.port}, {"name": "b", "port": second.port}],
            retry_sleeper=advance_cooldown,
            monotonic_clock=clock,
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
