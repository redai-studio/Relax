"""Focused CPU regressions for bounded Mini SWE transport recovery."""

import contextlib
import http.client
import io
import json
import os
import signal
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


CLIENT = Path(__file__).resolve().parents[3] / "examples/mini_swe_agent/agent_client.py"


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.hook = None

    def monotonic(self):
        return self.now

    def time(self):
        return 1000 + self.now

    def sleep(self, delay):
        assert 0 <= delay <= 0.10000001
        self.sleeps.append(delay)
        self.now += delay
        if self.hook:
            self.hook()


class Signals:
    SIGTERM, SIGINT = signal.SIGTERM, signal.SIGINT

    def __init__(self):
        self.handlers = {}

    def signal(self, number, callback):
        self.handlers[number] = callback

    def cancel(self):
        self.handlers[self.SIGTERM](self.SIGTERM, None)


class Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self):
        return self.value if isinstance(self.value, bytes) else json.dumps(self.value).encode()


class Fixture:
    def __init__(self, case):
        self.case = case
        self.stack = contextlib.ExitStack()
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.module = types.ModuleType("client_retry_test")
        exec(compile(CLIENT.read_text(), str(CLIENT), "exec"), self.module.__dict__)
        self.clock, self.signals = Clock(), Signals()
        self.receipt, self.output = self.root / "receipt.json", self.root / "output.json"
        self.expected = {"schema": 1, "session_id": "client", "nonce": "nonce", "status": "released"}
        env = {
            "AGENT_SERVER_URL": "http://example.invalid",
            "RELAX_SESSION_ID": "client",
            "RELAX_GROUP_ID": "group",
            "RELAX_ROLLOUT_MODE": "train",
            "RELAX_BASE_URL": "http://router.invalid",
            "RELAX_REQUIRE_RESOURCE_RELEASE": "1",
            "RELAX_RESOURCE_RELEASE_NONCE": "nonce",
            "RELAX_RESOURCE_RELEASE_PATH": str(self.receipt),
            "RELAX_OUTPUT_JSON": str(self.output),
        }
        self.stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
        self.stack.enter_context(mock.patch.object(self.module, "time", self.clock))
        self.stack.enter_context(mock.patch.object(self.module, "signal", self.signals))
        self.jitter = self.stack.enter_context(mock.patch.object(self.module.random, "uniform", return_value=1.0))
        self.logs, self.calls = [], []
        self.stack.enter_context(mock.patch.object(self.module, "_diagnostic", side_effect=self.logs.append))
        self.stack.enter_context(mock.patch.object(self.module.urllib.request, "build_opener", return_value=self))
        self.atomic = self.module.atomic_json
        self.handler = None
        case.addCleanup(self.stack.close)

    def terminal(self, *, released=True, status="completed"):
        return {
            "session_id": "server",
            "client_session_id": "client",
            "status": status,
            "resource_release": {**self.expected, "status": "released" if released else "unknown"},
            "exit_code": 0,
            "agent_exit_status": "Submitted",
            "submission": "original patch",
            "n_calls": 7,
            "reward": 1.0,
        }

    def open(self, request, *, timeout):
        path = request.full_url.removeprefix("http://example.invalid")
        call = {
            "path": path,
            "method": request.get_method(),
            "start": self.clock.now,
            "timeout": timeout,
            "payload": json.loads(request.data) if request.data else None,
        }
        self.calls.append(call)
        try:
            return Response(self.handler(call))
        finally:
            call["end"] = self.clock.now

    def timeout(self, call):
        self.clock.now += call["timeout"]
        raise TimeoutError("simulated lost response")

    def run(self, handler):
        self.handler = handler
        return self.module.run()

    def poll_calls(self):
        return [call for call in self.calls if call["method"] == "GET"]

    def cancel_calls(self):
        return [call for call in self.calls if call["path"].endswith("/cancel")]


class ClientRetryTests(unittest.TestCase):
    def test_backoff_caps_and_success_resets_without_changing_reward(self):
        f = Fixture(self)
        actions = iter(["timeout"] * 5 + ["running", "timeout", "timeout", "completed"])

        def server(call):
            if call["method"] == "POST":
                return {"session_id": "server"}
            action = next(actions)
            if action == "timeout":
                f.timeout(call)
            return f.terminal(released=action == "completed", status=action)

        self.assertEqual(f.run(server), 0)
        calls = f.poll_calls()
        delays = [b["start"] - a["end"] for a, b in zip(calls, calls[1:])]
        for actual, expected in zip(delays, [0.5, 1, 2, 4, 5, 0.5, 0.5, 1]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertFalse(f.cancel_calls())
        self.assertEqual(json.loads(f.receipt.read_text()), f.expected)
        self.assertEqual(
            json.loads(f.output.read_text()),
            {
                "metadata": {"exit_status": "Submitted", "submission": "original patch", "n_calls": 7},
                "reward": 1.0,
            },
        )

    def test_twenty_failures_remain_bounded_and_cancel_polling_stays_half_second(self):
        f = Fixture(self)
        repeated_signal = False

        def server(call):
            if call["path"] == "/sessions":
                return {"session_id": "server"}
            if call["path"].endswith("/cancel"):
                return {}
            f.timeout(call)

        def signal_again():
            nonlocal repeated_signal
            if f.cancel_calls() and not repeated_signal:
                repeated_signal = True
                f.signals.cancel()

        f.clock.hook = signal_again
        self.assertEqual(f.run(server), 143)
        self.assertAlmostEqual(f.cancel_calls()[0]["start"], 123.0, places=6)
        self.assertAlmostEqual(f.clock.now, 162.5, places=6)
        self.assertTrue(repeated_signal)
        self.assertEqual(len(f.cancel_calls()), 1)
        self.assertEqual(json.loads(f.receipt.read_text())["status"], "unknown")
        after = [call for call in f.poll_calls() if call["start"] >= 123]
        for first, second in zip(after, after[1:]):
            self.assertAlmostEqual(second["start"] - first["end"], 0.5, places=6)
        self.assertTrue(all(call["timeout"] <= 2 for call in f.poll_calls()))

    def test_sigterm_interrupts_active_five_second_backoff(self):
        f = Fixture(self)
        failures, signal_at = 0, None

        def server(call):
            nonlocal failures, signal_at
            if call["path"] == "/sessions":
                return {"session_id": "server"}
            if call["path"].endswith("/cancel"):
                return {}
            if f.cancel_calls():
                return f.terminal(status="cancelled")
            failures += 1
            if failures == 5:
                signal_at = f.clock.now + call["timeout"] + 0.2
            f.timeout(call)

        def interrupt():
            if signal_at is not None and f.clock.now >= signal_at:
                f.clock.hook = None
                f.signals.cancel()

        f.clock.hook = interrupt
        self.assertEqual(f.run(server), 143)
        self.assertLessEqual(f.cancel_calls()[0]["start"] - signal_at, 0.1000001)
        self.assertEqual(json.loads(f.receipt.read_text()), f.expected)

    def test_retryable_http_errors_close_the_response_and_recover(self):
        for status in [408, 429, 500, 502, 503, 504]:
            with self.subTest(status=status):
                f = Fixture(self)
                body = io.BytesIO(b"not read")

                def server(call):
                    if call["path"] == "/sessions":
                        return {"session_id": "server"}
                    if len(f.poll_calls()) == 1:
                        raise urllib.error.HTTPError("http://example.invalid", status, "failure", {}, body)
                    return f.terminal()

                self.assertEqual(f.run(server), 0)
                self.assertTrue(body.closed)
                self.assertFalse(f.cancel_calls())
                f.stack.close()

    def test_permanent_http_errors_enter_cleanup_without_transport_backoff(self):
        for status in [400, 401, 403, 404, 409, 422, 501]:
            with self.subTest(status=status):
                f = Fixture(self)
                body = io.BytesIO(b"not read")

                def server(call):
                    if call["path"] == "/sessions":
                        return {"session_id": "server"}
                    if call["path"].endswith("/cancel"):
                        return {}
                    if len(f.poll_calls()) == 1:
                        raise urllib.error.HTTPError("http://example.invalid", status, "failure", {}, body)
                    return f.terminal(status="cancelled")

                self.assertEqual(f.run(server), 1)
                self.assertTrue(body.closed)
                self.assertEqual(len(f.cancel_calls()), 1)
                f.jitter.assert_not_called()
                self.assertFalse(f.output.exists())
                f.stack.close()

    def test_invalid_protocol_remains_failure_even_if_cleanup_later_confirms(self):
        for invalid in [b"[]", b"not-json", {"bad": "identity"}, http.client.IncompleteRead(b"partial", 100)]:
            with self.subTest(invalid=invalid):
                f = Fixture(self)

                def server(call):
                    if call["path"] == "/sessions":
                        return {"session_id": "server"}
                    if call["path"].endswith("/cancel"):
                        return {}
                    if len(f.poll_calls()) == 1:
                        if isinstance(invalid, Exception):
                            raise invalid
                        return invalid
                    return f.terminal(status="cancelled")

                self.assertEqual(f.run(server), 1)
                f.jitter.assert_not_called()
                self.assertEqual(json.loads(f.receipt.read_text()), f.expected)
                self.assertFalse(f.output.exists())
                f.stack.close()

    def test_wrong_release_identity_cannot_refund_even_after_cancellation(self):
        f = Fixture(self)

        def server(call):
            if call["path"] == "/sessions":
                return {"session_id": "server"}
            if call["path"].endswith("/cancel"):
                return {}
            result = f.terminal()
            result["resource_release"]["nonce"] = "another-session"
            return result

        self.assertEqual(f.run(server), 1)
        self.assertEqual(len(f.cancel_calls()), 1)
        self.assertEqual(json.loads(f.receipt.read_text())["status"], "unknown")
        self.assertFalse(f.output.exists())
        self.assertAlmostEqual(f.clock.now, 40.0, places=6)

    def test_local_receipt_failure_never_uses_network_backoff_or_claims_release(self):
        f = Fixture(self)

        def persist(path, value):
            if value.get("status") == "released":
                raise OSError("receipt storage unavailable")
            return f.atomic(path, value)

        f.stack.enter_context(mock.patch.object(f.module, "atomic_json", side_effect=persist))

        def server(call):
            if call["path"] == "/sessions":
                return {"session_id": "server"}
            return {} if call["path"].endswith("/cancel") else f.terminal()

        self.assertEqual(f.run(server), 1)
        self.assertAlmostEqual(f.clock.now, 40.0, places=6)
        self.assertEqual(json.loads(f.receipt.read_text())["status"], "unknown")
        f.jitter.assert_not_called()

    def test_local_output_failure_preserves_failure_and_does_not_relabel_network(self):
        f = Fixture(self)

        def persist(path, value):
            if path == f.output:
                raise OSError("output storage unavailable")
            return f.atomic(path, value)

        f.stack.enter_context(mock.patch.object(f.module, "atomic_json", side_effect=persist))

        def server(call):
            if call["path"] == "/sessions":
                return {"session_id": "server"}
            return {} if call["path"].endswith("/cancel") else f.terminal()

        self.assertEqual(f.run(server), 1)
        self.assertEqual(json.loads(f.receipt.read_text()), f.expected)
        self.assertFalse(f.output.exists())
        f.jitter.assert_not_called()

    def test_lost_create_response_retries_the_same_identity_once(self):
        f = Fixture(self)
        created, attempts = set(), 0

        def server(call):
            nonlocal attempts
            if call["path"] == "/sessions":
                attempts += 1
                key = call["payload"]["session_id"], call["payload"]["release_nonce"]
                created.add(key)
                if attempts < 5:
                    f.timeout(call)
                return {"session_id": "server"}
            return f.terminal()

        self.assertEqual(f.run(server), 0)
        self.assertEqual(attempts, 5)
        self.assertEqual(created, {("client", "nonce")})
        self.assertEqual(json.loads(f.receipt.read_text()), f.expected)

    def test_cancel_during_initial_unknown_write_releases_without_any_http(self):
        for boundary in ["before", "after"]:
            with self.subTest(boundary=boundary):
                f = Fixture(self)

                def persist(path, value):
                    initial = value.get("status") == "unknown"
                    if initial and boundary == "before":
                        f.signals.cancel()
                    result = f.atomic(path, value)
                    if initial and boundary == "after":
                        f.signals.cancel()
                    return result

                f.stack.enter_context(mock.patch.object(f.module, "atomic_json", side_effect=persist))
                self.assertEqual(f.run(lambda _: self.fail("HTTP must not start")), 143)
                self.assertFalse(f.calls)
                self.assertEqual(json.loads(f.receipt.read_text()), f.expected)
                f.stack.close()

    def test_cancel_after_create_may_have_sent_must_recover_remote_identity(self):
        f = Fixture(self)
        attempts = 0

        def server(call):
            nonlocal attempts
            if call["path"] == "/sessions":
                attempts += 1
                if attempts == 1:
                    f.signals.cancel()
                    f.timeout(call)
                self.assertEqual(json.loads(f.receipt.read_text())["status"], "unknown")
                return {"session_id": "server"}
            return {} if call["path"].endswith("/cancel") else f.terminal(status="cancelled")

        self.assertEqual(f.run(server), 143)
        self.assertEqual(attempts, 2)
        self.assertEqual(len(f.cancel_calls()), 1)
        self.assertEqual(json.loads(f.receipt.read_text()), f.expected)

    def test_cancel_after_unknown_create_never_forges_release(self):
        f = Fixture(self)

        def server(call):
            self.assertEqual(call["path"], "/sessions")
            if len(f.calls) == 1:
                f.signals.cancel()
            f.timeout(call)

        self.assertEqual(f.run(server), 143)
        self.assertAlmostEqual(f.clock.now, 40.0, places=6)
        self.assertEqual(json.loads(f.receipt.read_text())["status"], "unknown")
        self.assertFalse(f.output.exists())
        f.jitter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
