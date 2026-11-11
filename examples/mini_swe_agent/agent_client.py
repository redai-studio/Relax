#!/usr/bin/env python3
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Mini-swe HTTP client with identity-bound physical resource receipts."""

from __future__ import annotations

import http.client
import json
import os
import random
import signal
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


class _TransientRequestError(Exception):
    """An HTTP transport failure, distinct from local output/receipt I/O."""

    def __init__(self, cause: BaseException):
        super().__init__(type(cause).__name__)
        self.cause = cause


def _diagnostic(message: str) -> None:
    try:
        # Avoid buffered stderr reentrancy in the signal handler. Diagnostics
        # must never interrupt the resource-release handshake if stderr closes.
        os.write(2, f"agent_client: time={time.time():.3f} {message}\n".encode())
    except (OSError, ValueError, RuntimeError):
        pass


def trace_failure(session_id: str, result: dict) -> None:
    """Keep the former shell client's opt-in failure snapshots in Python."""
    directory = os.environ.get("AGENT_CLIENT_TRACE_DIR")
    if not directory or result.get("status") != "failed":
        return
    try:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        atomic_json(path / f"{session_id}.failed.json", result)
    except OSError as error:
        _diagnostic(f"failure trace unavailable error={type(error).__name__}")


def run() -> int:
    base = os.environ["AGENT_SERVER_URL"].rstrip("/")
    sid = os.environ["RELAX_SESSION_ID"]
    required = os.environ.get("RELAX_REQUIRE_RESOURCE_RELEASE") == "1"
    nonce = os.environ["RELAX_RESOURCE_RELEASE_NONCE"] if required else uuid.uuid4().hex
    release_path = Path(os.environ["RELAX_RESOURCE_RELEASE_PATH"]) if required else None
    expected = {"schema": 1, "session_id": sid, "nonce": nonce, "status": "released"}
    cancel_deadline: float | None = None
    cancel_reason = ""
    protocol_error = False
    poll_interval = float(os.environ.get("POLL_INTERVAL_SECONDS", "0.5"))
    initial_poll_delay = float(os.environ.get("INITIAL_POLL_DELAY_SECONDS", "0"))
    cancel_timeout = float(os.environ.get("CANCEL_REQUEST_TIMEOUT_SECONDS", "5"))
    failure_limit = int(os.environ.get("MAX_POLL_FAILURES", "20"))
    if poll_interval <= 0 or initial_poll_delay < 0 or cancel_timeout <= 0 or failure_limit <= 0:
        raise ValueError("Client polling and cancellation limits must be positive; initial delay may be zero")

    def cancel(_signum, _frame, *, reason: str | None = None):
        nonlocal cancel_deadline, cancel_reason
        if cancel_deadline is None:
            cancel_deadline = time.monotonic() + 40.0
            cancel_reason = reason or (f"signal:{_signum}" if _signum else "consecutive_io_failures")
            _diagnostic(f"cancelling reason={cancel_reason}")

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    if release_path is not None:
        atomic_json(release_path, {**expected, "status": "unknown"})

    def request(path: str, payload: dict | None = None, *, timeout: float = 2.0) -> dict:
        if cancel_deadline is not None:
            timeout = min(timeout, cancel_deadline - time.monotonic())
            if timeout <= 0:
                raise _TransientRequestError(TimeoutError("Resource-release deadline elapsed"))
        req = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        # Node-local traffic must not be sent to an HTTP proxy.
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
                raw = response.read()
                result = json.loads(raw) if raw else {}
                if not isinstance(result, dict):
                    raise ValueError("Agent server response must be an object")
                return result
        except urllib.error.HTTPError as exc:
            # HTTPError owns a response body/socket. Close without draining an
            # unbounded body before either retrying or entering cleanup.
            try:
                exc.close()
            except OSError:
                pass
            if exc.code in {408, 429, 500, 502, 503, 504}:
                raise _TransientRequestError(exc) from exc
            raise
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise _TransientRequestError(exc) from exc

    payload = {
        "session_id": sid,
        "group_id": os.environ["RELAX_GROUP_ID"],
        "mode": os.environ["RELAX_ROLLOUT_MODE"],
        "base_url": os.environ["RELAX_BASE_URL"],
        "api_key": sid,
        "release_nonce": nonce,
    }
    server_sid = None
    create_attempted = False
    errors = 0
    cancel_sent = False
    result: dict = {}
    while cancel_deadline is None or time.monotonic() < cancel_deadline:
        phase = "create"
        retry_delay = poll_interval
        try:
            if cancel_deadline is not None and not create_attempted:
                # No request could have allocated remote resources. Publish the
                # exact lease proof in normal control flow, never in the handler.
                phase = "release_receipt"
                if release_path is not None:
                    atomic_json(release_path, expected)
                _diagnostic(f"cancelled before create reason={cancel_reason}")
                return 1 if protocol_error else 143
            if server_sid is None:
                # A lost response cannot prove that creation did not happen.
                create_attempted = True
                created_sid = request("/sessions", payload)["session_id"]
                errors = 0
                if not isinstance(created_sid, str) or not created_sid:
                    raise ValueError("Agent server Session ID must be a nonempty string")
                server_sid = created_sid
                # An explicit warmup delay remains configurable, but cancellation
                # must never wait for it. The default starts polling immediately.
                first_poll_at = time.monotonic() + initial_poll_delay
                while cancel_deadline is None and time.monotonic() < first_poll_at:
                    time.sleep(min(0.1, max(0.0, first_poll_at - time.monotonic())))
            if cancel_deadline is not None and not cancel_sent:
                phase = "cancel"
                request(f"/sessions/{server_sid}/cancel", {}, timeout=cancel_timeout)
                errors = 0
                cancel_sent = True
            phase = "poll"
            result = request(f"/sessions/{server_sid}")
            errors = 0
            if result.get("client_session_id") != sid or result.get("session_id") != server_sid:
                raise RuntimeError("Agent server Session identity mismatch")
            release = result.get("resource_release")
            if (
                not isinstance(release, dict)
                or type(release.get("schema")) is not int
                or release.get("schema") != 1
                or release.get("session_id") != sid
                or release.get("nonce") != nonce
                or release.get("status") not in {"unknown", "released"}
            ):
                raise RuntimeError("Agent server resource-release identity mismatch")
            released = release == expected
            if released and release_path is not None:
                phase = "release_receipt"
                atomic_json(release_path, expected)
            if result.get("resource_release_failed"):
                trace_failure(sid, result)
                sys.stderr.write("Agent Session cleanup failed; physical release remains unconfirmed\n")
                return 1
            # Mini SWE always owns external resources. Even without a Runtime
            # capacity limit, returning a result must wait for their release.
            if result.get("status") in {"completed", "failed", "cancelled"} and released:
                if protocol_error:
                    _diagnostic("failed after invalid server response; resources released")
                    return 1
                if cancel_deadline is not None:
                    _diagnostic(f"cancelled released={released} reason={cancel_reason}")
                    return 143
                if result["status"] == "completed":
                    phase = "output"
                    atomic_json(
                        Path(os.environ["RELAX_OUTPUT_JSON"]),
                        {
                            "metadata": {
                                "exit_status": result["agent_exit_status"],
                                "submission": result["submission"],
                                "n_calls": result["n_calls"],
                            },
                            "reward": result["reward"],
                        },
                    )
                else:
                    trace_failure(sid, result)
                    sys.stderr.write(f"Agent session {result['status']}: {result.get('error', '')}\n")
                return int(result.get("exit_code", 1))
        except _TransientRequestError as failure:
            exc = failure.cause
            errors += 1
            if errors in {1, failure_limit}:
                reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
                # Do not log URLs, payloads or exception text containing credentials.
                _diagnostic(
                    f"I/O failure phase={phase} consecutive={errors} "
                    f"error={type(exc).__name__} reason_type={type(reason).__name__} "
                    f"errno={getattr(reason, 'errno', None)} http_status={getattr(exc, 'code', None)}"
                )
            if cancel_deadline is None:
                if errors >= failure_limit:
                    cancel(0, None)
                else:
                    # Bound recovery attempts and desynchronize clients sharing
                    # the same server. Jitter never exceeds the five-second cap.
                    retry_delay = min(5.0, 0.5 * 2 ** min(errors - 1, 4)) * random.uniform(0.9, 1.0)
        except urllib.error.HTTPError as exc:
            protocol_error = True
            cancel(0, None, reason=f"permanent_http:{exc.code}")
        except OSError as exc:
            # HTTP I/O is wrapped inside request(); this is local persistence.
            protocol_error = True
            cancel(0, None, reason=f"local_io:{type(exc).__name__}")
        except (ValueError, KeyError, TypeError, RuntimeError, http.client.HTTPException) as exc:
            protocol_error = True
            cancel(0, None, reason=f"invalid_response:{type(exc).__name__}")
        # Keep the normal polling frequency during cleanup, while interrupting
        # an active retry backoff promptly if cancellation arrives mid-sleep.
        cancelling = cancel_deadline is not None
        retry_at = time.monotonic() + (poll_interval if cancelling else retry_delay)
        if cancelling:
            retry_at = min(retry_at, cancel_deadline)
        while time.monotonic() < retry_at:
            if not cancelling and cancel_deadline is not None:
                break
            time.sleep(min(0.1, max(0.0, retry_at - time.monotonic())))
    _diagnostic(f"release deadline elapsed reason={cancel_reason}")
    return 143 if cancel_deadline is not None and not protocol_error else 1


if __name__ == "__main__":
    sys.exit(run())
