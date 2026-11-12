# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Job-owned, stateless Python endpoint using Landlock, seccomp and rlimits."""

import json
import os
import secrets
import signal
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

from examples.retool_agentic.landlock_sandbox import POLICY, run_isolated
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def check_guard(result: dict) -> None:
    guard = result["guard"]
    if not guard or any(
        guard.get(key) != value
        for key, value in dict(profile=POLICY, uid=65534, gid=65534, seccomp_mode=2, no_new_privs=1).items()
    ):
        raise RuntimeError(
            f"Restricted Python child failed to establish its isolation guard: {result['stderr'][:2000]}"
        )


def validate_runtime(root: Path) -> None:
    """Check fresh computation and forbidden operations in the current
    runtime."""
    nonce = secrets.randbelow(1000000000)
    canary = root / "canary.txt"
    canary.write_text(str(nonce))
    os.chown(canary, 65534, 65534)
    canary.chmod(0o600)
    cases = [(f"print(({nonce} * {i + 3} + 19) % 997)", str((nonce * (i + 3) + 19) % 997)) for i in range(8)]
    cases += [
        (f"print(open({str(canary)!r}).read())", None),
        (f"open({str(canary)!r}, 'w').write('changed')", None),
        ("import socket; socket.socket()", None),
        ("import os; os.fork()", None),
        ("import threading; threading.Thread(target=lambda: None).start()", None),
        ("import subprocess; subprocess.run(['/bin/echo', 'forbidden'])", None),
        ("import os; os.kill(os.getpid(), 0)", None),
        ("import os; os.mkdir('forbidden')", None),
    ]
    try:
        for code, expected in cases:
            result = run_isolated(code, root, timeout=30, cpu_seconds=30, memory_mb=1024)
            check_guard(result)
            if result["timed_out"] or (result["returncode"] == 0) != (expected is not None):
                raise RuntimeError(f"Landlock runtime validation failed: {result['stderr']}")
            if expected is not None and result["stdout"].strip() != expected:
                raise RuntimeError("Landlock runtime computation returned an unexpected result")
        if canary.read_text() != str(nonce):
            raise RuntimeError("Landlock runtime validation modified a forbidden file")
    finally:
        canary.unlink()
    logger.info("Landlock runtime validation passed: %s", POLICY)


def execute(code: str, root: Path) -> dict:
    result = run_isolated(code, root, timeout=30, cpu_seconds=30, memory_mb=1024)
    check_guard(result)
    if result["timed_out"] or result["returncode"] == -signal.SIGXCPU:
        status = "TimeLimitExceeded"
    elif result["returncode"] < 0:
        # After guard attestation, synchronous faults belong to the payload;
        # external termination remains an infrastructure failure.
        payload_faults = {
            signal.SIGSEGV,
            signal.SIGBUS,
            signal.SIGILL,
            signal.SIGFPE,
            signal.SIGABRT,
            signal.SIGXFSZ,
            signal.SIGTRAP,
        }
        if -result["returncode"] not in payload_faults:
            raise RuntimeError(f"Unexpected isolated Python signal: {result['returncode']}")
        status = "Error"
    else:
        status = "Finished" if result["returncode"] == 0 else "Error"
    return dict(status="Success", run_result=dict(status=status, stdout=result["stdout"], stderr=result["stderr"]))


@contextmanager
def local_service() -> Iterator[str]:
    # Use /tmp directly: the dropped uid must traverse every parent directory.
    # Model, data and checkpoint directories keep their original permissions.
    with tempfile.TemporaryDirectory(prefix="retool-sandbox-", dir="/tmp") as directory:
        root = Path(directory)
        root.chmod(0o711)
        validate_runtime(root)
        slots = threading.BoundedSemaphore(32)

        class Handler(BaseHTTPRequestHandler):
            def reply(self, status: int, body: dict) -> None:
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self) -> None:
                if self.path != "/run_code":
                    self.reply(404, dict(error="Unknown endpoint"))
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 512 * 1024:
                        raise ValueError("Invalid request size")
                    body = json.loads(self.rfile.read(length))
                    if (
                        not isinstance(body, dict)
                        or body.get("language") != "python"
                        or body.get("run_timeout") != 30
                        or body.get("memory_limit_MB") != 1024
                        or body.get("files")
                        or body.get("fetch_files")
                        or body.get("stdin")
                        or not isinstance(body.get("code"), str)
                        or not body["code"].strip()
                        or len(body["code"].encode()) > 65536
                    ):
                        raise ValueError("Require stateless Python, 30 seconds, 1 GiB and no file transfer")
                except (ValueError, TypeError) as error:
                    self.reply(400, dict(error=str(error)))
                    return
                try:
                    with slots:
                        result = execute(body["code"], root)
                except Exception:
                    logger.exception("Landlock execution infrastructure failed")
                    self.reply(500, dict(error="Landlock execution infrastructure failed"))
                    return
                self.reply(200, result)

            def log_message(self, format: str, *args: object) -> None:
                logger.debug(format, *args)

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}/run_code"
            logger.info("Started job-owned Landlock Python service at %s", url)
            try:
                yield url
            finally:
                server.shutdown()
                thread.join()
