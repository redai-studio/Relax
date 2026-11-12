# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""SandboxFusion-compatible HTTP bridge to one frozen guarded backend."""

import argparse
import asyncio
import json
import os
import signal
import time
from pathlib import Path

from examples.retool.execution_backend import select_backend
from examples.retool.execution_backend import verify_guard as _verify_guard
from examples.retool.landlock_sandbox import POLICY


def put(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".pending.{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def verify_guard(result: dict, policy: str = POLICY) -> None:
    """Keep the existing Landlock guard-checking entry point compatible."""
    _verify_guard(result, policy)


async def persist(path: Path, value: dict) -> None:
    """Shared-disk fsync must not stall every HTTP request's event loop."""
    await asyncio.to_thread(put, path, value)


def execution_status(result: dict) -> str:
    """Classify the actual restricted child's status, not a wrapper's
    encoding."""
    if result["timed_out"] or result["returncode"] == -signal.SIGXCPU:
        return "TimeLimitExceeded"
    if result["returncode"] < 0:
        # verify_guard runs before this function. Synchronous faults after the
        # attestation are payload failures; external kills remain infrastructure
        # failures and must never be silently scored as model errors.
        payload_faults = {
            signal.SIGSEGV,
            signal.SIGBUS,
            signal.SIGILL,
            signal.SIGFPE,
            signal.SIGABRT,
            signal.SIGXFSZ,
            signal.SIGTRAP,
        }
        signum = -result["returncode"]
        if signum in payload_faults:
            return "Error"
        raise RuntimeError(f"Unexpected isolated process signal {signum}")
    return "Finished" if result["returncode"] == 0 else "Error"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    assert args.root.parent.stat().st_mode & 0o777 == 0o711
    args.root.mkdir(mode=0o711)
    args.root.chmod(0o711)
    audit = (
        Path(os.environ.get("RETOOL_EXEC_AUDIT_ROOT", os.environ["ROOT"]))
        / "executions"
        / args.root.name
        / str(os.getpid())
    )
    audit.mkdir(parents=True)
    backend = select_backend(args.root)
    put(audit / "backend_selected.json", backend.startup)
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request

    app = FastAPI()
    slots = asyncio.Semaphore(32)
    counter = 0
    summary_only = os.environ.get("RETOOL_EXEC_SUMMARY_ONLY") == "1"
    counts = dict(started=0, finished=0, failed=0)
    last_summary = 0.0

    def record_summary(kind: str) -> None:
        nonlocal last_summary
        counts[kind] += 1
        now = time.monotonic()
        if now - last_summary >= 300:
            temporary = audit / "summary.pending"
            temporary.write_text(json.dumps(dict(counts=counts, observed=time.time())))
            os.replace(temporary, audit / "summary.json")
            last_summary = now

    @app.get("/health")
    async def health() -> dict:
        return dict(backend=backend.name, profile=backend.policy, cap=32)

    @app.post("/run_code")
    async def execute(request: Request) -> dict:
        nonlocal counter
        body = await request.json()
        if (
            body.get("language") != "python"
            or body.get("run_timeout") != 30
            or body.get("memory_limit_MB") != 1024
            or body.get("files")
            or body.get("fetch_files")
            or body.get("stdin")
            or not isinstance(body.get("code"), str)
            or len(body["code"].encode()) > 65536
        ):
            raise HTTPException(400, "Require stateless Python, 30 seconds, 1 GiB, <=64KiB source")
        async with slots:
            counter += 1
            key = str(counter)
            if summary_only:
                record_summary("started")
            else:
                await persist(audit / f"{key}.started.json", dict(code=body["code"], historical_execution=False))
            try:
                result = await asyncio.to_thread(
                    backend.run_isolated, body["code"], args.root, timeout=30, cpu_seconds=30, memory_mb=1024
                )
                try:
                    verify_guard(result, backend.policy)
                    status = execution_status(result)
                except BaseException:
                    # Keep the raw guard/signal diagnostic only on failure.
                    # Accepted calls persist the complete result exactly once.
                    await persist(audit / f"{key}.executor_result.json", result)
                    raise
                if status == "Error" and result["returncode"] < 0:
                    # Keep the actual signal and guard evidence even in
                    # summary-only mode when a payload crashes.
                    await persist(audit / f"{key}.executor_result.json", result)
                if summary_only:
                    record_summary("finished")
                else:
                    await persist(audit / f"{key}.returned.json", result)
                return dict(
                    status="Success", run_result=dict(status=status, stdout=result["stdout"], stderr=result["stderr"])
                )
            except BaseException as error:
                if summary_only:
                    record_summary("failed")
                await persist(audit / f"{key}.failed.json", dict(error=str(error), infrastructure_failure=True))
                raise

    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
