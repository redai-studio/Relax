"""Bounded owned-process execution and verifiable stage receipts."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path


CANCEL_REQUESTED = threading.Event()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def execute(argv: list[str], directory: Path, env: dict[str, str], receipt: Path, timeout: float) -> dict:
    receipt.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record = {"argv": argv, "cwd": str(directory), "started_unix": started, "status": "running"}
    write_json(receipt, record)
    with receipt.with_suffix(".log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            argv, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        record["pid"] = process.pid
        write_json(receipt, record)
        try:
            end = time.monotonic() + max(1, timeout)
            while True:
                if CANCEL_REQUESTED.is_set():
                    raise KeyboardInterrupt("Campaign cancellation")
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(argv, timeout)
                try:
                    code = process.wait(timeout=min(2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            record.update(
                status="timed_out_or_interrupted", native_returncode=process.returncode, seconds=time.time() - started
            )
            write_json(receipt, record)
            raise
    record.update(
        status="completed" if code == 0 else "failed",
        native_returncode=code,
        seconds=time.time() - started,
        log_sha256=digest(receipt.with_suffix(".log")),
    )
    write_json(receipt, record)
    if code:
        raise RuntimeError(f"Stage failed ({code}): {receipt}")
    return record
