# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Bound native FULL calls even when a peer is stuck inside a legacy
collective."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Callable


def guarded_native_call(
    call: Callable[[], Any],
    *,
    process_group: Any,
    timeout_s: float,
    failure_path: Path,
    quarantine: Callable[[], list[str]] | None = None,
) -> Any:
    """Use an independent CPU group to observe return errors and stuck peers.

    Returned exceptions converge without abandoning healthy source actors. An
    in-flight failed collective cannot safely be unwound: all source processes
    exit after recording failure, so the owning Ray Job must fail. The caller
    never permits generation during this native FULL operation.
    """
    import torch
    import torch.distributed as dist

    if timeout_s <= 0:
        raise ValueError("Native call deadline must be positive")
    done = threading.Event()
    state: dict = dict(error=None, result=None)
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    stream = torch.cuda.current_stream(device) if device is not None else None

    def work():
        try:
            if device is not None:
                torch.cuda.set_device(device)
                with torch.cuda.stream(stream):
                    state["result"] = call()
            else:
                state["result"] = call()
        except BaseException as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            done.set()

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    started, first_error = monotonic(), None
    while True:
        status = dict(done=done.is_set(), error=state["error"])
        statuses = [None] * dist.get_world_size(process_group)
        try:
            dist.all_gather_object(statuses, status, group=process_group)
        except BaseException as exc:
            statuses = [dict(done=False, error=f"Native control group failed: {type(exc).__name__}: {exc}")]
        failed = any(item["error"] for item in statuses)
        complete = all(item["done"] for item in statuses)
        timed_out = monotonic() - started >= timeout_s
        if failed and first_error is None:
            first_error = monotonic()
        if failed or timed_out:
            # Allow peers whose ordinary calls already returned to report that
            # fact, but never wait on an abandoned data collective indefinitely.
            if not complete and not timed_out and monotonic() - first_error < 1:
                sleep(0.02)
                continue
            errors = []
            if quarantine is not None:
                try:
                    errors = quarantine()
                except BaseException as exc:
                    errors = [f"Quarantine failed: {type(exc).__name__}: {exc}"]
            report = dict(
                statuses=statuses, deadline_exceeded=timed_out, in_flight=not complete, quarantine_errors=errors
            )
            try:
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                failure_path.write_text(json.dumps(report, indent=2))
            except BaseException:
                pass  # Failed audit IO cannot keep a poisoned GPU actor alive.
            if not complete:
                os._exit(74)
            worker.join()
            raise RuntimeError(f"Ordinary FULL failed across source ranks: {report}")
        if complete:
            worker.join()
            return state["result"]
        sleep(0.02)
