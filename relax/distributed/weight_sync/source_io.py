# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Coordinated CPU/file work; training threads retain all collectives."""

import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import nullcontext
from contextvars import copy_context

from relax.utils.logging_utils import get_logger

from .deadline import time_budget, validate_timeout


logger = get_logger(__name__)


def agree_errors(error, group):
    """Exchange a small success flag; gather rank diagnostics only on
    failure."""
    import torch
    import torch.distributed as dist

    from . import traffic

    world = dist.get_world_size(group)
    if world == 1:
        return [error]
    failed = torch.tensor([int(error is not None)], dtype=torch.int64)
    traffic.all_reduce(failed, group=group)
    errors = [None] * world
    if failed[0].item():
        traffic.all_gather_object(errors, error, group=group)
    return errors


def agree_io_state(state, group):
    """Agree on completion/expiry and retain exact per-rank error messages.

    The waiter only needs all-done and any-expired. Three CPU integers avoid
    serializing a Python dictionary on every short file operation or poll.
    """
    import torch
    import torch.distributed as dist

    from . import traffic

    world = dist.get_world_size(group)
    if world == 1:
        return [state]
    flags = torch.tensor(
        [int(state["done"]), int(state["error"] is not None), int(state["expired"])], dtype=torch.int64
    )
    traffic.all_reduce(flags, group=group)
    done, failed, expired = flags.tolist()
    errors = [None] * world
    if failed:
        traffic.all_gather_object(errors, state["error"], group=group)
    return [{"done": done == world, "error": error, "expired": bool(expired)} for error in errors]


def coordinated_io(operation, agree, *, stage, timeout=3600.0, committing=False, executor=None):
    """Wait responsively for local file work on every participant.

    Cooperative operations check their deadline while working. Already running
    file calls drain before inputs are released. Successful completion returns
    its result even when late; actual worker errors still fail on every rank.
    """
    validate_timeout(timeout)
    start = time.monotonic()
    expired = False
    warned = False
    next_notice = start + 60

    def run():
        with time_budget(timeout):
            return operation() if operation is not None else None

    context = (
        nullcontext(executor)
        if executor is not None or operation is None
        else ThreadPoolExecutor(max_workers=1, thread_name_prefix="weight-file-io")
    )
    with context as pool:
        if operation is None:
            future = Future()
            future.set_result(None)
        else:
            future = pool.submit(copy_context().run, run)
        try:
            while True:
                value, error, done = None, None, False
                try:
                    value = future.result(timeout=0.05)
                    done = True
                except FutureTimeout:
                    if future.done():
                        # Completion can race the polling timeout. Read the actual
                        # future outcome instead of turning that race into failure.
                        done = True
                        failure = future.exception()
                        if failure is None:
                            value = future.result()
                        else:
                            error = f"{type(failure).__name__}: {failure}"
                except Exception as caught:
                    done, error = True, f"{type(caught).__name__}: {caught}"
                expired = expired or time.monotonic() - start > timeout
                states = agree({"done": done, "error": error, "expired": expired})
                if (
                    not warned
                    and any(state["expired"] for state in states)
                    and not all(state["done"] for state in states)
                ):
                    logger.warning(
                        "%s exceeded its %ss budget; draining the running file worker before releasing scratch. "
                        "Check storage latency; the candidate remains owned until the worker exits.",
                        stage,
                        timeout,
                    )
                    warned = True
                if time.monotonic() >= next_notice and not all(state["done"] for state in states):
                    logger.warning(
                        "%s still running after %.1fs; waiting for all %s participants to finish. "
                        "Inspect source I/O progress and storage latency; scratch is retained for active writes.",
                        stage,
                        time.monotonic() - start,
                        len(states),
                    )
                    next_notice = time.monotonic() + 60
                if all(state["done"] for state in states):
                    failures = [state["error"] for state in states]
                    if any(failures):
                        raise RuntimeError(f"{stage} failed: {failures}")
                    if any(state["expired"] for state in states):
                        logger.warning(
                            "%s completed after its %ss I/O budget; %s result retained",
                            stage,
                            timeout,
                            "durable" if committing else "completed",
                        )
                    return value
        finally:
            # A failed collective must not release inputs while a reusable
            # worker is still reading or writing them.
            while not future.done():
                if not wait([future], timeout=60).done:
                    logger.warning(
                        "%s: waiting for an active file worker during cleanup; inputs remain retained. "
                        "Inspect the worker stack and storage before retrying.",
                        stage,
                    )
