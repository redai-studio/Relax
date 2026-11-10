# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Native receiver range safety and HTTP batch lifetime management."""

import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable


# An HTTP timeout cannot prove that the receiver or a NIXL writer has stopped.
# Keep the entire failed session alive until its process exits; never recycle it.
_FAILED_SESSIONS: list[Any] = []


def _request_error(
    pool: ThreadPoolExecutor, request: Callable, payloads: list, requests: list[Future], deadline: Callable[[], float]
) -> Exception | None:
    """Drain submitted requests; their caller retains storage and owns failure
    policy."""
    error = None
    try:
        for engine, payload in payloads:
            requests.append(pool.submit(request, engine, "/update_weights_from_tensor", payload))
    except Exception as exc:
        error = exc
    done, pending = wait(requests, timeout=max(0.0, deadline() - time.monotonic()))
    for future in requests:
        if future in done:
            try:
                future.result()
            except Exception as exc:
                if error is None:
                    error = exc
    if pending and error is None:
        error = TimeoutError("Receiver HTTP requests remain unconfirmed after the batch deadline")
    return error


def validate_disjoint_ranges(params: dict, experts: set[str], fallback: set[str]) -> None:
    """Require expert destinations to be independent of all native storage."""
    if not experts or experts & fallback or experts | fallback != set(params):
        raise ValueError("Overlap requires complete, disjoint target parameter coverage")
    intervals = []
    for name, layout in params.items():
        address, nbytes = layout["address"], layout["nbytes"]
        if type(address) is not int or address <= 0 or type(nbytes) is not int or nbytes <= 0:
            raise ValueError("Overlap requires valid native address ranges")
        intervals.append((address, address + nbytes, name in experts))
    maximum_end = expert_end = 0
    for start, end, expert in sorted(intervals):
        if start < (maximum_end if expert else expert_end):
            raise ValueError("Expert storage overlaps another target parameter")
        maximum_end = max(maximum_end, end)
        if expert:
            expert_end = max(expert_end, end)


_FAILED_PLAN_SESSIONS: list[Any] = []


class PlanRequests:
    """Drain the whole synchronous HTTP batch before Actor error consensus."""

    def __init__(self, timeout: float = 310.0) -> None:
        self.timeout = timeout
        self.failed = False
        self.requests: list[Future] = []
        self.storage: Any = None
        self.deadline = 0.0

    def ready(self) -> None:
        if self.failed or self.requests or self.storage is not None:
            raise RuntimeError("An active or failed plan session cannot be reused")

    def request_all(self, pool: ThreadPoolExecutor, request: Callable, payloads: list, storage: Any) -> None:
        self.ready()
        if not payloads or len({engine for engine, _ in payloads}) != len(payloads):
            raise ValueError("Plan RPC requires distinct engine payloads")
        self.storage = (payloads, storage)
        self.deadline = time.monotonic() + self.timeout
        error = _request_error(pool, request, payloads, self.requests, lambda: self.deadline)
        if error is not None:
            self.failed = True
            raise error
        self.requests = []
        self.storage = None

    def quarantine(self, owner: Any) -> None:
        if not any(value is owner for value in _FAILED_PLAN_SESSIONS):
            _FAILED_PLAN_SESSIONS.append(owner)
        self.failed = True
        # Timeout is not completion: never cancel or release unconfirmed work.
        try:
            wait(self.requests, timeout=max(0.0, self.deadline - time.monotonic()))
        except BaseException:
            pass
