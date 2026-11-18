# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Settlement of distributed operations before resources can be reused."""

import ray


class UnsafeInferenceError(RuntimeError):
    """An unknown old operation cannot participate in local recovery."""


def wait_all(refs: list, timeout: float = 600) -> list:
    """An exception from one rank does not prove its peers have stopped."""
    if not refs:
        return []
    ready, pending = ray.wait(refs, num_returns=len(refs), timeout=timeout)
    if pending:
        raise UnsafeInferenceError(f"{len(pending)} inference operations still running; require global restart")
    results, errors = [], []
    for ref in refs:
        try:
            results.append(ray.get(ref))
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise errors[0]
    return results
