# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Mooncake safety guards and raw-byte payload correctness helpers.

Relax validates the capabilities and environment it can inspect without
modifying TransferQueue at runtime.  The pinned TransferQueue advertises a
versioned correctness contract for batch/retry result counts, remove failure
propagation, and fail-closed production-status ACKs.

Mooncake 0.3.10.post2 was observed corrupting TCP-protocol transfers through
its auto-enabled memcpy fast path, so that path is force-disabled here and an
explicit enable is rejected (see :func:`_enforce_safe_memcpy`).
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Mapping
from typing import Any


LeafDigest = tuple[str, str, str]
_REQUIRED_TQ_MOONCAKE_CONTRACT_VERSION = 1


def _enforce_safe_memcpy() -> None:
    """Force-disable mooncake's memcpy fast path; reject attempts to enable it.

    mooncake 0.3.10.post2 auto-enables ``MC_STORE_MEMCPY`` in TCP-only
    environments (``transfer_task.cpp`` "auto-detected: TCP-only environment,
    memcpy enabled") and that path silently truncates cross-node gets: roughly
    half of fresh-session first transfers returned rows whose tails were zero
    bytes from a 64 KiB-aligned offset onward while every batch code reported
    success (two-node forensic probes, 2026-08; 12/12 sessions clean with
    ``MC_STORE_MEMCPY=0`` vs ~50% corrupt without).  The same code path
    SIGSEGVs on single-node loopback.  Because the corruption is confirmed on
    that build, this guard fails closed: an explicit ``MC_STORE_MEMCPY=1`` is
    rejected at startup instead of honoured.  Re-gate this behavior only when a
    reliable fixed-build capability is available.
    """
    override = os.environ.get("MC_STORE_MEMCPY", "").strip()
    if override not in ("", "0"):
        raise RuntimeError(
            "MC_STORE_MEMCPY explicitly enables an unsafe value: the mooncake 0.3.10.post2 "
            "memcpy fast path silently truncates TCP transfers and can SIGSEGV. "
            "Unset MC_STORE_MEMCPY; Relax forces it to 0 until a fixed-build capability is available."
        )
    os.environ["MC_STORE_MEMCPY"] = "0"


def ensure_mooncake_correctness_guards() -> None:
    """Validate that the installed stack can run MooncakeStore safely.

    Enforces the memcpy environment contract and checks that the pinned
    TransferQueue advertises the Mooncake correctness contract Relax's data
    plane relies on. It does not modify TransferQueue at runtime.
    """
    _enforce_safe_memcpy()
    try:
        import transfer_queue as tq
    except ImportError as error:
        raise RuntimeError(
            "Installed TransferQueue does not satisfy the required Mooncake correctness contract"
        ) from error

    actual = getattr(tq, "MOONCAKE_CORRECTNESS_CONTRACT_VERSION", 0)
    if isinstance(actual, bool) or not isinstance(actual, int) or actual < _REQUIRED_TQ_MOONCAKE_CONTRACT_VERSION:
        raise RuntimeError("Installed TransferQueue does not satisfy the required Mooncake correctness contract")


def _tensor_digest(value: Any) -> LeafDigest:
    import torch

    flat = value.detach().cpu().contiguous().reshape(-1)
    raw = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
    shape = "x".join(str(dim) for dim in value.shape)
    return str(value.dtype), shape, hashlib.sha256(raw).hexdigest()


def _ndarray_digest(value: Any) -> LeafDigest:
    import numpy as np

    contiguous = np.ascontiguousarray(value)
    shape = "x".join(str(dim) for dim in value.shape)
    return f"np.{contiguous.dtype}", shape, hashlib.sha256(contiguous.tobytes()).hexdigest()


def _scalar_digest(value: Any) -> LeafDigest:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    elif isinstance(value, float):
        raw = struct.pack("!d", value)
    else:
        raw = repr(value).encode("utf-8")
    return f"py.{type(value).__name__}", "", hashlib.sha256(raw).hexdigest()


def _unwrap_non_tensor(value: Any) -> Any:
    if type(value).__name__ == "NonTensorStack":
        return value.tolist()
    if type(value).__name__ == "NonTensorData":
        return value.data
    return value


def _dict_child_path(prefix: str, key: Any) -> str:
    if not isinstance(key, str):
        raise TypeError(f"Unsupported payload dict key at {prefix}: {type(key).__name__}")
    return f"{prefix}.{key}" if key.isidentifier() else f"{prefix}[{key!r}]"


def leaf_digests(payload: Any, prefix: str = "payload") -> dict[str, LeafDigest]:
    """Map every supported leaf to ``(dtype, shape, raw-byte SHA-256)``."""
    import numpy as np
    import torch

    payload = _unwrap_non_tensor(payload)
    digests: dict[str, LeafDigest] = {}
    if isinstance(payload, torch.Tensor):
        if payload.is_nested:
            for index, row in enumerate(payload.unbind()):
                digests[f"{prefix}[{index}]"] = _tensor_digest(row)
        else:
            digests[prefix] = _tensor_digest(payload)
    elif isinstance(payload, np.ndarray):
        digests[prefix] = _ndarray_digest(payload)
    elif isinstance(payload, Mapping):
        for key in payload:
            if not isinstance(key, str):
                raise TypeError(f"Unsupported payload dict key at {prefix}: {type(key).__name__}")
        for key in sorted(payload):
            digests.update(leaf_digests(payload[key], _dict_child_path(prefix, key)))
    elif isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            digests.update(leaf_digests(item, f"{prefix}[{index}]"))
    elif isinstance(payload, (str, bytes, bool, int, float)) or payload is None:
        digests[prefix] = _scalar_digest(payload)
    else:
        raise TypeError(f"Unsupported payload leaf at {prefix}: {type(payload).__name__}")
    return digests


def diff_digests(expected: dict[str, LeafDigest], actual: dict[str, LeafDigest]) -> list[str]:
    """Return mismatch descriptions; an empty list means byte-exact."""
    problems: list[str] = []
    for path in sorted(expected.keys() | actual.keys()):
        want, have = expected.get(path), actual.get(path)
        if want is None:
            problems.append(f"{path}: unexpected extra leaf {have}")
        elif have is None:
            problems.append(f"{path}: missing (expected {want})")
        elif want != have:
            for axis, want_part, have_part in zip(("dtype", "shape", "sha256"), want, have, strict=True):
                if want_part != have_part:
                    problems.append(f"{path}: {axis} mismatch (expected {want_part}, got {have_part})")
    return problems


def payload_rows(value: Any) -> list[Any]:
    """Return logical rows from a tensor or non-tensor TQ column."""
    import torch

    value = _unwrap_non_tensor(value)
    if isinstance(value, torch.Tensor):
        return [value] if value.ndim == 0 else list(value.unbind())
    if isinstance(value, (list, tuple)):
        return [_unwrap_non_tensor(row) for row in value]
    raise TypeError(f"Unsupported payload column: {type(value).__name__}")


def payload_nbytes(value: Any) -> int:
    """Count raw bytes represented by supported TQ payload values."""
    import numpy as np
    import torch

    value = _unwrap_non_tensor(value)
    if isinstance(value, torch.Tensor):
        if value.is_nested:
            return sum(payload_nbytes(row) for row in value.unbind())
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(payload_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(payload_nbytes(item) for item in value)
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, float):
        return 8
    if isinstance(value, (bool, int)) or value is None:
        return len(repr(value).encode("utf-8"))
    raise TypeError(f"Unsupported payload leaf: {type(value).__name__}")
