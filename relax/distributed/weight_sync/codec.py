# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Exact byte replacement, including NaN payloads and signed zero."""

import zlib

import numpy as np


MAX_CHUNK_BYTES = 8 * 1024 * 1024


def _stored_size(kind: str, payload_size: int) -> int:
    # A chunk manifest has fixed-width object/output digests and identical
    # offset/size fields for every candidate. Only these two JSON values vary.
    return payload_size + len(kind) + len(str(payload_size))


def _candidate_order(candidate: tuple[str, bytes]) -> tuple[int, int]:
    kind, payload = candidate
    preference = {"raw": 0, "copy": 1, "sparse": 2, "raw_zlib": 3, "xor_zlib": 4}
    return _stored_size(kind, len(payload)), preference[kind]


def _encode_replacement(current: bytes, base: bytes | None, width: int, limit: int | None = None) -> tuple[str, bytes]:
    if width not in (1, 2, 4, 8) or len(current) % width or len(current) > MAX_CHUNK_BYTES:
        raise ValueError("Invalid chunk dimensions")
    if base is None:
        return "raw", current
    if len(base) != len(current):
        raise ValueError("Base size differs")
    if current == base:
        return "copy", b""
    # Integer bit views enable vectorized equality while preserving signed-zero
    # and NaN payload distinctions. Replacement records keep the original bytes.
    sparse = bytearray()
    values = np.frombuffer(current, dtype=f"V{width}")
    current_bits = values.view(f"<u{width}")
    previous_bits = np.frombuffer(base, dtype=f"<u{width}")
    record_dtype = np.dtype([("index", "<u4"), ("value", f"V{width}")])
    raw_cost = _stored_size("raw", len(current))
    budget = raw_cost if limit is None else min(raw_cost, limit)
    # Bound the comparison mask to 1 MiB while amortizing NumPy dispatch and
    # record construction across large sparse model blocks. The budget test
    # still runs before allocating replacement records, with identical bytes.
    window = 1 << 20
    for start in range(0, len(values), window):
        changed = current_bits[start : start + window] != previous_bits[start : start + window]
        count = np.count_nonzero(changed)
        if not count:
            continue
        sparse_size = len(sparse) + count * record_dtype.itemsize
        sparse_cost = _stored_size("sparse", sparse_size)
        if sparse_cost > budget or sparse_cost == raw_cost:
            return "raw", current
        indices = np.flatnonzero(changed) + start
        records = np.empty(count, dtype=record_dtype)
        records["index"] = indices
        records["value"] = values[indices]
        sparse.extend(records.tobytes())
    return "sparse", bytes(sparse)


def _xor(left: bytes, right: bytes) -> bytes:
    # Byte views preserve every bit and let independent block workers use the
    # native ufunc instead of serializing large Python integer conversions.
    return np.bitwise_xor(np.frombuffer(left, dtype=np.uint8), np.frombuffer(right, dtype=np.uint8)).tobytes()


def encode(current: bytes, base: bytes | None, width: int, *, compression: bool = False) -> tuple[str, bytes]:
    if not compression:
        return _encode_replacement(current, base, width)
    if width not in (1, 2, 4, 8) or len(current) % width or len(current) > MAX_CHUNK_BYTES:
        raise ValueError("Invalid chunk dimensions")
    if base is not None and len(base) != len(current):
        raise ValueError("Base size differs")
    if current == base:
        return "copy", b""
    # Full and delta both get the same raw compression opportunity.
    candidates = [("raw", current), ("raw_zlib", zlib.compress(current, level=1))]
    if base is not None:
        candidates.append(("xor_zlib", zlib.compress(_xor(current, base), level=1)))
    best = min(candidates, key=_candidate_order)
    # Stop constructing sparse indices once an already encoded candidate wins.
    sparse = _encode_replacement(current, base, width, _stored_size(best[0], len(best[1])))
    return min((best, sparse), key=_candidate_order)


def decode(kind: str, payload: bytes, base: bytes | None, width: int, size: int) -> bytes:
    if width not in (1, 2, 4, 8) or not 0 <= size <= MAX_CHUNK_BYTES or size % width:
        raise ValueError("Invalid decoded size")
    if kind in ("raw_zlib", "xor_zlib"):
        decoder = zlib.decompressobj()
        try:
            value = decoder.decompress(payload, size + 1)
        except zlib.error as error:
            raise ValueError("Invalid compressed chunk") from error
        if len(value) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError("Compressed chunk exceeds its size or has trailing data")
        if kind == "raw_zlib":
            return value
        if base is None or len(base) != size:
            raise ValueError("Missing or incompatible XOR base")
        return _xor(value, base)
    if kind == "raw":
        if len(payload) != size:
            raise ValueError("Truncated raw chunk")
        return payload
    if base is None or len(base) != size:
        raise ValueError("Missing or incompatible base")
    if kind == "copy":
        if payload:
            raise ValueError("COPY must be empty")
        return base
    if kind != "sparse" or len(payload) % (4 + width) or len(payload) >= size:
        raise ValueError("Invalid sparse encoding")
    result = bytearray(base)
    records = np.frombuffer(payload, dtype=np.dtype([("index", "<u4"), ("value", f"V{width}")]))
    indices = records["index"]
    if np.any(indices >= size // width) or np.any(indices[1:] <= indices[:-1]):
        raise ValueError("Sparse indices must be ordered, unique and in bounds")
    np.frombuffer(result, dtype=f"V{width}")[indices] = records["value"]
    return bytes(result)
