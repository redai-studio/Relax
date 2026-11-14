# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Format 1: canonical logical tensors and bounded lossless block codecs.

This module has no device, filesystem, or training-runtime dependencies.
"""

import hashlib
import json
import math
import struct
import sys
import zlib
from typing import Any

import numpy as np


MAX_BLOCK_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
DTYPE_SIZES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}
ENCODINGS = {"COPY", "SPARSE", "RAW", "RAW_ZLIB", "XOR_ZLIB"}


class WeightSyncError(ValueError):
    """Invalid, unavailable, or inconsistent weight content."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_digest(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise WeightSyncError("Invalid SHA-256")
    return value


def integer(value: Any, minimum: int = 0, maximum: int = 2**63 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise WeightSyncError(f"Expected integer in [{minimum}, {maximum}]")
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise WeightSyncError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(data: bytes, limit: int = MAX_MANIFEST_BYTES) -> Any:
    if len(data) > limit:
        raise WeightSyncError("JSON exceeds size limit")
    try:
        return json.loads(data, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise WeightSyncError("Invalid JSON") from error


def _reject_constant(value: str) -> None:
    raise WeightSyncError(f"Non-finite JSON number: {value}")


def validate_catalog(catalog: dict) -> list[dict]:
    if not isinstance(catalog, dict) or catalog.get("format") != 1 or type(catalog["format"]) is not int:
        raise WeightSyncError("Unsupported logical format")
    if catalog.get("byte_order") != "little" or sys.byteorder != "little":
        raise WeightSyncError("Only little-endian tensor bytes are supported")
    chunk_bytes = integer(catalog.get("chunk_bytes"), 1, MAX_BLOCK_BYTES)
    if not isinstance(catalog.get("schema_context"), dict) or not isinstance(catalog.get("tensors"), list):
        raise WeightSyncError("Missing schema context or tensor directory")
    previous = None
    flattened = []
    for tensor in catalog["tensors"]:
        if not isinstance(tensor, dict):
            raise WeightSyncError("Invalid tensor entry")
        name = tensor.get("name")
        if not isinstance(name, str) or not name or (previous is not None and name.encode() <= previous):
            raise WeightSyncError("Tensor names must be unique and sorted by UTF-8 bytes")
        previous = name.encode()
        dtype = tensor.get("dtype")
        if dtype not in DTYPE_SIZES or not isinstance(tensor.get("shape"), list):
            raise WeightSyncError("Unsupported tensor dtype or shape")
        width = DTYPE_SIZES[dtype]
        size = math.prod(integer(dim) for dim in tensor["shape"]) * width
        integer(size)
        if chunk_bytes < width or not isinstance(tensor.get("blocks"), list):
            raise WeightSyncError("Invalid block boundaries")
        aligned_chunk = chunk_bytes // width * width
        offset = 0
        for block in tensor["blocks"]:
            if not isinstance(block, dict):
                raise WeightSyncError("Invalid block entry")
            length = min(aligned_chunk, size - offset)
            if length <= 0 or integer(block.get("offset")) != offset or integer(block.get("length"), 1) != length:
                raise WeightSyncError("Incomplete or overlapping tensor coverage")
            require_digest(block.get("digest"))
            flattened.append(block | {"name": name, "width": width})
            offset += length
        if offset != size:
            raise WeightSyncError("Incomplete tensor coverage")
    return flattened


def validate_manifest(manifest: dict, expected_id: str | None = None) -> list[dict]:
    if not isinstance(manifest, dict) or type(manifest.get("format")) is not int or manifest["format"] != 1:
        raise WeightSyncError("Unsupported artifact format")
    if not isinstance(manifest.get("stream"), str) or not manifest["stream"]:
        raise WeightSyncError("Missing stream")
    integer(manifest.get("version"))
    if manifest.get("kind") not in {"FULL", "DELTA"}:
        raise WeightSyncError("Invalid artifact kind")
    require_digest(manifest.get("root"))
    blocks = validate_catalog(manifest.get("catalog"))
    if digest(canonical_json(manifest["catalog"])) != manifest["root"]:
        raise WeightSyncError("Logical root mismatch")
    if manifest["kind"] == "DELTA":
        integer(manifest.get("base_version"), 0, manifest["version"] - 1)
        require_digest(manifest.get("base_root"))
    elif manifest.get("base_version") is not None or manifest.get("base_root") is not None:
        raise WeightSyncError("FULL cannot depend on a base")
    if not isinstance(manifest.get("blocks"), list) or len(manifest["blocks"]) != len(blocks):
        raise WeightSyncError("Encoded block coverage mismatch")
    for block, encoded in zip(blocks, manifest["blocks"], strict=True):
        if not isinstance(encoded, dict) or encoded.get("encoding") not in ENCODINGS:
            raise WeightSyncError("Invalid block encoding")
        if manifest["kind"] == "FULL" and encoded["encoding"] not in {"RAW", "RAW_ZLIB"}:
            raise WeightSyncError("FULL contains a base-dependent encoding")
        length = integer(encoded.get("length"), 0, MAX_BLOCK_BYTES)
        if encoded["encoding"] == "COPY":
            if length or encoded.get("object") is not None:
                raise WeightSyncError("COPY cannot carry payload")
        else:
            require_digest(encoded.get("object"))
            if length == 0:
                raise WeightSyncError("Empty encoded payload")
        if encoded["encoding"] == "RAW" and length != block["length"]:
            raise WeightSyncError("RAW size mismatch")
    if len(canonical_json(manifest)) > MAX_MANIFEST_BYTES:
        raise WeightSyncError("Manifest exceeds size limit")
    if expected_id is not None and digest(canonical_json(manifest)) != require_digest(expected_id):
        raise WeightSyncError("Artifact digest mismatch")
    return blocks


def encode_block(target: bytes, base: bytes | None, width: int, compression: bool = True) -> tuple[dict, bytes]:
    integer(width, 1, 8)
    if not 0 < len(target) <= MAX_BLOCK_BYTES or len(target) % width:
        raise WeightSyncError("Invalid target block size")
    if base is not None and len(base) != len(target):
        raise WeightSyncError("Base block size mismatch")
    if target == base:
        return {"encoding": "COPY", "object": None, "length": 0}, b""
    candidates = [("RAW", target)]
    if compression:
        candidates.append(("RAW_ZLIB", zlib.compress(target, level=1)))
    if base is not None:
        before = np.frombuffer(base, dtype=np.uint8)
        after = np.frombuffer(target, dtype=np.uint8)
        changed = np.flatnonzero(np.any(before.reshape(-1, width) != after.reshape(-1, width), axis=1))
        # Do not allocate sparse values if their payload already exceeds the best replacement.
        if 4 + len(changed) * (4 + width) <= min(len(payload) for _, payload in candidates):
            positions = changed.astype("<u4").tobytes()
            values = after.reshape(-1, width)[changed].tobytes()
            candidates.append(("SPARSE", struct.pack("<I", len(changed)) + positions + values))
        if compression:
            candidates.append(("XOR_ZLIB", zlib.compress(np.bitwise_xor(before, after).tobytes(), level=1)))
    ranked = []
    for order, (encoding, payload) in enumerate(candidates):
        if len(payload) > MAX_BLOCK_BYTES:
            continue
        descriptor = {"encoding": encoding, "object": digest(payload), "length": len(payload)}
        ranked.append((len(payload) + len(canonical_json(descriptor)), order, descriptor, payload))
    _, _, descriptor, payload = min(ranked, key=lambda candidate: candidate[:2])
    return descriptor, payload


def _decompress(payload: bytes, size: int) -> bytes:
    decoder = zlib.decompressobj()
    try:
        output = decoder.decompress(payload, size + 1)
    except zlib.error as error:
        raise WeightSyncError("Invalid compressed block") from error
    if len(output) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise WeightSyncError("Truncated, oversized, or trailing compressed data")
    return output


def decode_block(
    descriptor: dict, payload: bytes, base: bytes | None, size: int, width: int, target_digest: str
) -> bytes:
    integer(size, 1, MAX_BLOCK_BYTES)
    integer(width, 1, 8)
    encoding = descriptor.get("encoding")
    if encoding not in ENCODINGS or len(payload) != integer(descriptor.get("length"), 0, MAX_BLOCK_BYTES):
        raise WeightSyncError("Encoded object length mismatch")
    if encoding != "COPY" and digest(payload) != require_digest(descriptor.get("object")):
        raise WeightSyncError("Encoded object checksum mismatch")
    if encoding in {"COPY", "SPARSE", "XOR_ZLIB"} and (base is None or len(base) != size):
        raise WeightSyncError("Base required with matching block size")
    if encoding == "COPY":
        if payload:
            raise WeightSyncError("COPY cannot carry payload")
        output = base
    elif encoding == "RAW":
        output = payload
    elif encoding == "RAW_ZLIB":
        output = _decompress(payload, size)
    elif encoding == "XOR_ZLIB":
        output = np.bitwise_xor(
            np.frombuffer(_decompress(payload, size), np.uint8), np.frombuffer(base, np.uint8)
        ).tobytes()
    else:
        if len(payload) < 4 or size % width:
            raise WeightSyncError("Invalid sparse payload")
        count = struct.unpack_from("<I", payload)[0]
        if count > size // width or len(payload) != 4 + count * (4 + width):
            raise WeightSyncError("Invalid sparse length")
        positions = np.frombuffer(payload, "<u4", count=count, offset=4)
        if count and (positions[-1] >= size // width or np.any(positions[1:] <= positions[:-1])):
            raise WeightSyncError("Sparse positions must be increasing and in bounds")
        output_array = np.frombuffer(base, np.uint8).copy().reshape(-1, width)
        output_array[positions] = np.frombuffer(payload, np.uint8, offset=4 + 4 * count).reshape(-1, width)
        output = output_array.tobytes()
    if len(output) != size or digest(output) != require_digest(target_digest):
        raise WeightSyncError("Reconstructed target checksum mismatch")
    return output
