# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import copy
import struct
import zlib

import numpy as np
import pytest

from relax.distributed.weight_sync.format import (
    WeightSyncError,
    decode_block,
    digest,
    encode_block,
    parse_json,
    validate_catalog,
)


@pytest.mark.parametrize("width", [1, 2, 4, 8])
@pytest.mark.parametrize("compression", [False, True])
def test_bitwise_roundtrip_and_encoding_choice(width, compression):
    # Include noncanonical NaNs, negative zero, arbitrary integer bit patterns,
    # unchanged blocks, one changed element, and a dense random replacement.
    rng = np.random.default_rng(410)
    base = rng.integers(0, 256, size=4096, dtype=np.uint8).tobytes()
    targets = [base, bytes(4096), rng.integers(0, 256, size=4096, dtype=np.uint8).tobytes()]
    sparse = bytearray(base)
    sparse[width : width * 2] = bytes(width)
    targets.append(bytes(sparse))
    for target in targets:
        encoded, payload = encode_block(target, base, width, compression)
        assert decode_block(encoded, payload, base, len(target), width, digest(target)) == target
    encoded, payload = encode_block(base, base, width, compression)
    assert encoded["encoding"] == "COPY" and not payload
    encoded, _ = encode_block(targets[-1], base, width, False)
    assert encoded["encoding"] == "SPARSE"


@pytest.mark.parametrize("bad_positions", [[2, 2], [3, 1], [0, 4]])
def test_sparse_rejects_duplicate_unsorted_or_out_of_range_positions(bad_positions):
    payload = struct.pack("<III", 2, *bad_positions) + b"\x01\x02"
    descriptor = {"encoding": "SPARSE", "object": digest(payload), "length": len(payload)}
    with pytest.raises(WeightSyncError, match="positions"):
        decode_block(descriptor, payload, bytes(4), 4, 1, digest(bytes(4)))


@pytest.mark.parametrize(
    "payload", [zlib.compress(bytes(64))[:-1], zlib.compress(bytes(64)) + b"tail", zlib.compress(bytes(65))]
)
def test_zlib_rejects_truncation_trailing_bytes_and_decompression_bombs(payload):
    descriptor = {"encoding": "RAW_ZLIB", "object": digest(payload), "length": len(payload)}
    with pytest.raises(WeightSyncError):
        decode_block(descriptor, payload, None, 64, 1, digest(bytes(64)))


def test_base_and_payload_identity_are_checked():
    before = bytes(1024)
    target = before[:10] + b"x" + before[11:]
    encoded, payload = encode_block(target, before, 1, False)
    with pytest.raises(WeightSyncError, match="checksum"):
        decode_block(encoded, payload[:-1] + b"y", before, 1024, 1, digest(target))
    with pytest.raises(WeightSyncError, match="checksum"):
        decode_block(encoded, payload, b"z" * 1024, 1024, 1, digest(target))


def test_strict_json_and_logical_coverage():
    for data in [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}']:
        with pytest.raises(WeightSyncError):
            parse_json(data)
    catalog = {
        "format": 1,
        "byte_order": "little",
        "schema_context": {},
        "chunk_bytes": 8,
        "tensors": [
            {
                "name": "x",
                "shape": [4],
                "dtype": "BF16",
                "blocks": [{"offset": 0, "length": 8, "digest": digest(bytes(8))}],
            }
        ],
    }
    validate_catalog(catalog)
    for field, value in [("shape", [True]), ("shape", [4.0]), ("blocks", []), ("dtype", "UNKNOWN")]:
        broken = copy.deepcopy(catalog)
        broken["tensors"][0][field] = value
        with pytest.raises(WeightSyncError):
            validate_catalog(broken)
