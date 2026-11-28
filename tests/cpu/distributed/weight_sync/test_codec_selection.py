"""Compare encoded candidates using their actual serialized chunk records."""

import hashlib
import json
import random
import zlib

import pytest

from relax.distributed.weight_sync.codec import decode, encode


def wire_cost(kind, payload, current):
    record = {
        "offset": 8192,
        "size": len(current),
        "kind": kind,
        "object": hashlib.sha256(payload).hexdigest(),
        "encoded_size": len(payload),
        "output": hashlib.sha256(current).hexdigest(),
    }
    return len(payload) + len(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())


def test_small_sparse_payload_can_have_larger_record_than_raw():
    base = bytes(8)
    current = b"\x01\x00" + bytes(6)
    sparse = bytes(4) + current[:2]
    assert len(sparse) < len(current)
    assert wire_cost("sparse", sparse, current) > wire_cost("raw", current, current)
    assert encode(current, base, 2) == ("raw", current)


@pytest.mark.parametrize("width", [1, 2, 4, 8])
@pytest.mark.parametrize("compression", [False, True])
def test_selected_record_is_minimum_among_enabled_candidates(width, compression):
    generator = random.Random(0)
    preference = {"raw": 0, "sparse": 2, "raw_zlib": 3, "xor_zlib": 4}
    for elements in (4, 8, 16, 32, 64, 128):
        for changed in (1, max(1, elements // 4), elements):
            base = generator.randbytes(elements * width)
            current = bytearray(base)
            for index in range(changed):
                current[index * width] ^= 1
            current = bytes(current)
            sparse = b"".join(
                index.to_bytes(4, "little") + current[index * width : (index + 1) * width]
                for index in range(elements)
                if current[index * width : (index + 1) * width] != base[index * width : (index + 1) * width]
            )
            candidates = [("raw", current)]
            if len(sparse) < len(current):
                candidates.append(("sparse", sparse))
            if compression:
                candidates.extend(
                    [
                        ("raw_zlib", zlib.compress(current, level=1)),
                        ("xor_zlib", zlib.compress(bytes(a ^ b for a, b in zip(base, current, strict=True)), level=1)),
                    ]
                )
            expected = min(candidates, key=lambda item: (wire_cost(*item, current), preference[item[0]]))
            selected = encode(current, base, width, compression=compression)
            assert selected == expected
            assert decode(*selected, base, width, len(current)) == current
