# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from relax.distributed.weight_sync.codec import encode, tensors, validate_directory
from relax.distributed.weight_sync.tcp_transport import TcpFrameSource, receive


@pytest.mark.parametrize("empty", [False, True])
def test_tcp_delivers_same_immutable_wire_to_all_fixed_recipients(empty):
    positions = torch.empty(0, dtype=torch.int32) if empty else torch.tensor([0, 3], dtype=torch.int32)
    bits = torch.empty(0, dtype=torch.int16) if empty else torch.tensor([0x7FC1, -32768], dtype=torch.int16)
    metadata, raw = encode([("weight", (2, 2), torch.bfloat16, positions, bits.view(torch.bfloat16))], max_bytes=1024)
    immutable = raw.numpy().tobytes()
    source = TcpFrameSource("127.0.0.1", timeout_s=3, host_budget_bytes=1024)
    try:
        descriptor = source.publish(immutable, {"rank0", "rank1"})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(receive, descriptor, member, timeout_s=3, max_bytes=1024) for member in ("rank0", "rank1")
            ]
            payloads = [future.result(timeout=5) for future in futures]
        source.finish()
        raw.zero_()
        for payload in payloads:
            assert payload == immutable
            restored = torch.frombuffer(payload, dtype=torch.uint8) if payload else torch.empty(0, dtype=torch.uint8)
            entry, restored_positions, restored_values = next(iter(tensors(metadata, restored)))
            assert entry["dtype"] == "bfloat16"
            assert torch.equal(restored_positions, positions)
            assert torch.equal(restored_values.view(torch.int16), bits)
    finally:
        source.close()


def test_wire_retains_mixed_dtype_full_values_and_rejects_overlapping_ranges():
    bf16 = torch.tensor([0x7FC1, -32768], dtype=torch.int16).view(torch.bfloat16)
    fp32 = torch.tensor([0x7FC00002, 0], dtype=torch.int32).view(torch.float32)
    integer = torch.tensor([2**40, -(2**40)], dtype=torch.int64)
    metadata, raw = encode(
        [
            ("bf16", (2,), bf16.dtype, None, bf16),
            ("fp32", (2,), fp32.dtype, None, fp32),
            ("integer", (2,), integer.dtype, None, integer),
        ],
        max_bytes=1024,
    )
    values = [value for _, _, value in tensors(metadata, raw)]
    assert torch.equal(values[0].view(torch.int16), bf16.view(torch.int16))
    assert torch.equal(values[1].view(torch.int32), fp32.view(torch.int32))
    assert torch.equal(values[2], integer)
    metadata["tensors"][1]["offset"] = 0
    with pytest.raises(ValueError, match="overlap"):
        validate_directory(metadata, max_bytes=1024)
