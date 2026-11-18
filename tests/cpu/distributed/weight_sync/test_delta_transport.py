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


def test_staged_tcp_prefix_is_immutable_bounded_and_never_applies_before_owner_barrier(monkeypatch):
    """Real TCP data with a CPU model-RPC boundary; no CUDA acceptance
    credit."""
    from types import SimpleNamespace

    import ray

    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    actions, buffers, consumed = [], {}, []

    def operation(action, **kwargs):
        actions.append(action)
        if action == "receive":
            key = kwargs["round_id"]
            buffers[key] = bytes(receive(kwargs["descriptor"], "1", timeout_s=3, max_bytes=16))
        elif action == "begin":
            assert not consumed
        elif action == "consume":
            assert "begin" in actions
            consumed.append((kwargs["round_id"], buffers.pop(kwargs["round_id"])))
            return dict(full_required=["weight"])
        elif action == "ready":
            return [dict(retained_payload_bytes=sum(map(len, buffers.values())))]

    monkeypatch.setattr(ray, "get", lambda refs, **kwargs: refs)
    transfer = DeltaTransfer(DeltaSyncConfig(stream="test", staging_budget_bytes=16), "cpu")
    transfer.engines = (SimpleNamespace(delta_weight_sync=SimpleNamespace(remote=operation)),)
    transfer.ranks, transfer.tcp_members = [[1]], {"1"}
    transfer.tcp = TcpFrameSource("127.0.0.1", timeout_s=3, host_budget_bytes=16)
    try:
        transfer.begin(version="1", base="0", seed=False, defer_apply=True)
        metadata, raw = encode([("weight", (2,), torch.float32, None, torch.ones(2))], max_bytes=16)
        expected = raw.numpy().tobytes()
        transfer.stage(metadata, raw)
        raw.zero_()
        transfer.stage(metadata, raw)
        assert not transfer.can_stage(metadata, raw)
        with pytest.raises(ValueError, match="capacity"):
            transfer.stage(metadata, raw)
        assert consumed == [] and actions == ["receive", "ready", "receive", "ready"]
        with pytest.raises(RuntimeError, match="generating model"):
            transfer.apply()
        transfer.activate()
        assert transfer.apply() == {"weight"}
        assert consumed == [(0, expected), (1, bytes(8))]
        assert transfer.can_stage(metadata, raw)
        assert transfer.send(metadata, raw) == {"weight"}
        assert consumed[-1] == (2, bytes(8))  # activate did not reset received round IDs.
        assert transfer.metrics["prestaged_frames"] == 2
        assert transfer.metrics["prestaged_bytes"] == 16
        with pytest.raises(RuntimeError, match="only once"):
            transfer.activate()
    finally:
        transfer.tcp.close()


def test_empty_staged_groups_still_consume_directory_admission():
    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    transfer = DeltaTransfer(DeltaSyncConfig(stream="test"), "cpu")
    transfer._directory_bytes = (16 << 20) - 2
    assert not transfer.can_stage({"tensors": []}, torch.empty(0, dtype=torch.uint8))


def test_empty_directory_heap_is_charged_even_when_json_is_small():
    import json

    from relax.distributed.weight_sync.config import DeltaSyncConfig
    from relax.distributed.weight_sync.transport import DeltaTransfer

    metadata, raw = encode(
        [("weight", (1,), torch.float32, torch.empty(0, dtype=torch.int32), torch.empty(0))], max_bytes=8
    )
    transfer = DeltaTransfer(DeltaSyncConfig(stream="test"), "cpu")
    wire = len(json.dumps(metadata, separators=(",", ":")).encode())
    assert transfer._retained_directory_size(metadata) > wire
    transfer._directory_bytes = (16 << 20) - wire
    assert not transfer.can_stage(metadata, raw)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_adjacent_staged_frames_preserve_eight_byte_decode_alignment(dtype):
    from relax.backends.sglang.weight_sync_proxy import _StagingArena

    arena = _StagingArena(64, "cpu")
    first = torch.arange(5, dtype=torch.bfloat16)
    second = torch.tensor([17], dtype=dtype)
    _, raw = encode([("bf16", (5,), first.dtype, None, first)], max_bytes=64)
    arena.reserve(0, raw.numel()).copy_(raw)
    metadata, raw = encode([("buffer", (1,), dtype, None, second)], max_bytes=64)
    slab = arena.reserve(1, raw.numel())
    slab.copy_(raw)
    _, _, decoded = next(iter(tensors(metadata, slab)))
    assert torch.equal(decoded, second)
    arena.release(0)
    arena.release(1)
    assert arena.free == [(0, 64)]
