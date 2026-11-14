# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Raw bits, incremental ownership and shared capacity of sender shards."""

import pytest
import torch

from relax.distributed.weight_sync.memory_delta import MemoryBucket, validate_bucket
from relax.distributed.weight_sync.memory_transport import MemorySource, receive_memory_update
from relax.distributed.weight_sync.streaming_delta import (
    SnapshotBudget,
    StreamingDeltaEncoder,
    steady_state_snapshot_bytes,
)


def test_streaming_mixed_dtype_real_tcp_preserves_bits_and_published_bytes():
    budget = SnapshotBudget(1 << 20)
    encoder = StreamingDeltaEncoder(budget, bucket_bytes=256)
    values = [
        ("bf16", torch.arange(64, dtype=torch.int16).view(torch.bfloat16)),
        ("fp32", torch.arange(64, dtype=torch.int32).view(torch.float32)),
    ]
    delivered = {}
    steady_estimate = None
    source = MemorySource("127.0.0.1", timeout_s=3)

    def apply(manifest, payloads):
        assert manifest["group_update"] == f"optimizer-{manifest['version']}"
        for metadata, payload in zip(manifest["buckets"], payloads):
            validate_bucket(metadata, payload, max_bytes=4096)
            name = metadata["names"][0][0]
            dtype = torch.int16 if metadata["dtype"] == "bfloat16" else torch.int32
            if metadata["encoding"] == "full":
                delivered[name] = torch.frombuffer(bytearray(payload), dtype=dtype).clone()
            else:
                changed = metadata["changed"]
                positions = torch.frombuffer(bytearray(payload[: changed * 4]), dtype=torch.int32).long()
                bits = torch.frombuffer(bytearray(payload[changed * 4 :]), dtype=dtype)
                delivered[name][positions] = bits

    try:
        for version in range(2):
            encoder.begin()
            for name, tensor in values:
                encoder.add(name, tensor)
            buckets = encoder.finish()
            estimate = steady_state_snapshot_bytes(buckets)
            assert estimate >= budget.peak_bytes
            if steady_estimate is not None:
                # Sparse payloads shrink; the owned snapshots and admission do not.
                assert estimate == steady_estimate
            steady_estimate = estimate
            expected = {name: tensor.view(torch.uint8).numpy().tobytes() for name, tensor in values}
            descriptor = source.publish(
                buckets,
                stream="mixed",
                version=version,
                base=version - 1,
                schema=encoder.pending_schema,
                members={"receiver"},
                reset=version == 0,
                group_update=f"optimizer-{version}",
            )
            # Both the iterator's values and its mutable bucket objects may be
            # reused after publish. Only the frozen pending snapshots advance.
            for bucket in buckets:
                assert bucket.snapshot is None
                bucket.names = (("rebound", (1,)),)
                bucket.payload = b"broken"
                bucket.snapshot = torch.ones(64, dtype=torch.bfloat16)
            receive_memory_update(descriptor, "receiver", apply, timeout_s=3)
            for name, bits in delivered.items():
                assert bits.view(torch.uint8).numpy().tobytes() == expected[name]
            with pytest.raises(RuntimeError, match="Prepare commit"):
                encoder.commit()
            assert encoder.version == version - 1
            encoder.prepare_commit(source.wait(1), {"receiver"})
            encoder.commit()
            source.finish()
            if version == 0:
                # Different NaN payload and signed zero require exact integer
                # comparisons, including for the actual FP32 source values.
                values[0][1].view(torch.int16)[1] = -32768
                values[0][1].view(torch.int16)[2] = 0x7F81
                values[1][1].view(torch.int32)[1] = -(2**31)
                values[1][1].view(torch.int32)[2] = 0x7F800001
        assert budget.used_bytes == sum(t.numel() * t.element_size() for _, t in values)
    finally:
        source.close()
        encoder.close()
    assert budget.used_bytes == 0


def test_transposed_capture_has_one_owned_clone_and_no_reshape_copy():
    budget = SnapshotBudget(256)
    encoder = StreamingDeltaEncoder(budget, bucket_bytes=128)
    tensor = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8).t()
    encoder.begin()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        encoder.add("transposed", tensor)
        buckets = encoder.finish()
    # preserve_format would cause an additional clone in reshape while both
    # the captured tensor and the concatenated snapshot are still live.
    assert sum(event.count for event in profile.key_averages() if event.key == "aten::clone") == 1
    assert budget.peak_bytes == 256
    assert buckets[0].payload == tensor.contiguous().view(torch.uint8).numpy().tobytes()
    encoder.close()
    assert budget.used_bytes == 0


def test_full_control_encodes_without_comparing_old_values(monkeypatch):
    encoder = StreamingDeltaEncoder(SnapshotBudget(4096), bucket_bytes=128)
    with pytest.raises(ValueError, match="FULL encoding"):
        encoder.begin(compare_changes=False)
    encoder.begin()
    encoder.add("x", torch.zeros(16, dtype=torch.bfloat16))
    encoder.finish()
    encoder.prepare_commit({"a"}, {"a"})
    encoder.commit()

    # FULL alone preserves diff statistics; skipping comparison is explicit.
    encoder.begin(full=True)
    target = torch.zeros(16, dtype=torch.bfloat16)
    target[0] = 1
    encoder.add("x", target)
    bucket = encoder.finish()[0]
    assert bucket.encoding == "full" and bucket.changed == 1
    encoder.prepare_commit({"a"}, {"a"})
    encoder.commit()

    def unexpected_comparison(*args, **kwargs):
        raise AssertionError("FULL transport control must not find changed positions")

    monkeypatch.setattr(torch, "nonzero", unexpected_comparison)
    encoder.begin(full=True, compare_changes=False)
    target = torch.ones(16, dtype=torch.bfloat16)
    encoder.add("x", target)
    bucket = encoder.finish()[0]
    assert bucket.encoding == "full"
    assert bucket.payload == target.view(torch.uint8).numpy().tobytes()
    # changed denotes sent elements in this mode, not a measured diff count.
    assert bucket.changed == bucket.elements == 16
    encoder.prepare_commit({"a"}, {"a"})
    encoder.commit()
    encoder.close()


def test_streaming_capture_freezes_reused_buffers_before_iterator_advances():
    budget = SnapshotBudget(4096)
    encoder = StreamingDeltaEncoder(budget, bucket_bytes=8)
    buffer = torch.ones(4, dtype=torch.bfloat16)
    encoder.begin()
    encoder.add("first", buffer)
    buffer.fill_(2)
    encoder.add("second", buffer)
    buffer.fill_(3)
    buckets = encoder.finish()
    assert [torch.frombuffer(bytearray(b.payload), dtype=torch.bfloat16).tolist() for b in buckets] == [
        [1.0] * 4,
        [2.0] * 4,
    ]
    encoder.close()
    assert budget.used_bytes == 0


def test_shared_rank_budget_rejects_second_encoder_before_allocating_and_abort_releases_pending():
    budget = SnapshotBudget(48)
    first = StreamingDeltaEncoder(budget, bucket_bytes=32)
    second = StreamingDeltaEncoder(budget, bucket_bytes=32)
    first.begin()
    first.add("x", torch.zeros(16, dtype=torch.bfloat16))
    assert budget.used_bytes == 32
    second.begin()
    with pytest.raises(ValueError, match="shared rank budget"):
        second.add("y", torch.zeros(16, dtype=torch.bfloat16))
    second.abort()
    first.abort()
    assert budget.used_bytes == 0


def test_schema_change_and_missing_member_never_advance_stream():
    budget = SnapshotBudget(4096)
    encoder = StreamingDeltaEncoder(budget, bucket_bytes=32)
    encoder.begin()
    encoder.add("x", torch.zeros(8, dtype=torch.bfloat16))
    encoder.finish()
    with pytest.raises(RuntimeError, match="Every fixed"):
        encoder.prepare_commit({"a"}, {"a", "b"})
    assert encoder.version == -1
    encoder.prepare_commit({"a", "b"}, {"a", "b"})
    encoder.commit()
    baseline_bytes = budget.used_bytes
    encoder.begin()
    encoder.add("renamed", torch.ones(8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="layout changed"):
        encoder.finish()
    encoder.abort()
    assert encoder.version == 0 and budget.used_bytes == baseline_bytes
    encoder.close()


def test_sparse_fp32_positions_are_validated_even_with_a_fresh_checksum():
    import hashlib

    payload = (8).to_bytes(4, "little", signed=True) + b"\x00\x00\x80\x3f"
    bucket = MemoryBucket((("x", (8,)),), 8, 1, "sparse", payload, None, "float32")
    metadata = dict(
        names=bucket.names,
        elements=8,
        changed=1,
        encoding="sparse",
        dtype="float32",
        bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    with pytest.raises(ValueError, match="in bounds"):
        validate_bucket(metadata, payload, max_bytes=4096)
    with pytest.raises(ValueError, match="in bounds"):
        bucket.metadata()
