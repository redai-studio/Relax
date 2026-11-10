# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import hashlib

import pytest
import torch

from relax.backends.sglang.memory_delta_loader import MemoryModelReceiver, apply_sparse, record_full_load
from relax.distributed.weight_sync.memory_delta import MemoryDeltaEncoder, validate_bucket
from relax.distributed.weight_sync.memory_transport import MemorySource, receive_memory_update


class PackedModel(torch.nn.Module):
    """Real torch copies with the QKV/gate-up slice pattern used by SGLang."""

    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Parameter(torch.zeros(12, 4, dtype=torch.bfloat16))
        self.gate_up = torch.nn.Parameter(torch.zeros(6, 4, dtype=torch.bfloat16))

    def load_weights(self, tensors):
        for name, value in tensors:
            if name in ("q", "k", "v"):
                self.qkv.narrow(0, ("q", "k", "v").index(name) * 4, 4).copy_(value)
            elif name in ("gate", "up"):
                self.gate_up.narrow(0, ("gate", "up").index(name) * 3, 3).copy_(value)


def weights():
    return [
        (name, torch.zeros(rows, 4, dtype=torch.bfloat16))
        for name, rows in [("q", 4), ("k", 4), ("v", 4), ("gate", 3), ("up", 3)]
    ]


def unpack(bucket):
    flat = torch.frombuffer(bytearray(bucket.payload), dtype=torch.bfloat16)
    tensors, offset = [], 0
    for name, shape in bucket.names:
        size = torch.Size(shape).numel()
        tensors.append((name, flat[offset : offset + size].view(shape)))
        offset += size
    return tensors


def test_sparse_memory_round_trip_preserves_fused_layout_and_all_bits():
    source = weights()
    encoder = MemoryDeltaEncoder()
    seed = encoder.prepare(source)[0]
    actual, expected = PackedModel(), PackedModel()
    routes = record_full_load(actual, unpack(seed))
    encoder.commit({"engine0", "engine1"}, {"engine0", "engine1"})
    for step in range(3):
        # Signed zero and distinct NaN payload bits are data, never sentinels.
        source[step][1].view(torch.int16)[0, 1] = [-32768, 0x7FC1, 0x7FC2][step]
        bucket = encoder.prepare(source)[0]
        assert bucket.encoding == "sparse"
        validate_bucket(bucket.metadata(), bucket.payload, max_bytes=4096)
        apply_sparse(actual, routes, bucket.metadata(), bucket.payload)
        with torch.no_grad():
            expected.load_weights(source)
        for name, value in actual.named_parameters():
            assert torch.equal(value.view(torch.int16), dict(expected.named_parameters())[name].view(torch.int16))
        encoder.commit({"engine0", "engine1"}, {"engine0", "engine1"})
    assert encoder.version == 3


def test_memory_baseline_is_owned_and_advances_only_after_every_ack():
    source = weights()
    encoder = MemoryDeltaEncoder()
    pending = encoder.prepare(source)
    source[0][1].fill_(1)
    assert not pending[0].snapshot.any()
    with pytest.raises(RuntimeError, match="Every fixed inference member"):
        encoder.commit({"a"}, {"a", "b"})
    assert encoder.version == -1
    with pytest.raises(RuntimeError, match="not committed"):
        encoder.prepare(source)
    encoder.commit({"a", "b"}, {"a", "b"})
    encoder.prepare(source)
    encoder.abort()
    assert encoder.version == 0


def test_dense_fallback_and_empty_delta_are_bounded_and_exact():
    source = [("x", torch.zeros(32, dtype=torch.bfloat16))]
    encoder = MemoryDeltaEncoder()
    encoder.prepare(source)
    encoder.commit({"a"}, {"a"})
    empty = encoder.prepare(source)[0]
    assert empty.encoding == "sparse" and empty.changed == 0 and empty.payload == b""
    validate_bucket(empty.metadata(), empty.payload, max_bytes=1024)
    encoder.commit({"a"}, {"a"})
    source[0][1].fill_(2)
    dense = encoder.prepare(source)[0]
    assert dense.encoding == "full" and len(dense.payload) == 64
    assert torch.equal(unpack(dense)[0][1], source[0][1])


def test_bad_frame_hash_and_sparse_positions_are_rejected_before_apply():
    encoder = MemoryDeltaEncoder()
    source = [("x", torch.zeros(32, dtype=torch.bfloat16))]
    encoder.prepare(source)
    encoder.commit({"a"}, {"a"})
    source[0][1][1] = 1
    bucket = encoder.prepare(source)[0]
    with pytest.raises(ValueError, match="checksum"):
        validate_bucket(bucket.metadata(), bucket.payload[:-1] + b"\xff", max_bytes=1024)
    invalid = (32).to_bytes(4, "little", signed=True) + bucket.payload[4:]
    metadata = bucket.metadata() | {"sha256": hashlib.sha256(invalid).hexdigest()}
    with pytest.raises(ValueError, match="in bounds"):
        validate_bucket(metadata, invalid, max_bytes=1024)


def test_memory_schema_changes_require_a_new_stream():
    encoder = MemoryDeltaEncoder()
    encoder.prepare(weights())
    encoder.commit({"a"}, {"a"})
    with pytest.raises(ValueError, match="schema changed"):
        encoder.prepare(weights()[:-1])


def test_loader_observation_is_thread_local_and_rejects_ignored_weights():
    original = torch.Tensor.copy_
    with pytest.raises(ValueError, match="did not consume"):
        record_full_load(PackedModel(), [("ignored", torch.ones(4, dtype=torch.bfloat16))])
    assert torch.Tensor.copy_ is original


def test_sparse_loader_rejects_relocated_parameter_storage():
    encoder = MemoryDeltaEncoder()
    source = weights()
    seed = encoder.prepare(source)[0]
    model = PackedModel()
    routes = record_full_load(model, unpack(seed))
    encoder.commit({"a"}, {"a"})
    source[0][1][0, 0] = 1
    bucket = encoder.prepare(source)[0]
    model.qkv.data = model.qkv.data.clone()
    with pytest.raises(ValueError, match="storage changed"):
        apply_sparse(model, routes, bucket.metadata(), bucket.payload)


def test_snapshot_budget_and_unregistered_value_transforms_fail_closed():
    with pytest.raises(ValueError, match="budget"):
        MemoryDeltaEncoder(snapshot_budget_bytes=4).prepare(weights())

    class Transformed(PackedModel):
        def load_weights(self, tensors):
            for _, tensor in tensors:
                self.qkv.copy_(tensor + 1)

    with pytest.raises(ValueError, match="transformed source values"):
        record_full_load(Transformed(), [("qkv", torch.zeros(12, 4, dtype=torch.bfloat16))])


def test_iterator_buffer_reuse_is_frozen_before_advancing():
    shared = torch.zeros(4, dtype=torch.bfloat16)

    def iterator():
        for index in range(3):
            shared.fill_(index)
            yield str(index), shared

    bucket = MemoryDeltaEncoder().prepare(iterator())[0]
    assert [tensor.unique().item() for _, tensor in unpack(bucket)] == [0, 1, 2]


def test_forced_full_reports_actual_changed_elements():
    encoder = MemoryDeltaEncoder()
    source = weights()
    encoder.prepare(source)
    encoder.commit({"a"}, {"a"})
    source[0][1][0, 0] = 1
    bucket = encoder.prepare(source, full=True)[0]
    assert bucket.encoding == "full" and bucket.changed == 1


def test_post_copy_parameter_transform_is_not_admitted():
    class Transformed(PackedModel):
        def load_weights(self, tensors):
            for _, tensor in tensors:
                self.qkv.copy_(tensor)
                self.qkv.mul_(2)

    with pytest.raises(ValueError, match="outside a direct copy"):
        record_full_load(Transformed(), [("qkv", torch.ones(12, 4, dtype=torch.bfloat16))])


@pytest.mark.parametrize("operation", ["source_mul", "source_copy", "parameter_out"])
def test_source_mutation_and_keyword_parameter_writes_are_rejected(operation):
    class Transformed(PackedModel):
        def load_weights(self, tensors):
            for _, tensor in tensors:
                if operation == "source_mul":
                    tensor.mul_(2)
                if operation == "source_copy":
                    tensor.copy_(tensor + 1)
                self.qkv.copy_(tensor)
                if operation == "parameter_out":
                    torch.mul(self.qkv, 2, out=self.qkv)

    with pytest.raises(ValueError, match="direct copy|modified an HF source"):
        record_full_load(Transformed(), [("qkv", torch.zeros(12, 4, dtype=torch.bfloat16))])


def test_real_tcp_multi_member_updates_and_duplicate_ack_preserve_bits():
    from concurrent.futures import ThreadPoolExecutor

    source = MemorySource("127.0.0.1", timeout_s=2)
    encoder = MemoryDeltaEncoder(bucket_bytes=48)
    tensors = weights()
    members = {name: MemoryModelReceiver(PackedModel()) for name in ("a", "b")}
    try:
        for step in range(3):
            tensors[step][1].view(torch.int16)[0, 0] = [0x7FC1, -32768, 0x7FC2][step]
            buckets = encoder.prepare(tensors)
            descriptor = source.publish(
                buckets,
                stream="s",
                version=step,
                base=step - 1,
                schema=encoder.pending_schema,
                members=set(members),
                reset=step == 0,
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                jobs = [
                    executor.submit(receive_memory_update, descriptor, name, receiver.apply, timeout_s=2)
                    for name, receiver in members.items()
                ]
                for job in jobs:
                    job.result(timeout=3)
            # Retransmission after an ACK loss cannot reapply an old version.
            receive_memory_update(descriptor, "a", members["a"].apply, timeout_s=2)
            encoder.commit(source.wait(1), set(members))
            source.finish()
            expected = PackedModel()
            with torch.no_grad():
                expected.load_weights(tensors)
            for receiver in members.values():
                assert receiver.version == step and not receiver.poisoned
                for name, value in receiver.model.named_parameters():
                    assert torch.equal(
                        value.view(torch.int16), dict(expected.named_parameters())[name].view(torch.int16)
                    )
    finally:
        source.close()


def test_tcp_failed_apply_cannot_ack_or_advance_source_baseline():
    source = MemorySource("127.0.0.1", timeout_s=0.2)
    encoder = MemoryDeltaEncoder()
    try:
        descriptor = source.publish(
            encoder.prepare(weights()),
            stream="s",
            version=0,
            base=-1,
            schema=encoder.pending_schema,
            members={"a"},
            reset=True,
        )

        def failed_apply(manifest, payloads):
            raise RuntimeError("load failed")

        with pytest.raises(RuntimeError, match="load failed"):
            receive_memory_update(descriptor, "a", failed_apply, timeout_s=1)
        with pytest.raises(TimeoutError, match="Missing memory ACKs"):
            source.wait(0.05)
        with pytest.raises(RuntimeError, match="unacknowledged"):
            source.finish()
        assert encoder.version == -1
        # Same immutable publication remains retryable after receiver recovery.
        receiver = MemoryModelReceiver(PackedModel())
        receive_memory_update(descriptor, "a", receiver.apply, timeout_s=1)
        encoder.commit(source.wait(1), {"a"})
        source.finish()
    finally:
        source.close()


def test_receiver_partial_load_is_poisoned_and_full_reset_recovers():
    class Broken(PackedModel):
        broken = True

        def load_weights(self, tensors):
            super().load_weights(tensors)
            if self.broken:
                raise RuntimeError("loader failed after writing")

    encoder = MemoryDeltaEncoder()
    buckets = encoder.prepare(weights())
    manifest = dict(
        stream="s",
        version=0,
        base=-1,
        schema=encoder.pending_schema,
        reset=True,
        update="first",
        buckets=[b.metadata() for b in buckets],
    )
    receiver = MemoryModelReceiver(Broken())
    with pytest.raises(RuntimeError, match="after writing"):
        receiver.apply(manifest, [b.payload for b in buckets])
    assert receiver.poisoned and receiver.version == -1
    with pytest.raises(RuntimeError, match="FULL reset"):
        receiver.apply(manifest | {"reset": False}, [b.payload for b in buckets])
    receiver.model.broken = False
    receiver.apply(manifest, [b.payload for b in buckets])
    assert not receiver.poisoned and receiver.version == 0


def test_receiver_rejects_base_mismatch_before_mutation_and_missing_parameter_coverage():
    encoder = MemoryDeltaEncoder()
    buckets = encoder.prepare(weights()[:-1])
    receiver = MemoryModelReceiver(PackedModel())
    manifest = dict(
        stream="s",
        version=0,
        base=-1,
        schema=encoder.pending_schema,
        reset=False,
        update="first",
        buckets=[b.metadata() for b in buckets],
    )
    with pytest.raises(ValueError, match="base version"):
        receiver.apply(manifest, [b.payload for b in buckets])
    assert not receiver.poisoned
    with pytest.raises(ValueError, match="cover every"):
        receiver.apply(manifest | {"reset": True}, [b.payload for b in buckets])
    assert receiver.poisoned


def test_overlapping_loader_routes_and_persistent_buffers_are_not_admitted():
    class Overlap(PackedModel):
        def load_weights(self, tensors):
            for _, tensor in tensors:
                self.qkv.copy_(tensor)

    encoder = MemoryDeltaEncoder()
    buckets = encoder.prepare([(name, torch.zeros(12, 4, dtype=torch.bfloat16)) for name in ("a", "b")])
    manifest = dict(
        stream="s",
        version=0,
        base=-1,
        schema=encoder.pending_schema,
        reset=True,
        update="first",
        buckets=[b.metadata() for b in buckets],
    )
    with pytest.raises(ValueError, match="Overlapping"):
        MemoryModelReceiver(Overlap()).apply(manifest, [b.payload for b in buckets])
    model = PackedModel()
    model.register_buffer("derived", torch.zeros(1))
    with pytest.raises(ValueError, match="persistent buffers"):
        MemoryModelReceiver(model).apply(manifest, [b.payload for b in buckets])


def test_group_failure_requires_new_stream_full_for_both_applied_and_failed_members():
    members = [MemoryModelReceiver(PackedModel()) for _ in range(2)]
    tensors = weights()

    def update(encoder, stream, step, reset):
        buckets = encoder.prepare(tensors, full=reset)
        manifest = dict(
            stream=stream,
            version=step,
            base=step - 1,
            schema=encoder.pending_schema,
            reset=reset,
            update=f"{stream}-{step}",
            buckets=[b.metadata() for b in buckets],
        )
        return manifest, [b.payload for b in buckets]

    encoder = MemoryDeltaEncoder()
    seed, payloads = update(encoder, "old", 0, True)
    for member in members:
        member.apply(seed, payloads)
    encoder.commit({"a", "b"}, {"a", "b"})
    tensors[0][1][0, 0] = 1
    delta, payloads = update(encoder, "old", 1, False)
    members[0].apply(delta, payloads)
    members[1].model.qkv.data = members[1].model.qkv.data.clone()
    with pytest.raises(ValueError, match="storage changed"):
        members[1].apply(delta, payloads)
    assert encoder.version == 0
    # Discard the failed stream, including its source/encoder; do not advance
    # just the successful receiver or try a different payload at old version 1.
    replacement = MemoryDeltaEncoder()
    reset, payloads = update(replacement, "new", 0, True)
    for member in members:
        member.apply(reset, payloads)
    replacement.commit({"a", "b"}, {"a", "b"})
    assert all(m.stream == "new" and m.version == 0 and not m.poisoned for m in members)


def test_source_close_releases_connections_waiting_for_ack():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from time import monotonic

    source = MemorySource("127.0.0.1", timeout_s=60)
    encoder = MemoryDeltaEncoder()
    descriptor = source.publish(
        encoder.prepare(weights()),
        stream="s",
        version=0,
        base=-1,
        schema=encoder.pending_schema,
        members={"a"},
        reset=True,
    )
    entered, release = Event(), Event()

    def stalled_apply(manifest, payloads):
        entered.set()
        assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(receive_memory_update, descriptor, "a", stalled_apply, timeout_s=5)
        try:
            assert entered.wait(2)
            started = monotonic()
            source.close()
            assert monotonic() - started < 3
            assert not source._connections and not source._buckets and source._pending is None
        finally:
            release.set()
        with pytest.raises((ConnectionError, OSError)):
            future.result(timeout=2)


def test_custom_loader_uses_control_tensor_and_real_tcp():
    import json

    from relax.backends.sglang.memory_delta_loader import load_memory_weights

    source = MemorySource("127.0.0.1", timeout_s=2)
    encoder = MemoryDeltaEncoder()
    model = PackedModel()
    try:
        descriptor = source.publish(
            encoder.prepare(weights()),
            stream="s",
            version=0,
            base=-1,
            schema=encoder.pending_schema,
            members={"a"},
            reset=True,
        )
        control = json.dumps(dict(descriptor=descriptor, member="a", timeout_s=2, host_budget_bytes=4096))
        load_memory_weights(model, [(control, torch.empty(0))])
        assert model._relax_memory_receiver.version == 0
        encoder.commit(source.wait(1), {"a"})
        source.finish()
        with pytest.raises(ValueError, match="empty control"):
            load_memory_weights(model, [(control, torch.zeros(1))])
    finally:
        source.close()


def test_memory_configuration_rejects_unsupported_execution_modes(tmp_path):
    import json
    from types import SimpleNamespace

    from relax.distributed.weight_sync.config import MemorySyncConfig, validate_memory_weight_sync_args

    path = tmp_path / "memory.json"
    path.write_text(json.dumps(dict(stream="test")))
    assert MemorySyncConfig.load(path).stream == "test"
    base = dict(memory_weight_sync_config=str(path), sglang_disable_cuda_graph=True)
    validate_memory_weight_sync_args(SimpleNamespace(**base))
    for override, message in [
        (dict(colocate=True), "does not support"),
        (dict(tensor_model_parallel_size=2), "training TP/PP/CP"),
        (dict(disk_weight_sync_config="disk.json"), "Select one"),
    ]:
        with pytest.raises(ValueError, match=message):
            validate_memory_weight_sync_args(SimpleNamespace(**(base | override)))
    path.write_text(json.dumps(dict(stream="test", bucket_bytes=True)))
    with pytest.raises(ValueError):
        MemorySyncConfig.load(path)
