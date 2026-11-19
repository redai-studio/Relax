# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU contracts for the production HF receiver, not real SGLang acceptance."""

import hashlib
import json

import pytest
import torch
from torch import nn

from relax.backends.sglang.hf_memory_loader import HFMemoryReceiver, load_hf_memory_weights
from relax.distributed.weight_sync.memory_transport import MemorySource
from relax.distributed.weight_sync.streaming_delta import SnapshotBudget, StreamingDeltaEncoder


class OrdinaryModel(nn.Module):
    """An ordinary per-item loader with expert slices and a real transform."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(64, dtype=torch.bfloat16), requires_grad=False)
        self.precise = nn.Parameter(torch.zeros(64, dtype=torch.float32), requires_grad=False)
        self.decay = nn.Parameter(torch.zeros(64, dtype=torch.float32), requires_grad=False)
        self.mlp = nn.Module()
        self.mlp.experts = nn.Module()
        self.mlp.experts.w13_weight = nn.Parameter(torch.zeros(2, 64, dtype=torch.bfloat16), requires_grad=False)
        self.skip = None
        self.buffer_inputs = False
        self.calls = 0

    def load_weights(self, weights):
        self.calls += 1
        if self.buffer_inputs:
            weights = list(weights)
        for name, tensor in weights:
            if name == self.skip:
                continue
            if name == "hf.embedding":
                self.embedding.copy_(tensor)
            elif name == "hf.precise":
                self.precise.copy_(tensor)
            elif name == "hf.decay":
                self.decay.copy_(-torch.exp(tensor))
            else:
                index = {"hf.expert.0": 0, "hf.expert.1": 1}[name]
                self.mlp.experts.w13_weight[index].copy_(tensor)


def _values() -> list[tuple[str, torch.Tensor]]:
    return [
        ("hf.embedding", torch.zeros(64, dtype=torch.bfloat16)),
        ("hf.expert.0", torch.arange(64, dtype=torch.bfloat16)),
        ("hf.expert.1", torch.arange(64, dtype=torch.bfloat16) + 2),
        ("hf.precise", torch.zeros(64, dtype=torch.float32)),
        ("hf.decay", torch.arange(64, dtype=torch.float32) / 100),
    ]


def _encoder() -> StreamingDeltaEncoder:
    return StreamingDeltaEncoder(SnapshotBudget(1 << 20), bucket_bytes=128)


def _receiver(model, **kwargs) -> HFMemoryReceiver:
    # The transform holds exp(input) and its negative: two FP32 tensors.
    return HFMemoryReceiver(model, loader_workspace_budget_bytes=2 * 64 * 4, **kwargs)


def _capture(encoder, values, *, stream="a", full=False):
    encoder.begin(full=full)
    for name, tensor in values:
        encoder.add(name, tensor)
    buckets = encoder.finish()
    manifest = dict(
        stream=stream,
        version=encoder.version + 1,
        base=encoder.version,
        schema=encoder.pending_schema,
        reset=encoder.version == -1,
        buckets=[bucket.metadata() for bucket in buckets],
    )
    manifest["update"] = hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return manifest, buckets


def _commit(encoder) -> None:
    encoder.prepare_commit({"worker"}, {"worker"})
    encoder.commit()


def _apply(receiver, manifest, buckets) -> None:
    receiver.apply(manifest, [bucket.payload for bucket in buckets])


def _assert_bits(actual, expected) -> None:
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _assert_full_oracle(model, values) -> None:
    expected = OrdinaryModel()
    with torch.no_grad():
        expected.load_weights(values)
    for name, tensor in model.named_parameters():
        _assert_bits(tensor, dict(expected.named_parameters())[name])


def test_real_tcp_mixed_bits_experts_transform_and_full_fallback() -> None:
    model, values, encoder = OrdinaryModel(), _values(), _encoder()
    source = MemorySource("127.0.0.1", timeout_s=3)
    try:
        for step in range(3):
            manifest, buckets = _capture(encoder, values, full=step == 2)
            if step == 1:
                assert any(bucket.encoding == "sparse" for bucket in buckets)
            if step == 2:
                assert all(bucket.encoding == "full" for bucket in buckets)
                assert not manifest["reset"]
            descriptor = source.publish(
                buckets,
                stream=manifest["stream"],
                version=manifest["version"],
                base=manifest["base"],
                schema=manifest["schema"],
                members={"worker"},
                reset=manifest["reset"],
                group_update="a" * 64,
            )
            control = dict(
                descriptor=descriptor,
                member="worker",
                stream="a",
                plan_fingerprint="a" * 64,
                timeout_s=3,
                host_budget_bytes=1 << 20,
                receiver_gpu_budget_bytes=1 << 20,
                loader_workspace_budget_bytes=512,
            )
            load_hf_memory_weights(model, [(json.dumps(control), torch.empty(0))])
            assert source.wait(1) == {"worker"}
            _commit(encoder)
            source.finish()
            receiver = model._relax_hf_memory_receiver
            assert receiver.version == step and not receiver.poisoned and model.calls == step + 1
            _assert_full_oracle(model, values)
            cached = dict(receiver._named_weights())
            for name, tensor in values:
                _assert_bits(cached[name], tensor)
                assert all(
                    cached[name].untyped_storage()._cdata != p.untyped_storage()._cdata for p in model.parameters()
                )
            # Preserve signed zero and distinct NaN payloads in both typed
            # caches; the model's arithmetic is compared with its FULL path.
            values[0][1].view(torch.int16)[0] = -32768
            values[0][1].view(torch.int16)[1] = 0x7F81 + step
            values[3][1].view(torch.int32)[0] = -(2**31)
            values[3][1].view(torch.int32)[1] = 0x7F800001 + step
            values[1][1][2] += 1
            values[4][1][2] += 0.125
    finally:
        source.close()
        encoder.close()


def test_missing_write_poisons_stream_and_new_full_restores_same_storage() -> None:
    model, values, encoder = OrdinaryModel(), _values(), _encoder()
    receiver = _receiver(model)
    manifest, buckets = _capture(encoder, values)
    _apply(receiver, manifest, buckets)
    _commit(encoder)
    model.skip = "hf.precise"
    values[0][1][2] = 5
    manifest, buckets = _capture(encoder, values)
    with pytest.raises(ValueError, match="no protected parameter writes"):
        _apply(receiver, manifest, buckets)
    assert receiver.poisoned and receiver.version == 0
    model.skip = None
    with pytest.raises(ValueError, match="retired"):
        _apply(receiver, manifest, buckets)
    recovery = _encoder()
    fresh, full = _capture(recovery, values, stream="recovered")
    _apply(receiver, fresh, full)
    assert not receiver.poisoned and receiver.stream == "recovered"
    _assert_full_oracle(model, values)
    with pytest.raises(ValueError, match="retired"):
        _apply(receiver, manifest, buckets)
    encoder.close()
    recovery.close()


@pytest.mark.parametrize("failure", ["checksum", "base", "layout", "duplicate", "storage"])
def test_preflight_rejects_before_mutating_any_cache_or_model(failure) -> None:
    model, values, encoder = OrdinaryModel(), _values(), _encoder()
    receiver = _receiver(model)
    manifest, buckets = _capture(encoder, values)
    _apply(receiver, manifest, buckets)
    _commit(encoder)
    values[0][1][0] = 17
    manifest, buckets = _capture(encoder, values)
    payloads = [bucket.payload for bucket in buckets]
    if failure == "checksum":
        # Damage the final frame, after valid changed frames have arrived.
        payloads[-1] = b"x" + payloads[-1][1:] if payloads[-1] else b"x"
    elif failure == "base":
        manifest["base"], manifest["version"] = 4, 5
    elif failure == "layout":
        manifest["buckets"][0]["names"] = (("renamed", (64,)),)
    elif failure == "duplicate":
        manifest["buckets"][1]["names"] = (("hf.embedding", (64,)),)
    else:
        model.embedding = nn.Parameter(model.embedding.clone(), requires_grad=False)
    before_cache = [cache.clone() for cache in receiver._cache]
    before_model = {name: p.clone() for name, p in model.named_parameters()}
    with pytest.raises(ValueError):
        receiver.apply(manifest, payloads)
    assert receiver.poisoned and receiver.version == 0
    for before, after in zip(before_cache, receiver._cache, strict=True):
        _assert_bits(before, after)
    for name, parameter in model.named_parameters():
        _assert_bits(parameter, before_model[name])
    encoder.close()


def test_budget_counts_complete_cache_and_sparse_scratch_before_allocation() -> None:
    model, values, encoder = OrdinaryModel(), _values(), _encoder()
    manifest, buckets = _capture(encoder, values)
    cache_bytes = sum(t.numel() * t.element_size() for _, t in values)
    full_peak = cache_bytes + max(len(bucket.payload) for bucket in buckets) + 512
    receiver = _receiver(model, receiver_gpu_budget_bytes=full_peak - 1)
    with pytest.raises(ValueError, match="Complete HF cache"):
        _apply(receiver, manifest, buckets)
    assert not receiver._cache and model.calls == 0
    receiver = _receiver(model, receiver_gpu_budget_bytes=full_peak)
    _apply(receiver, manifest, buckets)
    _commit(encoder)
    assert receiver.cache_bytes == cache_bytes and receiver.peak_bytes == full_peak
    # A 64-element BF16 bucket with 20 changes uses sparse encoding (120
    # bytes), but its 160-byte index scratch pushes the peak above the seed.
    values[0][1][:20] = 1
    manifest, buckets = _capture(encoder, values)
    assert buckets[0].encoding == "sparse"
    with pytest.raises(ValueError, match="sparse scratch"):
        _apply(receiver, manifest, buckets)
    assert model.calls == 1
    encoder.close()


def test_buffered_loader_and_persistent_buffer_require_explicit_contract() -> None:
    values, encoder = _values(), _encoder()
    manifest, buckets = _capture(encoder, values)
    model = OrdinaryModel()
    model.buffer_inputs = True
    receiver = _receiver(model)
    with pytest.raises(ValueError, match="without an active HF source"):
        _apply(receiver, manifest, buckets)
    assert receiver.poisoned
    model = OrdinaryModel()
    model.register_buffer("derived", torch.ones(1))
    with pytest.raises(ValueError, match="persistent"):
        _receiver(model)
    encoder.close()


def test_initialization_failure_retires_stream_before_any_transport() -> None:
    model = OrdinaryModel()
    control = dict(stream="failed", plan_fingerprint="a" * 64, host_budget_bytes=1024, receiver_gpu_budget_bytes=1024)
    with pytest.raises(KeyError, match="loader_workspace_budget_bytes"):
        load_hf_memory_weights(model, [(json.dumps(control), torch.empty(0))])
    control["loader_workspace_budget_bytes"] = 512
    with pytest.raises(ValueError, match="retired"):
        load_hf_memory_weights(model, [(json.dumps(control), torch.empty(0))])
    assert model.calls == 0


def test_tcp_bad_frame_before_callback_has_no_ack_and_retires_control_stream() -> None:
    model, values, encoder = OrdinaryModel(), _values(), _encoder()
    manifest, buckets = _capture(encoder, values)
    source = MemorySource("127.0.0.1", timeout_s=1)
    try:
        descriptor = source.publish(
            buckets, stream="broken", version=0, base=-1, schema=manifest["schema"], members={"worker"}, reset=True
        )
        # Inject corruption after publication, preserving the authenticated
        # manifest. This exercises the actual TCP checksum rejection path.
        source._buckets[0].payload = b"x" + source._buckets[0].payload[1:]
        control = dict(
            descriptor=descriptor,
            member="worker",
            stream="broken",
            plan_fingerprint="a" * 64,
            timeout_s=1,
            host_budget_bytes=1 << 20,
            receiver_gpu_budget_bytes=1 << 20,
            loader_workspace_budget_bytes=512,
        )
        request = [(json.dumps(control), torch.empty(0))]
        with pytest.raises(ValueError, match="checksum"):
            load_hf_memory_weights(model, request)
        assert not source._acks and model.calls == 0
        receiver = model._relax_hf_memory_receiver
        assert receiver.poisoned
        with pytest.raises(ValueError, match="retired"):
            load_hf_memory_weights(model, request)
        recovery = _encoder()
        try:
            fresh, full = _capture(recovery, values, stream="fresh")
            _apply(receiver, fresh, full)
            assert not receiver.poisoned
        finally:
            recovery.close()
    finally:
        source.close()
        encoder.close()
