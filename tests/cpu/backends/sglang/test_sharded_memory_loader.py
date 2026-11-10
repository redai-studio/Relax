# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Native expert owner coverage and mixed-dtype TCP group failure semantics."""

from contextlib import ExitStack

import pytest
import torch

from relax.backends.sglang.sharded_memory_loader import ShardedMemoryReceiver
from relax.backends.sglang.weight_loader import expert_slice_name
from relax.distributed.weight_sync.memory_transport import MemorySource, receive_memory_update
from relax.distributed.weight_sync.streaming_delta import SnapshotBudget, StreamingDeltaEncoder


class TinyMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = torch.nn.Module()
        layer.mlp.experts.w13_weight = torch.nn.Parameter(torch.zeros(4, 4, 2, dtype=torch.bfloat16))
        layer.mlp.experts.w2_weight = torch.nn.Parameter(torch.zeros(4, 2, 2, dtype=torch.bfloat16))
        self.model.layers = torch.nn.ModuleList([layer])
        self.dense = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
        self.A_log = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def load_weights(self, weights):
        with torch.no_grad():
            for name, tensor in weights:
                self.get_parameter(name).copy_(tensor)


def _receive(descriptor, member, apply):
    return receive_memory_update(
        descriptor, member, apply, timeout_s=2, host_budget_bytes=1 << 20, private_buffers=True
    )


def _publish(stack, encoders, root, version, *, wrong_nonce_owner=None, missing_expert=False):
    streams, sources, expected = [], [], {}
    for owner in range(2):
        key = f"expert:{owner}"
        encoder = encoders.setdefault(key, StreamingDeltaEncoder(SnapshotBudget(1 << 20), bucket_bytes=128))
        encoder.begin()
        for projection, shape in (("w13_weight", (4, 4, 2)), ("w2_weight", (4, 2, 2))):
            count = 1 if missing_expert and owner == 1 else 2
            name = f"model.layers.0.mlp.experts.{projection}"
            tensor = torch.full((count, *shape[1:]), 1 + owner, dtype=torch.bfloat16)
            if version:
                tensor.view(torch.int16).reshape(-1)[0] = -32768
            encoder.add(expert_slice_name(name, owner * 2, shape), tensor)
            expected.setdefault(name, []).append(tensor)
        source = MemorySource("127.0.0.1", timeout_s=2)
        stack.callback(source.close)
        stream = f"{root}:expert:{owner}:tp:0"
        descriptor = source.publish(
            encoder.finish(),
            stream=stream,
            version=version,
            base=version - 1,
            schema=encoder.pending_schema,
            members={"replica:tp:0"},
            reset=version == 0,
            group_update="wrong" if owner == wrong_nonce_owner else f"update-{version}",
        )
        streams.append(dict(stream=stream, descriptor=descriptor, member="replica:tp:0"))
        sources.append((encoder, source))
    encoder = encoders.setdefault("nonexpert", StreamingDeltaEncoder(SnapshotBudget(1 << 20), bucket_bytes=128))
    encoder.begin()
    expected["dense"] = [torch.full((2, 2), 3, dtype=torch.bfloat16)]
    if version:
        expected["dense"][0].view(torch.int16)[0, 0] = -32768
    # Exact FP32 raw bits include a signed zero and a NaN payload.
    expected["A_log"] = [torch.tensor([-(2**31), 0x7F800001 + version], dtype=torch.int32).view(torch.float32)]
    for name in ("dense", "A_log"):
        encoder.add(name, expected[name][0])
    source = MemorySource("127.0.0.1", timeout_s=2)
    stack.callback(source.close)
    stream = f"{root}:nonexpert"
    descriptor = source.publish(
        encoder.finish(),
        stream=stream,
        version=version,
        base=version - 1,
        schema=encoder.pending_schema,
        members={"replica:tp:0"},
        reset=version == 0,
        group_update=f"update-{version}",
    )
    streams.append(dict(stream=stream, descriptor=descriptor, member="replica:tp:0"))
    sources.append((encoder, source))
    control = dict(
        stream=root,
        version=version,
        group_update=f"update-{version}",
        reset=version == 0,
        tp_rank=0,
        tp_size=2,
        owners=2,
        streams=streams,
    )
    return control, sources, {name: torch.cat(values) for name, values in expected.items()}


def test_sharded_real_tcp_applies_every_expert_and_preserves_fp32_bits():
    model = TinyMoE()
    receiver = ShardedMemoryReceiver(model, gpu_budget_bytes=1 << 20)
    encoders = {}
    with ExitStack() as stack:
        for version in range(2):
            control, sources, expected = _publish(stack, encoders, "group", version)
            result = receiver.apply_group(control, _receive)
            assert result["streams"] == 3 and receiver.version == version and not receiver.poisoned
            for name, value in expected.items():
                assert torch.equal(model.get_parameter(name).view(torch.uint8), value.view(torch.uint8))
            for encoder, source in sources:
                encoder.prepare_commit(source.wait(1), {"replica:tp:0"})
                encoder.commit()
                source.finish()
    for encoder in encoders.values():
        encoder.close()


@pytest.mark.parametrize("failure", ["mixed-update", "missing-owner", "stream-set"])
def test_partial_group_failure_never_advances_and_requires_new_full(failure):
    receiver = ShardedMemoryReceiver(TinyMoE(), gpu_budget_bytes=1 << 20)
    with ExitStack() as stack:
        encoders = {}
        control, _, _ = _publish(
            stack,
            encoders,
            "failed",
            0,
            wrong_nonce_owner=1 if failure == "mixed-update" else None,
            missing_expert=failure == "missing-owner",
        )
        if failure == "stream-set":
            control["streams"] = control["streams"][:-1]
        with pytest.raises((RuntimeError, ValueError)):
            receiver.apply_group(control, _receive)
        assert receiver.version == -1
        if failure != "stream-set":
            assert receiver.poisoned
        bad_delta = dict(control, version=1, reset=False)
        with pytest.raises(ValueError):
            receiver.apply_group(bad_delta, _receive)
        for encoder in encoders.values():
            encoder.close()
        fresh = {}
        control, _, expected = _publish(stack, fresh, "replacement", 0)
        receiver.apply_group(control, _receive)
        assert receiver.version == 0 and not receiver.poisoned
        for name, tensor in expected.items():
            assert torch.equal(receiver.model.get_parameter(name).view(torch.uint8), tensor.view(torch.uint8))
        for encoder in fresh.values():
            encoder.close()
