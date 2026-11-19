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
    def __init__(self, experts=4, hidden=2, intermediate=4, tp_size=2):
        super().__init__()
        self.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = torch.nn.Module()
        layer.mlp.experts.w13_weight = torch.nn.Parameter(
            torch.zeros(experts, 2 * intermediate // tp_size, hidden, dtype=torch.bfloat16)
        )
        layer.mlp.experts.w2_weight = torch.nn.Parameter(
            torch.zeros(experts, hidden, intermediate // tp_size, dtype=torch.bfloat16)
        )
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


def _publish(
    stack,
    encoders,
    root,
    version,
    *,
    wrong_nonce_owner=None,
    missing_expert=False,
    geometry=(4, 2, 4),
    tp_size=2,
    tp_rank=0,
    owners=2,
):
    experts, hidden, intermediate = geometry
    member = f"replica:tp:{tp_rank}"
    streams, sources, expected = [], [], {}
    for owner in range(owners):
        key = f"expert:{owner}"
        encoder = encoders.setdefault(key, StreamingDeltaEncoder(SnapshotBudget(1 << 20), bucket_bytes=128))
        encoder.begin()
        for projection, shape in (
            ("w13_weight", (experts, 2 * intermediate // tp_size, hidden)),
            ("w2_weight", (experts, hidden, intermediate // tp_size)),
        ):
            owner_count = experts // owners
            count = owner_count - 1 if missing_expert and owner == owners - 1 else owner_count
            name = f"model.layers.0.mlp.experts.{projection}"
            tensor = torch.full((count, *shape[1:]), 1 + owner, dtype=torch.bfloat16)
            if version:
                tensor.view(torch.int16).reshape(-1)[0] = -32768
            encoder.add(expert_slice_name(name, owner * owner_count, shape), tensor)
            expected.setdefault(name, []).append(tensor)
        source = MemorySource("127.0.0.1", timeout_s=2)
        stack.callback(source.close)
        stream = f"{root}:expert:{owner}:tp:{tp_rank}"
        descriptor = source.publish(
            encoder.finish(),
            stream=stream,
            version=version,
            base=version - 1,
            schema=encoder.pending_schema,
            members={member},
            reset=version == 0,
            group_update="wrong" if owner == wrong_nonce_owner else f"update-{version}",
        )
        streams.append(dict(stream=stream, descriptor=descriptor, member=member))
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
        members={member},
        reset=version == 0,
        group_update=f"update-{version}",
    )
    streams.append(dict(stream=stream, descriptor=descriptor, member=member))
    sources.append((encoder, source))
    control = dict(
        stream=root,
        version=version,
        group_update=f"update-{version}",
        reset=version == 0,
        tp_rank=tp_rank,
        tp_size=tp_size,
        ep_size=1,
        moe_dp_size=1,
        owners=owners,
        streams=streams,
    )
    return control, sources, {name: torch.cat(values) for name, values in expected.items()}


@pytest.mark.parametrize("geometry,tp_size,owners", [((6, 3, 4), 1, 3), ((4, 2, 4), 2, 2), ((8, 3, 8), 4, 2)])
def test_sharded_real_tcp_applies_every_expert_and_preserves_fp32_bits(geometry, tp_size, owners):
    model = TinyMoE(*geometry, tp_size)
    receiver = ShardedMemoryReceiver(model, gpu_budget_bytes=1 << 20)
    encoders = {}
    with ExitStack() as stack:
        for version in range(2):
            control, sources, expected = _publish(
                stack,
                encoders,
                "group",
                version,
                geometry=geometry,
                tp_size=tp_size,
                tp_rank=tp_size - 1,
                owners=owners,
            )
            result = receiver.apply_group(control, _receive)
            assert result["streams"] == owners + 1 and receiver.version == version and not receiver.poisoned
            assert result["stream"] == "group" and result["group_update"] == f"update-{version}"
            assert result["schemas"] == {
                item["stream"]: encoder.pending_schema
                for item, (encoder, _) in zip(control["streams"], sources, strict=True)
            }
            for name, value in expected.items():
                assert torch.equal(model.get_parameter(name).view(torch.uint8), value.view(torch.uint8))
            for encoder, source in sources:
                encoder.prepare_commit(source.wait(1), {f"replica:tp:{tp_size - 1}"})
                encoder.commit()
                source.finish()
    for encoder in encoders.values():
        encoder.close()


@pytest.mark.parametrize("failure", ["mixed-update", "missing-owner", "stream-set", "budget", "receiver-ep"])
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
        if failure == "budget":
            receiver.gpu_budget_bytes = 1
        if failure == "receiver-ep":
            control["ep_size"] = 2
        with pytest.raises((RuntimeError, ValueError)):
            receiver.apply_group(control, _receive)
        assert receiver.version == -1
        if failure not in ("stream-set", "receiver-ep"):
            assert receiver.poisoned
            with pytest.raises(ValueError, match="new group stream identity"):
                receiver.apply_group(control, _receive)
        bad_delta = dict(control, version=1, reset=False)
        with pytest.raises(ValueError):
            receiver.apply_group(bad_delta, _receive)
        for encoder in encoders.values():
            encoder.close()
        fresh = {}
        receiver.gpu_budget_bytes = 1 << 20
        control, _, expected = _publish(stack, fresh, "replacement", 0)
        receiver.apply_group(control, _receive)
        assert receiver.version == 0 and not receiver.poisoned
        for name, tensor in expected.items():
            assert torch.equal(receiver.model.get_parameter(name).view(torch.uint8), tensor.view(torch.uint8))
        for encoder in fresh.values():
            encoder.close()
