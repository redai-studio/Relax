# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Expert IDs, gate/up order, changed weights and replacement process
groups."""

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def module():
    from relax.backends.megatron.weight_update import expert_gather

    return expert_gather


def sources():
    return [
        SimpleNamespace(
            name=f"module.module.language_model.decoder.layers.{layer}.mlp.experts.linear_fc{fc}.weight{e}",
            dtype=torch.bfloat16,
            shape=(8, 6) if fc == 1 else (6, 4),
            src_rank=(7, 2)[e // 12],
        )
        for layer in (0, 10, 2)
        for fc in (1, 2)
        for e in range(24)
    ]


def test_numeric_expert_order_and_group_rank_order(module):
    infos = sorted(sources(), key=lambda info: info.name)
    buckets = module.bucket_expert_projections(infos, 24, [7, 2])
    assert len(buckets) == 6
    assert [bucket[0].name.split("layers.")[1].split(".")[0] for bucket in buckets] == ["0", "0", "2", "2", "10", "10"]
    for bucket in buckets:
        assert [int(info.name.rsplit("weight", 1)[1]) for info in bucket] == list(range(24))
        assert [info.src_rank for info in bucket] == [7] * 12 + [2] * 12


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "owner", "shape", "dtype", "mtp", "projection"])
def test_invalid_source_layout_fails_before_payload_collectives(module, mutation):
    infos = sources()
    if mutation == "missing":
        infos.pop()
    elif mutation == "duplicate":
        infos.append(infos[0])
    elif mutation == "owner":
        infos[0].src_rank = 2
    elif mutation == "shape":
        infos[0].shape = (6, 8)
    elif mutation == "dtype":
        infos[0].dtype = torch.float32
    elif mutation == "mtp":
        infos[0].name = infos[0].name.replace("decoder.layers.0", "mtp.layers.0.transformer_layer")
    else:
        infos = [info for info in infos if ".linear_fc2." not in info.name]
    with pytest.raises(ValueError):
        module.bucket_expert_projections(infos, 24, [7, 2])


def peers(projection):
    output = []
    for owner in range(2):
        items = []
        for expert in range(owner * 12, (owner + 1) * 12):
            for part in ("gate", "up") if projection == "gate_up" else ("down",):
                shape = (4, 6) if part != "down" else (6, 4)
                tensor = (torch.arange(24).reshape(shape) + expert * 32 + (1000 if part == "up" else 0)).to(
                    torch.bfloat16
                )
                name = f"model.language_model.layers.2.mlp.experts.{expert}.{part}_proj.weight"
                items.append((name, tensor))
        output.append(sorted(items))
    return output


def flattened(peer, projection):
    by_name = dict(peer)
    ids = sorted({int(name.split(".experts.")[1].split(".")[0]) for name in by_name})
    return torch.cat(
        [
            by_name[f"model.language_model.layers.2.mlp.experts.{expert}.{part}_proj.weight"].reshape(-1)
            for expert in ids
            for part in (("gate", "up") if projection == "gate_up" else ("down",))
        ]
    )


@pytest.mark.parametrize("projection", ["gate_up", "down"])
@pytest.mark.parametrize("rank", [0, 1])
def test_packed_values_lifetime_and_fresh_group(module, monkeypatch, projection, rank):
    inputs = peers(projection)
    calls = []
    monkeypatch.setattr(module.dist, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: group.rank)

    def metadata(result, local, *, group):
        assert local == module._layout(inputs[group.rank])
        result[:] = [module._layout(peer) for peer in inputs]
        calls.append(("metadata", group))

    def gather(output, local, *, group, async_op):
        assert async_op and torch.equal(local, flattened(inputs[group.rank], projection))
        output.copy_(torch.cat([flattened(peer, projection) for peer in inputs]))
        calls.append(("payload", group))
        return SimpleNamespace(wait=lambda: calls.append(("wait", group)))

    monkeypatch.setattr(module.dist, "all_gather_object", metadata)
    monkeypatch.setattr(module.dist, "all_gather_into_tensor", gather)
    plan = module.PackedExpertGather(24)
    first_group = SimpleNamespace(ranks=[7, 2], rank=rank)
    [(name, value)] = plan(inputs[rank], group=first_group, device="cpu")
    assert name == f"model.language_model.layers.2.mlp.experts.{projection}_proj"
    assert tuple(value.shape) == ((24, 8, 6) if projection == "gate_up" else (24, 6, 4))
    expected = torch.cat([flattened(peer, projection) for peer in inputs]).view(value.shape)
    assert torch.equal(value, expected)
    frozen = value.clone()
    for peer in inputs:
        for _, tensor in peer:
            tensor.add_(16)
    second_group = SimpleNamespace(ranks=[7, 2], rank=rank)
    [(_, updated)] = plan(inputs[rank], group=second_group, device="cpu")
    assert torch.equal(value, frozen) and not torch.equal(value, updated)
    assert torch.equal(updated, torch.cat([flattened(peer, projection) for peer in inputs]).view(value.shape))
    assert sum(kind == "metadata" for kind, _ in calls) == 1
    assert [group for kind, group in calls if kind == "payload"] == [first_group, second_group]
    assert not any(isinstance(v, torch.Tensor) or v is first_group for v in vars(plan).values())
    with pytest.raises(ValueError, match="layout changed"):
        plan(inputs[rank][:-1], group=second_group, device="cpu")


@pytest.mark.parametrize("mutation", ["wrong_owner", "missing_up", "duplicate", "mixed_layer", "mixed_projection"])
def test_invalid_converted_layout(module, mutation):
    inputs = peers("gate_up")
    if mutation == "wrong_owner":
        inputs.reverse()
    elif mutation == "missing_up":
        inputs[1].pop()
    elif mutation == "duplicate":
        inputs[0].append(inputs[0][0])
    elif mutation == "mixed_layer":
        name, tensor = inputs[0][0]
        inputs[0][0] = (name.replace("layers.2", "layers.3"), tensor)
    else:
        inputs[1] = peers("down")[1]
    plan = module.PackedExpertGather(24)
    with pytest.raises(ValueError):
        plan._prepare([module._layout(peer) for peer in inputs], (7, 2))


@pytest.mark.parametrize(
    "source_prefix,hf_prefix",
    [
        ("module.module.decoder.layers", "model.layers"),
        ("module.module.language_model.decoder.layers", "model.language_model.layers"),
    ],
)
def test_deepseek_and_multimodal_names_use_same_projection_packing(module, source_prefix, hf_prefix):
    infos = sources()
    for info in infos:
        info.name = info.name.replace("module.module.language_model.decoder.layers", source_prefix)
    assert len(module.bucket_expert_projections(infos, 24, [7, 2])) == 6
    inputs = peers("gate_up")
    inputs = [
        [(name.replace("model.language_model.layers", hf_prefix), tensor) for name, tensor in peer] for peer in inputs
    ]
    plan = module.PackedExpertGather(24)
    plan._prepare([module._layout(peer) for peer in inputs], (7, 2))
    assert plan._name == f"{hf_prefix}.2.mlp.experts.gate_up_proj"
    assert plan._shape == (24, 8, 6)


@pytest.mark.parametrize("projection", ["gate_up", "down"])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("start,end", [(0, 5), (9, 14), (9, 15), (12, 24), (22, 24)])
def test_partial_projection_empty_owners_and_uneven_gather(module, monkeypatch, projection, rank, start, end):
    inputs = [
        [(name, tensor) for name, tensor in peer if start <= int(name.split(".experts.")[1].split(".")[0]) < end]
        for peer in peers(projection)
    ]
    monkeypatch.setattr(module.dist, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: group.rank)
    metadata_calls = []

    def metadata(result, local, *, group):
        result[:] = [module._layout(peer) for peer in inputs]
        metadata_calls.append(group)

    def flat(peer):
        return flattened(peer, projection) if peer else torch.empty(0, dtype=torch.bfloat16)

    def gather(outputs, local, *, group, async_op):
        assert async_op and torch.equal(local, flat(inputs[rank]))
        for output, peer in zip(outputs, inputs, strict=True):
            output.copy_(flat(peer))
        return SimpleNamespace(wait=lambda: None)

    def broadcast(output, *, src, group, async_op):
        assert async_op
        output.copy_(flat(inputs[group.ranks.index(src)]))
        return SimpleNamespace(wait=lambda: None)

    def uniform(output, local, *, group, async_op):
        assert async_op and torch.equal(local, flat(inputs[rank]))
        output.copy_(torch.cat([flat(peer) for peer in inputs]))
        return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(module.dist, "all_gather_object", metadata)
    monkeypatch.setattr(module.dist, "all_gather", gather)
    monkeypatch.setattr(module.dist, "all_gather_into_tensor", uniform)
    monkeypatch.setattr(module.dist, "broadcast", broadcast)
    plan = module.PackedExpertGather(24, start, end)
    first = None
    for step in range(3):
        group = SimpleNamespace(ranks=[7, 2], rank=rank)
        [(name, actual)] = plan(inputs[rank], group=group, device="cpu")
        assert name == f"model.language_model.layers.2.mlp.experts.{projection}_proj[{start}:{end}]"
        assert actual.shape[0] == end - start
        assert torch.equal(actual.flatten(), torch.cat([flat(peer) for peer in inputs]))
        if first is None:
            first, frozen = actual, actual.clone()
        assert torch.equal(first, frozen)
        for peer in inputs:
            for _, tensor in peer:
                tensor.add_(16)
    assert len(metadata_calls) == 1


@pytest.mark.parametrize("projection", ["gate_up", "down"])
@pytest.mark.parametrize("rank", [0, 1])
def test_owner_balanced_ranges_share_one_storage_and_one_collective(module, monkeypatch, projection, rank):
    ranges = [(2, 5), (15, 18)]
    inputs = [
        [(name, tensor) for name, tensor in peer if start <= int(name.split(".experts.")[1].split(".")[0]) < end]
        for peer, (start, end) in zip(peers(projection), ranges, strict=True)
    ]
    monkeypatch.setattr(module.dist, "get_process_group_ranks", lambda group: [7, 2])
    monkeypatch.setattr(module.dist, "get_rank", lambda group: rank)
    monkeypatch.setattr(
        module.dist,
        "all_gather_object",
        lambda result, local, **kw: result.__setitem__(slice(None), [module._layout(peer) for peer in inputs]),
    )
    calls = []

    def gather(output, local, *, group, async_op):
        assert async_op and torch.equal(local, flattened(inputs[rank], projection))
        output.copy_(torch.cat([flattened(peer, projection) for peer in inputs]))
        calls.append(group)
        return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(module.dist, "all_gather_into_tensor", gather)
    plan = module.PackedExpertGather(24, expert_ranges=ranges)
    actual = plan(inputs[rank], group="current-ep", device="cpu")
    assert calls == ["current-ep"] and len(actual) == 2
    assert actual[0][1].untyped_storage().data_ptr() == actual[1][1].untyped_storage().data_ptr()
    for (name, tensor), peer, (start, end) in zip(actual, inputs, ranges, strict=True):
        assert name.endswith(f".{projection}_proj[{start}:{end}]")
        assert torch.equal(tensor.flatten(), flattened(peer, projection))
