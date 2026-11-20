# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Packing, tensor lifetime and metadata reuse across replacement groups."""

from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def module():
    from relax.backends.megatron.weight_update import expert_gather

    return expert_gather


def simulate_collectives(monkeypatch, module, peers):
    calls = []
    monkeypatch.setattr(module.dist, "get_process_group_ranks", lambda group: group.ranks)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: group.rank)

    def metadata(result, local, *, group):
        assert local == module._layout(peers[group.rank])
        result[:] = [module._layout(peer) for peer in peers]
        calls.append(("metadata", group))

    def flattened(peer):
        return torch.cat([tensor.reshape(-1) for _, tensor in peer]) if peer else torch.empty(0, dtype=torch.bfloat16)

    def wait():
        calls.append(("wait", None))

    def uniform(output, local, *, group, async_op):
        assert async_op and torch.equal(local, flattened(peers[group.rank]))
        output.copy_(torch.cat([flattened(peer) for peer in peers]))
        calls.append(("uniform", group))
        return SimpleNamespace(wait=wait)

    def uneven(outputs, local, *, group, async_op):
        assert async_op and torch.equal(local, flattened(peers[group.rank]))
        for output, peer in zip(outputs, peers, strict=True):
            output.copy_(flattened(peer))
        calls.append(("uneven", group))
        return SimpleNamespace(wait=wait)

    def broadcast(output, *, src, group, async_op):
        assert async_op
        output.copy_(flattened(peers[group.ranks.index(src)]))
        calls.append(("broadcast", group))
        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(module.dist, "all_gather_object", metadata)
    monkeypatch.setattr(module.dist, "all_gather_into_tensor", uniform)
    monkeypatch.setattr(module.dist, "all_gather", uneven)
    monkeypatch.setattr(module.dist, "broadcast", broadcast)
    return calls


@pytest.mark.parametrize(
    "sizes,collective", [([6, 6], "uniform"), ([6, 4], "uneven"), ([0, 6], "broadcast"), ([0, 0], None)]
)
@pytest.mark.parametrize("rank", [0, 1])
def test_gather_values_old_output_lifetime_and_reloaded_group(module, monkeypatch, sizes, collective, rank):
    peers = [
        [(f"expert.{index}", torch.arange(size, dtype=torch.bfloat16).view(2, -1).t())] if size else []
        for index, size in enumerate(sizes)
    ]
    calls = simulate_collectives(monkeypatch, module, peers)
    first_group = SimpleNamespace(ranks=[2, 7], rank=rank)
    plan = module.Bf16ExpertGather()
    actual = plan(peers[rank], group=first_group, device=torch.device("cpu"))
    expected = [entry for peer in peers for entry in peer]
    assert [name for name, _ in actual] == [name for name, _ in expected]
    assert all(torch.equal(tensor, other) for (_, tensor), (_, other) in zip(actual, expected, strict=True))
    frozen = [(name, tensor.clone()) for name, tensor in actual]
    for peer in peers:
        for _, tensor in peer:
            tensor.add_(10)
    replacement_group = SimpleNamespace(ranks=[2, 7], rank=rank)
    updated = plan(peers[rank], group=replacement_group, device=torch.device("cpu"))
    assert all(torch.equal(tensor, other) for (_, tensor), (_, other) in zip(actual, frozen, strict=True))
    assert all(torch.equal(tensor, other) for (_, tensor), (_, other) in zip(updated, expected, strict=True))
    assert sum(kind == "metadata" for kind, _ in calls) == 1
    if collective:
        assert [group for kind, group in calls if kind == collective] == [first_group, replacement_group]
        assert sum(kind == "wait" for kind, _ in calls) == 2
    assert not any(isinstance(value, torch.Tensor) or value is first_group for value in vars(plan).values())


def test_changed_layout_and_topology_rejected(module, monkeypatch):
    peers = [[("a", torch.ones(4, dtype=torch.bfloat16))], [("b", torch.ones(4, dtype=torch.bfloat16))]]
    calls = simulate_collectives(monkeypatch, module, peers)
    group = SimpleNamespace(ranks=[2, 7], rank=0)
    plan = module.Bf16ExpertGather()
    plan(peers[0], group=group, device=torch.device("cpu"))
    count = len(calls)
    for tensors, next_group in [
        ([("a", torch.ones(2, dtype=torch.bfloat16))], group),
        (peers[0], SimpleNamespace(ranks=[2, 8], rank=0)),
    ]:
        with pytest.raises(ValueError, match="layout changed"):
            plan(tensors, group=next_group, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="requires BF16"):
        plan([("a", torch.ones(4))], group=group, device=torch.device("cpu"))
    assert len(calls) == count


def test_owner_rounds_cover_every_input_within_budget(module):
    infos = [
        SimpleNamespace(name=str(index), src_rank=owner, size=size)
        for index, (owner, size) in enumerate([(7, 6), (2, 4), (7, 4), (2, 4), (7, 6)])
    ]
    rounds = module.bucket_experts_by_owner(infos, 20, [2, 7])
    assert [[info.name for info in bucket] for bucket in rounds] == [["1", "3", "0", "2"], ["4"]]
    assert sorted(id(info) for bucket in rounds for info in bucket) == sorted(map(id, infos))
    assert all(sum(info.size for info in bucket) <= 20 for bucket in rounds)
    assert module.bucket_experts_by_owner([], 20, [2, 7]) == []
    for invalid in [
        SimpleNamespace(name="bad", src_rank=9, size=1),
        SimpleNamespace(name="large", src_rank=2, size=21),
    ]:
        with pytest.raises(ValueError, match="Unsupported"):
            module.bucket_experts_by_owner([invalid], 20, [2, 7])


def test_large_experts_can_use_an_entire_round_budget(module):
    infos = [SimpleNamespace(name=str(rank), src_rank=rank, size=12) for rank in range(8)]
    rounds = module.bucket_experts_by_owner(infos, 32, list(range(8)))
    assert [len(bucket) for bucket in rounds] == [2, 2, 2, 2]
    assert [info for bucket in rounds for info in bucket] == infos
    assert all(sum(info.size for info in bucket) <= 32 for bucket in rounds)


def test_owner_down_projection_requires_exact_storage_and_canonical_coordinates(module):
    raw = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    source = "module.module.decoder.layers.2.mlp.experts.linear_fc2.weight7"
    target = "model.layers.2.mlp.experts.7.down_proj.weight"
    check = module.identity_down_projection
    assert check(source, raw, [(target, raw.reshape(2, 8))])
    multimodal_source = source.replace("decoder.", "language_model.decoder.")
    multimodal_target = target.replace("model.layers.", "model.language_model.layers.")
    assert check(multimodal_source, raw, [(multimodal_target, raw.reshape(2, 8))])
    assert not check(multimodal_source, raw, [(target, raw)])
    assert not check(source, raw, [(multimodal_target, raw)])
    assert not check(multimodal_source, raw, [(multimodal_target, raw.clone())])
    for output in (raw.t(), raw.clone(), raw.exp(), raw[:, :-1], raw.reshape(-1), raw.float(), raw._neg_view()):
        assert not check(source, raw, [(target, output)])
    for name in (target.replace(".7.", ".1."), target.replace(".2.", ".3."), target.replace("down_", "up_")):
        assert not check(source, raw, [(name, raw)])
    assert not check(source, raw, [(target, raw), (target, raw)])
    assert not check(source.replace("decoder", "mtp"), raw, [(target, raw)])
    assert not check(source, raw[1:], [(target, raw[1:])])


def test_fc1_owner_uses_actual_complete_canonical_bridge_outputs():
    from relax.backends.megatron.weight_update.expert_gather import canonical_owner_projection

    raw = torch.arange(64, dtype=torch.bfloat16).reshape(16, 4)
    name = "decoder.layers.7.mlp.experts.linear_fc1.weight3"
    prefix = "model.layers.7.mlp.experts.3."
    gate, up = raw[:8].clone(), raw[8:].clone()
    actual = [(prefix + "up_proj.weight", up), (prefix + "gate_proj.weight", gate)]
    assert canonical_owner_projection(name, raw, actual)
    assert not canonical_owner_projection(name, raw, actual[:1])
    assert not canonical_owner_projection(name.replace("weight3", "weight4"), raw, actual)
    assert not canonical_owner_projection(name, raw, [(k, t.T) for k, t in actual])
