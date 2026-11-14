# Copyright (c) 2026 Relax Authors. All Rights Reserved.
from math import prod

import pytest
import torch

from relax.backends.megatron.weight_update.shard_p2p_source import ExpertViewPlanCache, plan_expert_views
from relax.backends.sglang.shard_p2p_weight_loader import expert_shapes
from relax.distributed.checkpoint_service.backends.shard_p2p.ipc import (
    copy_rectangle,
)


def _cache_views(version=0, *, separate=False, padding=0):
    flat = torch.arange(48 + padding, dtype=torch.float32).to(torch.bfloat16) + version * 64
    prefix = "model.layers.0.mlp.experts.0."
    views = {
        prefix + "gate_proj.weight": flat[padding : padding + 16].view(4, 4),
        prefix + "up_proj.weight": flat[padding + 16 : padding + 32].view(4, 4),
        prefix + "down_proj.weight": flat[padding + 32 :].view(4, 4),
    }
    return {name: value.clone() for name, value in views.items()} if separate else views


def test_delivery_does_not_recoalesce_finalized_write_ids():
    from relax.distributed.checkpoint_service.backends.shard_p2p.plan import bind_delivery, coalesce_writes

    prefix = "model.layers.0.mlp.experts."
    geometry = dict(num_experts=1, hidden_size=2, intermediate_size=3, tp_size=1, num_layers=1, source_world_size=1)
    writes = [
        dict(
            id=str(index),
            target="tp-0",
            source="flat",
            source_rank=0,
            param=prefix + name,
            src_offset=offset,
            dst_offset=offset,
            src_stride=12,
            dst_stride=12,
            rows=rows,
            row_bytes=width,
        )
        for index, (name, offset, rows, width) in enumerate(
            [
                ("w13_weight", 0, 1, 4),
                ("w13_weight", 12, 1, 4),
                ("w13_weight", 4, 2, 8),
                ("w2_weight", 0, 1, 12),
            ]
        )
    ]
    finalized = coalesce_writes(writes)
    assert coalesce_writes(finalized) != finalized  # Column/row passes are not idempotent.
    # The second pass must be retained. Splitting it by its source-owner
    # boundary preserves even this non-idempotent case, including exact IDs.
    for size in (1, 2, 4, 8, 16):
        owners = [
            [dict(write, id=f"{rank}:{write['id']}", source_rank=rank) for write in finalized] for rank in range(size)
        ]
        assert [write for owner in owners for write in coalesce_writes(owner)] == coalesce_writes(
            [write for owner in owners for write in owner]
        )
    endpoint = dict(
        boot_id="00000000-0000-0000-0000-000000000000",
        hostname="host",
        pid=1,
        gpu_uuid="GPU-A",
        device_index=0,
        generation="generation",
    )
    binding = bind_delivery(
        finalized,
        geometry,
        [dict(rank=0, endpoint=endpoint)],
        endpoint,
        layouts={name: dict(nbytes=prod(shape) * 2) for name, shape in expert_shapes(geometry).items()},
    )
    assert binding["delivery"]["routes"]["cuda_ipc"] == [write["id"] for write in finalized]
    assert not binding["delivery"]["routes"]["nixl"]
    remote = dict(endpoint, boot_id="00000000-0000-0000-0000-000000000001")
    mixed = bind_delivery(
        finalized,
        geometry,
        [dict(rank=0, endpoint=remote)],
        endpoint,
        layouts={name: dict(nbytes=prod(shape) * 2) for name, shape in expert_shapes(geometry).items()},
    )
    assert mixed["delivery"]["routes"]["nixl"] == [write["id"] for write in finalized]
    assert not mixed["delivery"]["routes"]["cuda_ipc"]


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("sources_only", [False, True])
def test_expert_plan_cache_rebinds_changed_snapshots_without_retaining_storage(monkeypatch, separate, sources_only):
    from unittest.mock import Mock

    from relax.backends.megatron.weight_update import shard_p2p_source as module

    planner = Mock(wraps=module.plan_expert_views)
    monkeypatch.setattr(module, "plan_expert_views", planner)
    cache = ExpertViewPlanCache()
    geometry = dict(num_experts=1, hidden_size=4, intermediate_size=4, tp_size=2, source_rank=0)

    def assert_cpu_description(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert_cpu_description(key)
                assert_cpu_description(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                assert_cpu_description(item)
        else:
            assert value is None or isinstance(value, (str, int, bool))

    for version in range(6):
        views = _cache_views(version, separate=separate)
        if version and sources_only:
            sources = cache.bind_sources(views, **geometry)
        else:
            sources, plans = cache.bind(views, **geometry)
        for tp, plan in enumerate(plans):
            targets = {
                "model.layers.0.mlp.experts.w13_weight": torch.full((1, 4, 4), float("nan"), dtype=torch.bfloat16),
                "model.layers.0.mlp.experts.w2_weight": torch.full((1, 4, 2), float("nan"), dtype=torch.bfloat16),
            }
            for write in plan:
                copy_rectangle(sources[write["source"]], targets[write["param"]], write)
            gate, up, down = views.values()
            torch.testing.assert_close(
                targets["model.layers.0.mlp.experts.w13_weight"][0],
                torch.cat((gate[tp * 2 : tp * 2 + 2], up[tp * 2 : tp * 2 + 2])),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                targets["model.layers.0.mlp.experts.w2_weight"][0], down[:, tp * 2 : tp * 2 + 2], rtol=0, atol=0
            )
        assert_cpu_description(vars(cache))
    assert planner.call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        "dtype",
        "shape",
        "stride",
        "offset",
        "capacity",
        "missing",
        "extra",
        "split",
        "merge",
        "order",
        "mapping",
        "rank",
        "tp",
        "hidden",
        "intermediate",
        "experts",
    ],
)
def test_expert_plan_cache_rejects_live_layout_changes(change):
    cache = ExpertViewPlanCache()
    geometry = dict(num_experts=1, hidden_size=4, intermediate_size=4, tp_size=2, source_rank=0, up_proj_first=False)
    qualification = ("mapping",)
    cache.bind(_cache_views(separate=change == "merge"), qualification=qualification, **geometry)
    views = _cache_views()
    first = next(iter(views))
    if change == "dtype":
        views[first] = views[first].float()
    elif change == "shape":
        views[first] = views[first].view(2, 8)
    elif change == "stride":
        views[first] = views[first].t()
    elif change == "offset":
        views[first] = views[first].as_strided((4, 4), (4, 1), storage_offset=1)
    elif change == "capacity":
        views[first].untyped_storage().resize_(100)
    elif change == "missing":
        views.pop(first)
    elif change == "extra":
        views[first.replace(".0.gate", ".1.gate")] = views[first]
    elif change == "split":
        views[first] = views[first].clone()
    elif change == "order":
        geometry["up_proj_first"] = True
    elif change == "mapping":
        qualification = ("other",)
    elif change in ("rank", "tp", "hidden", "intermediate", "experts"):
        key = dict(
            rank="source_rank",
            tp="tp_size",
            hidden="hidden_size",
            intermediate="intermediate_size",
            experts="num_experts",
        )[change]
        geometry[key] += 1
    with pytest.raises(ValueError, match="native reseeding"):
        cache.bind_sources(views, qualification=qualification, **geometry)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "stride", "expert"])
def test_expert_plan_rejects_unsupported_views(invalid):
    tensor = torch.zeros(6, 8, dtype=torch.bfloat16)
    expert = 0
    if invalid == "dtype":
        tensor = tensor.float()
    elif invalid == "shape":
        tensor = tensor[:4]
    elif invalid == "stride":
        tensor = tensor.t()
    else:
        expert = 4
    with pytest.raises(ValueError):
        plan_expert_views(
            {f"model.layers.0.mlp.experts.{expert}.gate_proj.weight": tensor},
            num_experts=4,
            hidden_size=8,
            intermediate_size=6,
            tp_size=2,
            source_rank=0,
        )
