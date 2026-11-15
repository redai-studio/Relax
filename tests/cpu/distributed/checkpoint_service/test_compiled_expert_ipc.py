# Copyright (c) 2026 Relax Authors. All Rights Reserved.
import gc
import weakref
from collections.abc import Mapping

import pytest
import torch

from relax.backends.megatron.weight_update.shard_p2p_source import plan_expert_views
from relax.backends.sglang.shard_p2p_weight_loader import expert_shapes
from relax.distributed.checkpoint_service.backends.shard_p2p.ipc import CompiledIPC
from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint
from relax.distributed.checkpoint_service.backends.shard_p2p.rectangle_batch import rectangle_metadata


def _case(version=0, *, shared_sources=False):
    geometry = dict(num_experts=2, hidden_size=4, intermediate_size=4, tp_size=2, num_layers=1, source_world_size=2)
    prefix = "model.layers.0.mlp.experts."
    sources, writes = {}, []
    shared = torch.arange(99, dtype=torch.float32).to(torch.bfloat16) + version
    for rank in range(2):
        flat = shared if shared_sources else torch.arange(51, dtype=torch.float32).to(torch.bfloat16) + version
        offset = 3 + rank * 48 if shared_sources else 3
        canonical = {
            prefix + f"{rank}.gate_proj.weight": flat[offset : offset + 16].view(4, 4),
            prefix + f"{rank}.up_proj.weight": flat[offset + 16 : offset + 32].view(4, 4),
            prefix + f"{rank}.down_proj.weight": flat[offset + 32 : offset + 48].view(4, 4),
        }
        local, plans = plan_expert_views(
            canonical, num_experts=2, hidden_size=4, intermediate_size=4, tp_size=2, source_rank=rank
        )
        sources.update({(rank, key): tensor for key, tensor in local.items()})
        writes.extend(plans[0])
    flat = torch.empty(53, dtype=torch.bfloat16)
    targets = {prefix + "w13_weight": flat[5:37].view(2, 4, 4), prefix + "w2_weight": flat[37:].view(2, 4, 2)}
    return sources, targets, writes, geometry


@pytest.mark.parametrize("shared_sources", [False, True])
@pytest.mark.parametrize("rotation", [0, 5])
def test_compiled_metadata_matches_full_planner_after_fresh_allocations(shared_sources, rotation):
    sources, targets, writes, geometry = _case(shared_sources=shared_sources)
    cache = CompiledIPC(targets, writes, geometry, owner_rotation=rotation, target_shapes=expert_shapes(geometry))
    assert cache.plan_hash == fingerprint(dict(geometry=geometry, writes=writes))
    for version in range(6):
        sources, targets, writes, _ = _case(version, shared_sources=shared_sources)
        # Dictionary insertion order is irrelevant; source alias order and
        # the existing owner interleave must remain exactly the same.
        actual = cache._metadata(dict(reversed(list(sources.items()))), dict(reversed(list(targets.items()))))
        expected = rectangle_metadata(sources, targets, writes, owner_rotation=rotation)
        assert {key: value.tolist() for key, value in actual.items()} == expected
        for value in actual.values():
            value.zero_()  # Returned working descriptors cannot corrupt templates.

    def check_cpu_only(value):
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu" and value.dtype == torch.int64
            assert value.ndim == 2 and value.shape[1] == 7
            assert (value[:, :2] < 2).all()  # Slots, never old addresses.
        elif isinstance(value, Mapping):
            for key, item in value.items():
                check_cpu_only(key)
                check_cpu_only(item)
        elif isinstance(value, (tuple, list, frozenset)):
            for item in value:
                check_cpu_only(item)
        else:
            assert value is None or type(value) in (str, int, bool)

    check_cpu_only(vars(cache))


@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize("change", ["dtype", "shape", "stride", "capacity", "missing", "extra", "alias"])
def test_compiled_plan_rejects_changed_allocation_layout(side, change):
    sources, targets, writes, geometry = _case()
    cache = CompiledIPC(targets, writes, geometry, target_shapes=expert_shapes(geometry))
    cache._metadata(sources, targets)
    sources, targets, _, _ = _case(1)
    values = sources if side == "source" else targets
    first, second = list(values)
    tensor = values[first]
    if change == "dtype":
        values[first] = tensor.float()
    elif change == "shape":
        values[first] = tensor.view(-1)
    elif change == "stride":
        values[first] = tensor.as_strided((tensor.numel() // 2,), (2,))
    elif change == "capacity":
        tensor.untyped_storage().resize_(tensor.untyped_storage().nbytes() + 16)
    elif change == "missing":
        values.pop(first)
    elif change == "extra":
        values[(9, "extra") if side == "source" else "extra"] = tensor
    elif side == "source":
        values[first] = values[second]
    else:
        # Same capacity, offset and shape, but the shared target allocation
        # splits into two. Checking only tensor metadata would miss this.
        storage = torch.empty(53, dtype=torch.bfloat16)
        values[first] = storage[5:37].view(2, 4, 4)
    with pytest.raises(ValueError):
        cache._metadata(sources, targets)


@pytest.mark.parametrize("after_compile", [False, True])
def test_compiled_plan_rejects_physical_target_overlap(after_compile):
    sources, targets, writes, geometry = _case()
    cache = CompiledIPC(targets, writes, geometry, target_shapes=expert_shapes(geometry))
    if after_compile:
        cache._metadata(sources, targets)
    names = list(targets)
    targets[names[1]] = targets[names[0]].as_strided((2, 4, 2), (8, 2, 1), storage_offset=6)
    with pytest.raises(ValueError, match="physically overlap"):
        cache._metadata(sources, targets)


def test_compiled_first_apply_checks_source_bounds_before_launch(monkeypatch):
    from relax.distributed.checkpoint_service.backends.shard_p2p import ipc, rectangle_batch

    sources, targets, writes, geometry = _case()
    writes[-1]["src_offset"] = 999999
    cache = CompiledIPC(targets, writes, geometry, target_shapes=expert_shapes(geometry))
    calls = []
    monkeypatch.setattr(ipc, "import_tensor", lambda payload, device: payload)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("fence"))
    monkeypatch.setattr(rectangle_batch, "launch_rectangles", lambda *args: calls.append("launch"))
    with pytest.raises(ValueError, match="bounds"):
        cache.apply(targets, sources)
    assert calls == ["fence"] and cache.failed
    with pytest.raises(RuntimeError, match="failed"):
        cache.apply(targets, sources)


@pytest.mark.parametrize("failure", [None, "import", "launch", "fence"])
def test_compiled_apply_fences_and_drops_imports_before_return(monkeypatch, failure):
    from relax.distributed.checkpoint_service.backends.shard_p2p import ipc, rectangle_batch

    sources, targets, writes, geometry = _case()
    cache = CompiledIPC(targets, writes, geometry, target_shapes=expert_shapes(geometry))
    refs, calls = [], []

    def import_tensor(payload, device):
        if refs and failure == "import":
            raise ValueError("import failure")
        tensor = payload.clone()
        refs.append(weakref.ref(tensor))
        return tensor

    def launch(groups, device, live):
        assert all(ref() is not None for ref in refs)
        assert all(value.device.type == "cpu" for value in groups.values())
        calls.append("launch")
        live.append(torch.empty(3))
        if failure == "launch":
            raise ValueError("launch failure")

    def fence():
        assert all(ref() is not None for ref in refs)
        calls.append("fence")
        if failure == "fence":
            raise ValueError("fence failure")

    monkeypatch.setattr(ipc, "import_tensor", import_tensor)
    monkeypatch.setattr(rectangle_batch, "launch_rectangles", launch)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", fence)
    if failure:
        with pytest.raises(ValueError, match="failure"):
            cache.apply(targets, sources)
    else:
        cache.apply(targets, sources)
    gc.collect()
    if failure == "fence":
        assert all(ref() is not None for ref in refs)
        ipc._FAILED_READS.clear()  # Test-only CPU imports; no pending CUDA work.
        gc.collect()
    assert all(ref() is None for ref in refs)
    assert calls == (["fence"] if failure == "import" else ["launch", "fence"])
    assert cache.failed == bool(failure)
