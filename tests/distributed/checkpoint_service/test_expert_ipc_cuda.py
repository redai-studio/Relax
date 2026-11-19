# Copyright (c) 2026 Relax Authors. All Rights Reserved.


"""Fresh per-consumer expert IPC across eight shared GPUs and TMS remaps."""

import gc
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from relax.backends.sglang.shard_p2p_weight_loader import expert_shapes


def _reference(layer: int, expert: int, version: int, h: int, i: int) -> tuple:
    rows, cols = torch.arange(i)[:, None], torch.arange(h)[None, :]
    gate = (layer * 11 + expert * 3 + rows % 7 + cols % 5 + version * 2) % 61
    down = (
        layer * 13 + expert * 5 + torch.arange(h)[:, None] % 9 + torch.arange(i)[None, :] % 11 + version * 3
    ) % 127 - 63
    return gate.to(torch.bfloat16), (gate + 64).to(torch.bfloat16), down.to(torch.bfloat16)


def _raw_iterator(weights: dict, rank: int, geometry: dict):
    from megatron.bridge.models.conversion import param_mapping as pm
    from megatron.core import parallel_state as mpu

    from relax.backends.megatron.weight_update.bridge_converter import BridgeConverter

    patch = pytest.MonkeyPatch()
    for name, size in (
        ("expert_tensor_parallel", 1),
        ("expert_model_parallel", 8),
        ("tensor_model_parallel", 2),
        ("pipeline_model_parallel", 1),
        ("context_parallel", 1),
    ):
        patch.setattr(mpu, f"get_{name}_world_size", lambda size=size: size)
    patch.setattr(pm.MegatronParamMapping, "tp_size", property(lambda self: 1))
    converter = object.__new__(BridgeConverter)
    converter._quantization_config, converter._bridge_task_map = None, {}
    infos = []
    for name in weights:
        layer = int(name.split(".")[2])
        fc1 = "linear_fc1" in name
        cls = pm.FusedGatedExpertMapping if fc1 else pm.FusedExpertMapping
        mapping = cls(name, f"model.layers.{layer}.mlp.experts." + ("gate_up_proj" if fc1 else "down_proj"))
        mapping._mapping = mapping._get_or_create_mapping("column" if fc1 else "row")
        converter._bridge_task_map[name] = SimpleNamespace(mapping=mapping, megatron_module=torch.nn.Module())
        parameter = weights[name]
        infos.append(
            SimpleNamespace(
                name=name,
                src_rank=rank,
                shape=tuple(parameter.shape),
                dtype=parameter.dtype,
                size=parameter.numel() * parameter.element_size(),
                attrs={},
            )
        )
    iterator = SimpleNamespace(
        _direct_expert_layout=SimpleNamespace(
            ep_size=1,
            moe_dp_size=1,
            tp_size=2,
            up_proj_first=False,
            **{key: geometry[key] for key in ("num_experts", "hidden_size", "intermediate_size")},
        ),
        _expert_buckets=[infos],
        _vanilla_key_map={name: name for name in weights},
        _bridge_converter=converter,
    )
    return iterator, patch


def _worker(
    rank: int,
    devices: list[str],
    rendezvous: str,
    batched: bool,
    cached: bool,
    overlap: bool = False,
    raw_recipe: bool = False,
) -> None:
    # Pytest's module-level CUDA capability checks can initialize CUDA before
    # this function. Keep the inherited visible-device list and select a
    # logical ordinal explicitly; changing visibility here can silently leave
    # every worker on physical GPU 0.
    assert len(devices) == torch.cuda.device_count() == 8
    torch.cuda.set_device(rank % 8)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=16, timeout=timedelta(seconds=180))
    group = dist.group.WORLD
    placement = dict(
        rank=rank,
        pid=os.getpid(),
        device=torch.cuda.current_device(),
        uuid=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
    )
    placements = [None] * 16
    dist.all_gather_object(placements, placement, group=group)
    assert len({item["uuid"] for item in placements[:8]}) == 8
    assert all(placements[owner]["uuid"] == placements[owner + 8]["uuid"] for owner in range(8))
    audit_dir = os.environ.get("RELAX_GPU_TEST_AUDIT_DIR")
    if rank == 0 and audit_dir:
        path = Path(audit_dir) / f"devices-{batched}-{cached}-{overlap}-{raw_recipe}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(placements, indent=2))
    from torch_memory_saver import torch_memory_saver as tms

    from relax.backends.megatron.weight_update.shard_p2p_source import ExpertViewPlanCache, plan_expert_views
    from relax.distributed.checkpoint_service.backends.shard_p2p.ipc import CompiledIPC, apply_ipc_plan, export_tensor
    from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint

    e, h, i, layers = 32, 128, 64, 2
    geometry = dict(
        num_experts=e, hidden_size=h, intermediate_size=i, tp_size=2, num_layers=layers, source_world_size=8
    )
    source = rank < 8
    native_groups = (
        [dist.new_group(ranks=[first, first + 1], backend="nccl") for first in range(0, 8, 2)] if overlap else []
    )
    parameters = {}
    if not source:
        with tms.region(tag="weights", enable_cpu_backup=False):
            for layer in range(layers):
                prefix = f"model.layers.{layer}.mlp.experts."
                parameters[prefix + "w13_weight"] = torch.empty(e, i, h, dtype=torch.bfloat16, device="cuda")
                parameters[prefix + "w2_weight"] = torch.empty(e, h, i // 2, dtype=torch.bfloat16, device="cuda")
    receiver_plan = None
    source_cache = ExpertViewPlanCache() if cached and source and not raw_recipe else None
    source_plans = None
    source_recipe, raw_iterator, raw_patch = None, None, None
    raw_plans = None
    for version in range(6):
        # A session may rebind allocations but cannot change logical layout.
        # The uncached cases retain their existing layout-change coverage.
        layout_version = 0 if cached else version
        up_first = bool(layout_version % 2)
        if source:
            padding = (layout_version % 2) * 8
            fc1_size, fc2_size = 2 * i * h, h * i
            count = e // 8
            layer_size = count * (fc1_size + fc2_size + 2 * padding)
            storage = torch.empty(13 + layers * layer_size, dtype=torch.bfloat16, device="cuda")
            canonical = {}
            raw_weights = {}
            for layer in range(layers):
                for local_expert in range(count):
                    expert = rank * count + local_expert
                    index = count - 1 - local_expert if layout_version % 3 == 2 else local_expert
                    start = 13 + layer * layer_size
                    fc1 = storage.narrow(0, start + index * (fc1_size + padding), fc1_size).view(2 * i, h)
                    fc2 = storage.narrow(
                        0, start + count * (fc1_size + padding) + index * (fc2_size + padding), fc2_size
                    ).view(h, i)
                    gate, up, down = _reference(layer, expert, version, h, i)
                    fc1[:i].copy_(gate)
                    fc1[i:].copy_(up)
                    fc2.copy_(down)
                    prefix = f"model.layers.{layer}.mlp.experts.{expert}."
                    if raw_recipe:
                        for fc, tensor in ((1, fc1), (2, fc2)):
                            raw_weights[f"decoder.layers.{layer}.mlp.experts.linear_fc{fc}.weight{expert}"] = tensor
                    if not raw_recipe or version == 0:
                        canonical.update(
                            {
                                prefix + "gate_proj.weight": fc1[:i],
                                prefix + "up_proj.weight": fc1[i:],
                                prefix + "down_proj.weight": fc2,
                            }
                        )
            torch.cuda.synchronize()
            backup = storage.cpu()
            if raw_recipe and version:
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                sources = source_recipe.bind(raw_iterator, raw_weights, rank)
                assert torch.cuda.memory_allocated() == before
                assert torch.cuda.max_memory_allocated() == before
                plans = [list(plan) for plan in raw_plans]
            else:
                layout = dict(
                    num_experts=e,
                    hidden_size=h,
                    intermediate_size=i,
                    tp_size=2,
                    source_rank=rank,
                    up_proj_first=up_first,
                )
                if source_cache is None:
                    sources, plans = plan_expert_views(canonical, **layout)
                elif version == 0:
                    sources, source_plans = source_cache.bind(canonical, **layout)
                    plans = source_plans
                else:
                    sources = source_cache.bind_sources(canonical, **layout)
                    plans = source_plans
                if source_cache is not None:
                    json.dumps(vars(source_cache))
                if raw_recipe:
                    from relax.backends.megatron.weight_update.shard_p2p_source import RawExpertRecipe

                    raw_iterator, raw_patch = _raw_iterator(raw_weights, rank, geometry)
                    source_recipe = RawExpertRecipe.compile(raw_iterator, raw_weights, rank, sources)
                    raw_plans = [list(plan) for plan in plans]
            if raw_recipe:
                json.dumps(vars(source_recipe))  # Persistent recipe must contain only CPU metadata.
            torch.cuda.synchronize()
            needed = [{write["source"] for write in plan} for plan in plans]
            payload = dict(
                plans=plans,
                exports={
                    reader: {key: export_tensor(value) for key, value in sources.items() if key in needed[reader % 2]}
                    for reader in range(8, 16)
                },
            )
        else:
            for parameter in parameters.values():
                parameter.fill_(float("nan"))
            torch.cuda.synchronize()
            payload = None
        gathered = [None] * 16
        dist.all_gather_object(gathered, payload, group=group)
        if overlap and source:
            # Native TP export reads different weights after the IPC events
            # were recorded. Exercise both on the same physical devices;
            # the exported expert sources must remain unchanged.
            native_source = torch.full((2**20,), rank + version * 8, device="cuda", dtype=torch.bfloat16)
            native_target = torch.empty(2**21, device="cuda", dtype=torch.bfloat16)
            native_work = dist.all_gather_into_tensor(
                native_target, native_source, group=native_groups[rank // 2], async_op=True
            )
        if not source:
            tp = rank % 2
            writes = [write for owner in gathered[:8] for write in owner["plans"][tp]]
            bindings = {
                (owner, key): value for owner in range(8) for key, value in gathered[owner]["exports"][rank].items()
            }
            if cached:
                if receiver_plan is None:
                    receiver_plan = CompiledIPC(
                        parameters, writes, geometry, owner_rotation=rank - 8, target_shapes=expert_shapes(geometry)
                    )
                assert receiver_plan.plan_hash == fingerprint(dict(geometry=geometry, writes=writes))
                receiver_plan.apply(parameters, bindings)
            else:
                apply_ipc_plan(
                    parameters,
                    writes,
                    bindings,
                    geometry,
                    batched=batched,
                    owner_rotation=rank - 8,
                    target_shapes=expert_shapes(geometry),
                )
            for layer in range(layers):
                values = [_reference(layer, expert, version, h, i) for expert in range(e)]
                gate, up, down = (torch.stack([parts[j] for parts in values]) for j in range(3))
                start, end = tp * (i // 2), (tp + 1) * (i // 2)
                gate, up = gate[:, start:end], up[:, start:end]
                prefix = f"model.layers.{layer}.mlp.experts."
                torch.testing.assert_close(
                    parameters[prefix + "w13_weight"].cpu(),
                    torch.cat((up, gate) if up_first else (gate, up), dim=1),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    parameters[prefix + "w2_weight"].cpu(), down[:, :, start:end], rtol=0, atol=0
                )
        if overlap and source:
            native_work.wait()
            for tp in range(2):
                expected = torch.full((2**20,), rank // 2 * 2 + tp + version * 8, dtype=torch.bfloat16)
                torch.testing.assert_close(
                    native_target[tp * 2**20 : (tp + 1) * 2**20].cpu(), expected, rtol=0, atol=0
                )
            del native_work, native_source, native_target
        # Models all eight receiver close ACKs, before source storage release.
        dist.barrier(group=group)
        if source:
            torch.testing.assert_close(storage.cpu(), backup, rtol=0, atol=0, equal_nan=True)
            sources.clear()
            canonical.clear()
            raw_weights.clear()
            if raw_recipe:
                del tensor
            del fc1, fc2
            gc.collect()
            torch.cuda.ipc_collect()
            storage.untyped_storage().resize_(0)
            assert storage.untyped_storage().nbytes() == 0
        else:
            tms.pause("weights")
        dist.barrier(group=group)
        if not source:
            tms.resume("weights")
        torch.cuda.synchronize()
        dist.barrier(group=group)
    dist.destroy_process_group()
    if raw_patch is not None:
        raw_patch.undo()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Requires eight peer-accessible GPUs and TMS")
@pytest.mark.parametrize(
    "batched,cached", [(False, False), (True, False), (True, True)], ids=["torch", "batched", "cached"]
)
def test_eight_owners_to_four_tp2_replicas_with_fresh_ipc_and_tms(tmp_path, batched, cached):
    import torch_memory_saver

    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")[:8]
    with torch_memory_saver.configure_subprocess():
        mp.spawn(
            _worker,
            args=(devices, (tmp_path / "rendezvous").as_uri(), batched, cached),
            nprocs=16,
            join=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA and Triton")
def test_batched_copy_preserves_all_bits_and_strided_tail(monkeypatch):
    from relax.distributed.checkpoint_service.backends.shard_p2p.rectangle_batch import (
        launch_rectangles,
        rectangle_metadata,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    bits = torch.arange(65536 + 29 * 23, device=device, dtype=torch.int32).to(torch.uint16)
    source = bits.view(torch.bfloat16)
    dense_bits = torch.zeros(65536, device=device, dtype=torch.uint16)
    tail_bits = torch.full((29 * 19 + 3,), 0xBEEF, device=device, dtype=torch.uint16)
    targets = {"dense": dense_bits.view(torch.bfloat16), "tail": tail_bits.view(torch.bfloat16)}
    writes = [
        dict(
            source_rank=0,
            source="s",
            param="dense",
            rows=1,
            row_bytes=65536 * 2,
            src_offset=0,
            dst_offset=0,
            src_stride=65536 * 2,
            dst_stride=65536 * 2,
        ),
        dict(
            source_rank=0,
            source="s",
            param="tail",
            rows=29,
            row_bytes=17 * 2,
            src_offset=(65536 + 2) * 2,
            dst_offset=3 * 2,
            src_stride=23 * 2,
            dst_stride=19 * 2,
        ),
    ]
    expected_tail = tail_bits.cpu()
    source_cpu = bits.cpu()
    for row in range(29):
        expected_tail[3 + row * 19 : 3 + row * 19 + 17] = source_cpu[65536 + 2 + row * 23 : 65536 + 2 + row * 23 + 17]
    metadata = []
    try:
        launch_rectangles(rectangle_metadata({(0, "s"): source}, targets, writes), device, metadata)
        torch.cuda.synchronize()
        assert (
            dense_bits.cpu().view(torch.uint8).numpy().tobytes()
            == source_cpu[:65536].view(torch.uint8).numpy().tobytes()
        )
        assert tail_bits.cpu().view(torch.uint8).numpy().tobytes() == expected_tail.view(torch.uint8).numpy().tobytes()
        groups = rectangle_metadata({(0, "s"): source}, targets, writes)
        checks = []
        launch_rectangles(groups, device, metadata, native_checks=checks)
        torch.cuda.synchronize()
        assert checks and not any(check.any().item() for check in checks)
        # Native parity is bitwise, including signed zeros and every NaN bit
        # pattern. A changed final strided element must fail before overwrite.
        tail_bits[3 + 28 * 19 + 16] = 0
        checks = []
        launch_rectangles(groups, device, metadata, native_checks=checks)
        torch.cuda.synchronize()
        assert any(check.any().item() for check in checks)
        from relax.distributed.checkpoint_service.backends.shard_p2p import rectangle_kernel

        monkeypatch.setattr(rectangle_kernel, "launch", lambda *args: None)
        checks = []
        launch_rectangles(groups, device, metadata, native_checks=checks)
        torch.cuda.synchronize()
        assert checks and all(check.all().item() for check in checks)
    finally:
        torch.cuda.synchronize()
        metadata.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA native storage")
def test_pinned_native_and_normal_copy_preserve_bits_and_span_boundaries():
    from relax.distributed.checkpoint_service.backends.shard_p2p.remote.transport import PersistentPinnedTargets

    # Cross three 64 MiB chunks with an unaligned chunk tail and nonzero
    # destination origin. uint16 patterns include all NaNs and signed zeros.
    elements = 64 * 1024**2 + 8195
    target = torch.arange(elements + 7, device="cuda", dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    spans = {"weight": (6, (elements + 3) * 2)}
    agent = SimpleNamespace(get_reg_descs=lambda values: values, register_memory=lambda descriptors: None)
    pinned = PersistentPinnedTargets(agent, {"weight": target}, spans=spans)
    begin, end = spans["weight"]
    pinned.buffer.copy_(target.view(torch.uint8)[begin:end])
    pinned.copy_to({"weight": target}, verify_native=True)
    expected = torch.arange(elements + 7, device="cuda", dtype=torch.int32).to(torch.uint16)
    assert torch.equal(target.view(torch.uint8), expected.view(torch.uint8))
    # Normal copies use a separate path and must preserve both span boundaries.
    pinned.buffer.fill_(0x5A)
    pinned.copy_to({"weight": target})
    assert bool((target.view(torch.uint8)[begin:end] == 0x5A).all())
    assert torch.equal(target.view(torch.uint8)[:begin], expected.view(torch.uint8)[:begin])
    assert torch.equal(target.view(torch.uint8)[end:], expected.view(torch.uint8)[end:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA native storage")
@pytest.mark.parametrize("storage_wrapper", ["view", "dlpack"])
@pytest.mark.parametrize("overlap", [False, True])
def test_native_target_physical_aliases_on_cuda(storage_wrapper, overlap):
    from relax.backends.sglang.shard_p2p_weight_loader import validate_expert_targets

    geometry = dict(num_experts=1, hidden_size=4, intermediate_size=4, tp_size=2, num_layers=1, source_world_size=1)
    flat = torch.full((7 + 16 + 8,), -3, dtype=torch.bfloat16, device="cuda")
    first = flat.narrow(0, 7, 16).view(1, 4, 4)
    second = flat.narrow(0, 10 if overlap else 23, 8).view(1, 4, 2)
    if storage_wrapper == "dlpack":
        second = torch.from_dlpack(second)
        assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    else:
        assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    first_end = first.data_ptr() + first.numel() * first.element_size()
    if overlap:
        assert first.data_ptr() < second.data_ptr() < first_end
        assert second.data_ptr() + second.numel() * second.element_size() <= first_end
    else:
        assert first_end == second.data_ptr()
    prefix = "model.layers.0.mlp.experts."
    parameters = {prefix + "w13_weight": first, prefix + "w2_weight": second}
    writes = [
        dict(
            id=name,
            source_rank=0,
            source=name,
            param=name,
            rows=1,
            row_bytes=value.numel() * 2,
            src_offset=0,
            dst_offset=0,
            src_stride=value.numel() * 2,
            dst_stride=value.numel() * 2,
        )
        for name, value in parameters.items()
    ]
    before = flat.view(torch.uint8).cpu().clone()
    if overlap:
        with pytest.raises(ValueError, match="physically overlap"):
            validate_expert_targets(writes, parameters, geometry)
    else:
        validate_expert_targets(writes, parameters, geometry)
    torch.cuda.synchronize()
    torch.testing.assert_close(flat.view(torch.uint8).cpu(), before, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Requires eight peer-accessible GPUs, NCCL and TMS")
def test_raw_expert_recipe_with_fresh_storage_tms_and_concurrent_native_gathers(tmp_path):
    import torch_memory_saver

    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")[:8]
    with torch_memory_saver.configure_subprocess():
        mp.spawn(
            _worker,
            args=(devices, (tmp_path / "raw-recipe-rendezvous").as_uri(), True, True, True, True),
            nprocs=16,
            join=True,
        )
