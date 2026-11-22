# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Describe direct TP copies from SGLang linear-module metadata.

Only known BF16 copy loaders qualify. Other parameters retain the native
loader, including packed GDN, aliases and model-specific transforms.
"""

from collections import defaultdict
from math import prod

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from relax.distributed.checkpoint_service.backends.shard_p2p.layout import TensorRegion
from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint, region_write, validate_coverage

from .tp_shard_adapters import needs_native_refresh, target_parameter


def _matrix(shape: tuple[int, ...]) -> tuple[int, int]:
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("Empty TP parameter")
    return shape[0], prod(shape[1:])


def _linear_module(module, param: torch.nn.Parameter, tp_rank: int) -> str | None:
    cls = type(module)
    if (
        cls.__module__ != "sglang.srt.layers.linear"
        or cls.__name__ not in ("ColumnParallelLinear", "RowParallelLinear", "MergedColumnParallelLinear")
        or type(param) is not torch.nn.Parameter
        or param.dtype != torch.bfloat16
        or not param.is_contiguous()
        or getattr(module, "tp_rank", None) != tp_rank
        or not isinstance(getattr(module, "tp_size", None), int)
        or module.tp_size <= 1
        or getattr(module, "use_presharded_weights", False)
        or type(getattr(module, "quant_method", None)).__name__ != "UnquantizedLinearMethod"
    ):
        return None
    loader = getattr(param, "weight_loader", None)
    if getattr(loader, "__self__", None) is not module or getattr(loader, "__func__", None) is not cls.weight_loader:
        return None
    if any(
        getattr(param, flag, False)
        for flag in ("is_gguf_weight", "is_metadata", "use_bitsandbytes_4bit", "needs_scalar_to_array")
    ):
        return None
    return cls.__name__


def _target_region(
    name: str, param: torch.nn.Parameter, module, kind: str, shard: str | int | None, hf: str, spec: dict
) -> dict | None:
    shape = tuple(param.shape)
    source_shape = tuple(spec["shape"])
    if spec["dtype"] != str(torch.bfloat16) or len(source_shape) != len(shape):
        return None
    target_rows, width = _matrix(shape)
    source_rows, source_width = _matrix(source_shape)
    if kind != "RowParallelLinear" and width != source_width:
        return None
    tp_rank, tp_size = module.tp_rank, module.tp_size
    start, rows, source_start = 0, target_rows, 0
    if kind == "RowParallelLinear":
        if shard is not None or len(shape) != 2 or getattr(param, "input_dim", None) != 1:
            return None
        if source_rows != target_rows or source_width != width * tp_size:
            return None
        source_start = tp_rank * width
        global_shape = [source_rows, source_width]
        offset, region_shape = [0, source_start], [target_rows, width]
    else:
        if getattr(param, "output_dim", None) != 0:
            return None
        if kind == "ColumnParallelLinear":
            if shard is not None or source_rows != target_rows * tp_size:
                return None
            source_start = tp_rank * target_rows
        elif kind == "MergedColumnParallelLinear":
            sizes = getattr(module, "output_sizes", ())
            partitions = getattr(module, "output_partition_sizes", ())
            if type(shard) is not int or shard < 0 or shard >= len(sizes):
                return None
            if (
                len(sizes) != len(partitions)
                or source_rows != sizes[shard]
                or sizes[shard] != partitions[shard] * tp_size
            ):
                return None
            rows = partitions[shard]
            start = sum(partitions[:shard])
            source_start = tp_rank * rows
        else:
            return None
        if rows <= 0 or start + rows > target_rows:
            return None
        global_shape = [source_rows, width]
        offset, region_shape = [source_start, 0], [rows, width]
    return dict(
        hf=hf,
        param=name,
        global_shape=global_shape,
        offset=offset,
        shape=region_shape,
        storage_offset=start * width,
        stride=width,
        dtype=spec["dtype"],
    )


def _writes(records, targets, tp_rank: int) -> list[dict]:
    sources = defaultdict(list)
    for record in records:
        for source in record["regions"]:
            sources[source["hf"]].append((record["rank"], source))
    result = []
    for target in targets:
        for rank, source in sources[target["hf"]]:
            if source["global_shape"] != target["global_shape"] or source["dtype"] != target["dtype"]:
                raise ValueError("Canonical TP source and native target disagree")
            write = region_write(
                TensorRegion.from_descriptor(source, itemsize=2),
                TensorRegion.from_descriptor(target, itemsize=2),
                id=f"ordinary:{tp_rank}:{len(result)}",
                source=source["source"],
                source_rank=rank,
                param=target["param"],
                target=f"tp-{tp_rank}",
                hf=target["hf"],
            )
            if write is not None:
                result.append(write)
    return result


@torch.no_grad()
def describe_tp_targets(model: torch.nn.Module, description: dict, tp_rank: int) -> dict:
    """Select complete copy components; unresolved layouts remain native."""
    records, specs, world = description["records"], description["native_specs"], description["source_world_size"]
    candidate_hf = {region["hf"] for record in records for region in record["regions"]}
    params = dict(model.named_parameters(remove_duplicate=False))
    modules = dict(model.named_modules())
    shared = defaultdict(set)
    for name, value in model.named_parameters(remove_duplicate=False):
        shared[StorageWeakRef(value.untyped_storage())].add(("param", name))
    for name, value in model.named_buffers(remove_duplicate=False):
        shared[StorageWeakRef(value.untyped_storage())].add(("buffer", name))
    aliased = {name for owners in shared.values() if len(owners) > 1 for kind, name in owners if kind == "param"}
    targets = defaultdict(list)
    for hf in sorted(candidate_hf & specs.keys()):
        mapping = target_parameter(model, hf)
        if mapping is None:
            continue
        name, shard = mapping
        param = params.get(name)
        module = modules.get(name.rpartition(".")[0])
        if param is None or module is None or name in aliased:
            continue
        kind = _linear_module(module, param, tp_rank)
        if kind is None:
            continue
        region = _target_region(name, param, module, kind, shard, hf, specs[hf])
        if region is not None:
            targets[name].append(region)

    graph = defaultdict(set)

    def edge(left, right):
        graph[left].add(right)
        graph[right].add(left)

    for name, regions in targets.items():
        for region in regions:
            edge(("hf", region["hf"]), ("target", name))
    for record in records:
        for names in record["groups"].values():
            for name in names[1:]:
                edge(("hf", names[0]), ("hf", name))
    visited, selected, writes, layouts, components, target_hf = set(), set(), [], {}, [], {}
    for node in sorted(graph):
        if node in visited:
            continue
        pending, component = [node], set()
        while pending:
            item = pending.pop()
            if item in component:
                continue
            component.add(item)
            pending.extend(graph[item] - component)
        visited.update(component)
        names = {name for kind, name in component if kind == "hf"}
        native = {name for kind, name in component if kind == "target"}
        if not names or not native or not names <= candidate_hf:
            continue
        copies = [region for name in native for region in targets[name]]
        shapes = {name: list(params[name].shape) for name in native}
        try:
            plan = _writes(records, copies, tp_rank)
            coverage = {name: dict(nbytes=prod(shape) * 2) for name, shape in shapes.items()}
            if validate_coverage(plan, coverage, source_world_size=world) != native:
                continue
        except (ValueError, TypeError):
            continue
        selected.update(names)
        writes.extend(plan)
        layouts.update(shapes)
        components.append(sorted(names))
        target_hf.update({name: sorted({region["hf"] for region in targets[name]}) for name in native})
    for index, write in enumerate(writes):
        write["id"] = f"ordinary:{tp_rank}:{index}"
    return dict(
        description_hash=fingerprint(description),
        tp_rank=tp_rank,
        selected_hf=sorted(selected),
        components=components,
        target_shapes=layouts,
        target_hf=target_hf,
        writes=writes,
        native_refresh=needs_native_refresh(model),
    )
