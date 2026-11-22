# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Describe direct TP copies from SGLang linear-module metadata.

Only known BF16 copy loaders qualify. Other parameters retain the native
loader, including quantized/aliased parameters and numerical transforms.
"""

from collections import defaultdict
from math import prod

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from relax.distributed.checkpoint_service.backends.shard_p2p.layout import TensorRegion
from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint, region_write, validate_coverage

from .tp_shard_adapters import full_copy_module, needs_native_refresh, packed_copy_loader, target_parameter


def _matrix(shape: tuple[int, ...]) -> tuple[int, int]:
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("Empty TP parameter")
    return shape[0], prod(shape[1:])


def _linear_module(module, param: torch.nn.Parameter, tp_rank: int) -> str | None:
    if (
        type(param) is not torch.nn.Parameter
        or param.dtype != torch.bfloat16
        or not param.is_contiguous()
        or any(
            getattr(param, flag, False)
            for flag in ("is_gguf_weight", "is_metadata", "use_bitsandbytes_4bit", "needs_scalar_to_array")
        )
    ):
        return None
    if full_copy_module(module, param):
        return "ReplicatedCopy"
    cls = type(module)
    embedding = cls.__module__ == "sglang.srt.layers.vocab_parallel_embedding" and cls.__name__ in (
        "VocabParallelEmbedding",
        "ParallelLMHead",
    )
    if (
        not (
            embedding
            or cls.__module__ == "sglang.srt.layers.linear"
            and cls.__name__
            in ("ColumnParallelLinear", "RowParallelLinear", "MergedColumnParallelLinear", "QKVParallelLinear")
        )
        or not embedding
        and getattr(module, "tp_rank", None) != tp_rank
        or type(getattr(module, "tp_size", None)) is not int
        or module.tp_size < 1
        or getattr(module, "use_presharded_weights", False)
        or type(getattr(module, "quant_method", None)).__name__
        != ("UnquantizedEmbeddingMethod" if embedding else "UnquantizedLinearMethod")
    ):
        return None
    loader = getattr(param, "weight_loader", None)
    if getattr(loader, "__self__", None) is not module:
        loader = packed_copy_loader(module, param)
    if getattr(loader, "__self__", None) is not module or getattr(loader, "__func__", None) is not cls.weight_loader:
        return None
    if embedding and (getattr(param, "packed_dim", None) is not None or getattr(param, "output_dim", None) != 0):
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
    tp_rank, tp_size = getattr(module, "tp_rank", 0), getattr(module, "tp_size", 1)
    start, rows, source_start = 0, target_rows, 0
    if kind == "ReplicatedCopy":
        if shard is not None or len(shape) != 2 or shape != source_shape:
            return None
        global_shape = list(shape)
        offset, region_shape = [0, 0], list(shape)
    elif kind in ("VocabParallelEmbedding", "ParallelLMHead"):
        indices = module.shard_indices
        begin, end = indices.org_vocab_start_index, indices.org_vocab_end_index
        if (
            shard is not None
            or len(shape) != 2
            or source_rows != module.org_vocab_size
            or module.num_embeddings != module.org_vocab_size
            or end - begin != target_rows
            or source_rows != target_rows * tp_size
            or begin < 0
            or end > source_rows
        ):
            return None
        global_shape = [source_rows, source_width]
        offset, region_shape = [begin, 0], [target_rows, width]
    elif kind == "RowParallelLinear":
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
        if kind == "QKVParallelLinear":
            if shard not in ("q", "k", "v") or len(shape) != 2:
                return None
            sizes = tuple(
                getattr(module, key, None)
                for key in (
                    "q_proj_shard_size",
                    "kv_proj_shard_size",
                    "v_proj_shard_size",
                )
            )
            replicas = getattr(module, "num_kv_head_replicas", None)
            metadata = tuple(
                getattr(module, key, None)
                for key in (
                    "total_num_heads",
                    "total_num_kv_heads",
                    "head_size",
                    "v_head_size",
                )
            )
            if (
                any(type(size) is not int or size <= 0 for size in (*sizes, *metadata))
                or type(replicas) is not int
                or replicas <= 0
                or tp_size % replicas
                or target_rows != sum(sizes)
                or metadata[0] * metadata[2] != sizes[0] * tp_size
                or metadata[1] * metadata[2] != sizes[1] * (tp_size // replicas)
                or metadata[1] * metadata[3] != sizes[2] * (tp_size // replicas)
            ):
                return None
            index = ("q", "k", "v").index(shard)
            rows, start = sizes[index], sum(sizes[:index])
            repeat = 1 if shard == "q" else replicas
            if source_rows != rows * (tp_size // repeat):
                return None
            source_start = (tp_rank // repeat) * rows
        elif kind == "ColumnParallelLinear":
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


def _target_regions(name: str, param: torch.nn.Parameter, module, kind: str, shard, hf: str, spec: dict) -> list[dict]:
    if type(shard) is not tuple:
        region = _target_region(name, param, module, kind, shard, hf, spec)
        return [region] if region is not None else []
    sizes = getattr(module, "output_sizes", ())
    if (
        kind != "MergedColumnParallelLinear"
        or not shard
        or len(set(shard)) != len(shard)
        or any(type(index) is not int or not 0 <= index < len(sizes) for index in shard)
        or sum(sizes[index] for index in shard) != spec["shape"][0]
    ):
        return []
    result, start = [], 0
    for index in shard:
        part = dict(spec, shape=[sizes[index], *spec["shape"][1:]])
        region = _target_region(name, param, module, kind, index, hf, part)
        if region is None:
            return []
        region["global_shape"][0] = spec["shape"][0]
        region["offset"][0] += start
        result.append(region)
        start += sizes[index]
    return result


def _writes(sources, targets, tp_rank: int) -> list[dict]:
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
    sources = defaultdict(list)
    for record in records:
        for source in record["regions"]:
            sources[source["hf"]].append((record["rank"], source))
    candidate_hf = set(sources)
    params = dict(model.named_parameters(remove_duplicate=False))
    modules = dict(model.named_modules(remove_duplicate=False))
    mappings = {hf: target_parameter(model, hf) for hf in sorted(candidate_hf & specs.keys())}
    shared = defaultdict(set)
    for name, value in params.items():
        shared[StorageWeakRef(value.untyped_storage())].add(("param", name))
    for name, value in model.named_buffers(remove_duplicate=False):
        shared[StorageWeakRef(value.untyped_storage())].add(("buffer", name))
    aliased = {name for owners in shared.values() if len(owners) > 1 for kind, name in owners if kind == "param"}
    # A tied head can expose the same full-copy module twice. Require one
    # HF producer; separate views, modules and buffer aliases stay native.
    native_mappings = None
    for owners in shared.values():
        if len(owners) < 2 or any(kind != "param" for kind, _ in owners):
            continue
        names = {name for _, name in owners}
        aliases = [modules.get(name.rpartition(".")[0]) for name in names]
        if any(module is not aliases[0] for module in aliases) or aliases[0] is None:
            continue
        parameter = next(iter(names))
        if not full_copy_module(aliases[0], params[parameter]) or any(
            params[name] is not params[parameter] for name in names
        ):
            continue
        if native_mappings is None:
            native_mappings = [target_parameter(model, hf) for hf in specs]
        selected = [mapping for mapping in native_mappings if mapping is not None and mapping[0] in names]
        if len(selected) == 1 and selected[0][1] is None:
            aliased.discard(selected[0][0])
    targets = defaultdict(list)
    for hf, mapping in mappings.items():
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
        targets[name].extend(_target_regions(name, param, module, kind, shard, hf, specs[hf]))

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
        copies = [region for name in sorted(native) for region in targets[name]]
        if names != {region["hf"] for region in copies}:
            continue
        shapes = {name: list(params[name].shape) for name in native}
        try:
            plan = _writes(sources, copies, tp_rank)
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
