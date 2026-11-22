# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bridge-qualified ordinary TP views for Dense and MoE models.

Only layout-preserving local views qualify. Model-specific numerical or packed
conversions keep the native exporter; the receiver qualifies its own loader.
"""

from collections.abc import Mapping
from math import prod

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu

from relax.distributed.checkpoint_service.backends.shard_p2p.ipc import group_storage_views
from relax.distributed.checkpoint_service.backends.shard_p2p.plan import fingerprint

from .bridge_converter import _normalize_global_name


def observe_native_specs(specs: dict, values) -> None:
    """Keep only scalar ordinary HF shapes from a completed native export."""
    for name, tensor in values:
        if ".experts." in name or name.startswith("relax.expert_slice.v1:"):
            del tensor
            continue
        spec = dict(shape=list(tensor.shape), dtype=str(tensor.dtype))
        del tensor
        if name in specs and specs[name] != spec:
            raise ValueError("Native HF schema changed during TP seed")
        specs[name] = spec


def _gdn_regions(name: str, parameter: torch.Tensor, task, tp_rank: int, tp_size: int) -> list[dict] | None:
    if type(task.mapping).__name__ != "GDNLinearMappingSeparate":
        return None
    from megatron.bridge.models.conversion import param_mapping as bridge

    mapping = task.mapping
    if type(mapping) is not getattr(bridge, "GDNLinearMappingSeparate", None):
        return None
    config = mapping._get_config(task.megatron_module)
    dimensions = tuple(
        getattr(config, field, None)
        for field in (
            "hidden_size",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
        )
    )
    names = mapping.hf_param
    if (
        type(parameter) not in (torch.Tensor, torch.nn.Parameter)
        or bridge._module_uses_fsdp(task.megatron_module)
        or any(type(size) is not int or size <= 0 for size in dimensions)
        or type(tp_size) is not int
        or tp_size < 1
        or type(tp_rank) is not int
        or not 0 <= tp_rank < tp_size
        or parameter.ndim != 2
        or getattr(parameter, "partition_dim", None) != 0
        or dimensions[3] % tp_size
        or dimensions[4] % tp_size
        or not isinstance(names, dict)
        or set(names) != {"qkv", "z", "b", "a"}
        or any(not isinstance(value, str) or not value for value in names.values())
        or len(set(names.values())) != 4
    ):
        raise NotImplementedError("GDN requires complete unquantized head-partitioned rows")
    hidden, key_dim, value_dim, key_heads, value_heads = dimensions
    q, v, n = key_dim * key_heads, value_dim * value_heads, value_heads
    sizes = (q, q, v, v, n, n)
    if tuple(parameter.shape) != (sum(sizes) // tp_size, hidden):
        raise NotImplementedError("GDN input storage differs from the native component layout")
    result, start = [], parameter.storage_offset()
    for key, rows, full_rows, global_start in zip(
        ("qkv", "qkv", "qkv", "z", "b", "a"),
        sizes,
        (2 * q + v, 2 * q + v, 2 * q + v, v, n, n),
        (0, q, 2 * q, 0, 0, 0),
        strict=True,
    ):
        local_rows = rows // tp_size
        result.append(
            dict(
                hf=names[key],
                source=name,
                global_shape=[full_rows, hidden],
                offset=[global_start + tp_rank * local_rows, 0],
                shape=[local_rows, hidden],
                storage_offset=start,
                stride=hidden,
                dtype=str(parameter.dtype),
                itemsize=parameter.element_size(),
            )
        )
        start += local_rows * hidden
    return result


def _qkv_regions(name: str, parameter: torch.Tensor, task, tp_rank: int, tp_size: int) -> list[dict] | None:
    if type(task.mapping).__name__ != "QKVMapping":
        return None
    from megatron.bridge.models.conversion import param_mapping as bridge

    mapping = task.mapping
    if type(mapping) is not getattr(bridge, "QKVMapping", None):
        return None
    config = mapping._get_config(task.megatron_module)
    dimensions = tuple(
        getattr(config, field, None)
        for field in (
            "hidden_size",
            "num_attention_heads",
            "num_query_groups",
            "kv_channels",
        )
    )
    names = mapping.hf_param
    gate = getattr(config, "attention_output_gate", False)
    if (
        type(parameter) not in (torch.Tensor, torch.nn.Parameter)
        or bridge._module_uses_fsdp(task.megatron_module)
        or any(type(size) is not int or size <= 0 for size in dimensions)
        or type(gate) is not bool
        or type(tp_size) is not int
        or tp_size < 1
        or type(tp_rank) is not int
        or not 0 <= tp_rank < tp_size
        or parameter.ndim != 2
        or getattr(parameter, "partition_dim", None) != 0
        or getattr(parameter, "partition_stride", 1) != 1
        or not isinstance(names, dict)
        or set(names) != {"q", "k", "v"}
        or any(not isinstance(value, str) or not value for value in names.values())
        or len(set(names.values())) != 3
    ):
        raise NotImplementedError("QKV requires complete plain head-partitioned storage")
    hidden, heads, groups, dim = dimensions
    full_rows = ((1 + gate) * heads + 2 * groups) * dim
    if heads % groups or parameter.shape[1] != hidden or parameter.shape[0] * tp_size != full_rows:
        raise NotImplementedError("QKV storage differs from native continuous TP chunks")
    per_group = heads // groups
    group_rows = ((1 + gate) * per_group + 2) * dim
    lower, upper = tp_rank * parameter.shape[0], (tp_rank + 1) * parameter.shape[0]
    result = []

    def add(key: str, packed: int, canonical: int) -> None:
        begin, end = max(lower, packed), min(upper, packed + dim)
        if begin < end:
            result.append(
                dict(
                    hf=names[key],
                    source=name,
                    group=sorted(names.values()),
                    global_shape=[((1 + gate) * heads if key == "q" else groups) * dim, hidden],
                    offset=[canonical + begin - packed, 0],
                    shape=[end - begin, hidden],
                    storage_offset=parameter.storage_offset() + (begin - lower) * hidden,
                    stride=hidden,
                    dtype=str(parameter.dtype),
                    itemsize=parameter.element_size(),
                )
            )

    for group in range(groups):
        start = group * group_rows
        for head in range(per_group):
            canonical = (group * per_group + head) * (1 + gate) * dim
            add("q", start + head * dim, canonical)
            if gate:
                add("q", start + (per_group + head) * dim, canonical + dim)
        add("k", start + (1 + gate) * per_group * dim, group * dim)
        add("v", start + ((1 + gate) * per_group + 1) * dim, group * dim)
    return result


def local_tp_regions(name: str, parameter: torch.Tensor, task, tp_rank: int, tp_size: int) -> list[dict]:
    """Express actual Bridge local views in canonical matrix coordinates."""
    mapping = task.mapping
    if (
        parameter.dtype != torch.bfloat16
        or parameter.ndim < 1
        or not parameter.is_contiguous()
        or getattr(mapping, "is_expert", False)
        or getattr(mapping, "is_adapter", False)
        or getattr(mapping, "tp_size", None) != tp_size
        or getattr(mapping, "tp_rank", None) != tp_rank
        or not hasattr(mapping, "local_hf_params")
    ):
        raise NotImplementedError("ordinary TP requires BF16 layout-preserving Bridge views")
    gdn = _gdn_regions(name, parameter, task, tp_rank, tp_size)
    if gdn is not None:
        return gdn
    qkv = _qkv_regions(name, parameter, task, tp_rank, tp_size)
    if qkv is not None:
        return qkv
    views = mapping.local_hf_params(
        parameter, global_param_name=task.global_param_name, megatron_module=task.megatron_module
    )
    if not views:
        raise NotImplementedError("Bridge has no local ordinary TP view contract")
    result = []
    for view in views:
        tensor = view.weight
        axis = view.shard_dim
        shape, full = tuple(tensor.shape), tuple(view.global_weight_shape)
        expected = list(shape)
        replicated = tp_size == 1 and view.shard_group == "replicated" and axis is None
        if replicated:
            axis = 0
        if type(axis) is not int or not -len(shape) <= axis < len(shape):
            raise NotImplementedError("Bridge view is replicated or lacks a shard axis")
        axis %= len(shape)
        expected[axis] *= tp_size
        if (
            (view.shard_group != "tp" and not replicated)
            or tuple(expected) != full
            or tensor.dtype != parameter.dtype
            or tensor.device != parameter.device
            or not tensor.is_contiguous()
            or tensor.untyped_storage().data_ptr() != parameter.untyped_storage().data_ptr()
            or tensor.storage_offset() < parameter.storage_offset()
            or tensor.storage_offset() + tensor.numel() > parameter.storage_offset() + parameter.numel()
        ):
            raise NotImplementedError("Bridge local TP view does not preserve complete source storage")
        if axis not in (0, len(shape) - 1) or axis != 0 and len(shape) != 2:
            raise NotImplementedError("TP shard cannot be represented as a matrix rectangle")
        rows, columns = shape[0], prod(shape[1:])
        result.append(
            dict(
                hf=view.name,
                source=name,
                global_shape=[full[0], prod(full[1:])],
                offset=[tp_rank * rows, 0] if axis == 0 else [0, tp_rank * columns],
                shape=[rows, columns],
                storage_offset=tensor.storage_offset(),
                stride=columns,
                dtype=str(tensor.dtype),
                itemsize=tensor.element_size(),
            )
        )
    return result


class OrdinaryTPSource:
    """Cache scalar view descriptions, never a tensor or an IPC handle."""

    def __init__(self, iterator, *, process_group) -> None:
        self.iterator, self.group = iterator, process_group
        self.rank = dist.get_rank(process_group)
        self.tp_rank = mpu.get_tensor_model_parallel_rank()
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        replica_group = mpu.get_data_parallel_group(with_context_parallel=True)
        self.replica_ranks = tuple(dist.get_process_group_ranks(replica_group))
        self.replica_rank = dist.get_rank(replica_group)
        if self.replica_ranks[self.replica_rank] != dist.get_global_rank(process_group, self.rank):
            raise ValueError("Ordinary TP replica group disagrees with the Actor rank")
        self.native_specs = {}
        self.selected = frozenset()
        self._schema = None
        self._recipes = {}

    def observe_native(self, values) -> None:
        observe_native_specs(self.native_specs, values)

    def _views(self, weights: Mapping[str, torch.Tensor], *, selected_only: bool) -> tuple[dict, list, dict]:
        converter = self.iterator._bridge_converter
        tensors, regions, groups = {}, [], {}
        for bucket in self.iterator._non_expert_buckets:
            for info in bucket:
                # PP-global buckets include parameters absent from this stage.
                if info.name not in self.iterator._vanilla_key_map:
                    continue
                if selected_only and info.name not in self.selected:
                    continue
                if self.tp_size > 1 and (
                    not info.attrs.get("tensor_model_parallel") or info.attrs.get("parallel_mode") == "duplicated"
                ):
                    continue
                value = weights[self.iterator._vanilla_key_map[info.name]].detach()
                if tuple(value.shape) != tuple(info.shape) or value.dtype != info.dtype or not value.is_cuda:
                    raise ValueError("Ordinary TP source no longer matches its live inventory")
                for key, attr in info.attrs.items():
                    setattr(value, key, attr)
                _, task = converter._lookup_task(_normalize_global_name(info.name))
                if task is None:
                    continue
                try:
                    items = local_tp_regions(info.name, value, task, self.tp_rank, self.tp_size)
                except (NotImplementedError, ValueError) as error:
                    if selected_only:
                        raise ValueError("A qualified TP source mapping changed") from error
                    continue
                if selected_only or all(
                    item["hf"] in self.native_specs
                    and item["global_shape"]
                    == [self.native_specs[item["hf"]]["shape"][0], prod(self.native_specs[item["hf"]]["shape"][1:])]
                    and item["dtype"] == self.native_specs[item["hf"]]["dtype"]
                    for item in items
                ):
                    tensors[info.name] = value
                    regions.extend(items)
                    groups[info.name] = sorted({hf for item in items for hf in item.get("group", (item["hf"],))})
        return tensors, regions, groups

    def catalogue(self, weights: Mapping[str, torch.Tensor]) -> dict:
        tensors, regions, groups = self._views(weights, selected_only=False)
        full_regions = regions
        parts = []
        for region in regions:
            if (
                region["offset"][1] == 0
                and region["shape"][1] == region["global_shape"][1]
                and getattr(tensors[region["source"]], "partition_dim", None) == 0
            ):
                # DP/CP replicas own identical weights. Share complete rows
                # without copying storage or changing native conversion.
                rows = region["shape"][0]
                begin = rows * self.replica_rank // len(self.replica_ranks)
                end = rows * (self.replica_rank + 1) // len(self.replica_ranks)
                if begin == end:
                    continue
                region = {
                    **region,
                    "offset": [region["offset"][0] + begin, 0],
                    "shape": [end - begin, region["shape"][1]],
                    "storage_offset": region["storage_offset"] + begin * region["stride"],
                }
            elif self.replica_rank != 0:
                # Keep full-height column shards for replica reconstruction.
                continue
            parts.append(region)
        # Keep every qualified replica as an address-free candidate. The shared
        # geometry still uses the original disjoint DP/CP partitions below.
        self._recipes = {
            name: dict(
                key=self.iterator._vanilla_key_map[name],
                shape=tuple(value.shape),
                dtype=value.dtype,
                device=value.device,
                stride=tuple(value.stride()),
                offset=value.storage_offset(),
                nbytes=value.untyped_storage().nbytes(),
                regions=[dict(region) for region in full_regions if region["source"] == name],
                group=tuple(groups[name]),
            )
            for name, value in tensors.items()
        }
        _, views = group_storage_views(tensors)
        self._catalogue_slots = {name: item["source"] for name, item in views.items()}
        replicas = [
            dict(
                region,
                source="ordinary:" + views[region["source"]]["source"],
                group=groups[region["source"]],
                capacity=tensors[region["source"]].untyped_storage().nbytes(),
            )
            for region in full_regions
        ]
        regions = [dict(region, source="ordinary:" + views[region["source"]]["source"]) for region in parts]
        return dict(
            rank=self.rank,
            global_rank=dist.get_global_rank(self.group, self.rank),
            replica_ranks=list(self.replica_ranks),
            replica_rank=self.replica_rank,
            regions=regions,
            replica_regions=replicas,
            groups=groups,
            native_specs=self.native_specs,
        )

    def bind(self, weights: Mapping[str, torch.Tensor], *, seed: bool = False) -> dict:
        tensors, regions, groups = {}, [], {}
        for name in sorted(self.selected):
            recipe = self._recipes[name]
            if self.iterator._vanilla_key_map.get(name) != recipe["key"]:
                raise ValueError("Ordinary TP source mapping changed after negotiation")
            value = weights[recipe["key"]].detach()
            if (
                tuple(value.shape) != recipe["shape"]
                or value.dtype != recipe["dtype"]
                or value.device != recipe["device"]
                or tuple(value.stride()) != recipe["stride"]
                or value.storage_offset() != recipe["offset"]
                or value.untyped_storage().nbytes() != recipe["nbytes"]
            ):
                raise ValueError("Ordinary TP source no longer matches its cold layout")
            tensors[name] = value
            regions.extend(recipe["regions"])
            groups[name] = recipe["group"]
        sources, views = group_storage_views(tensors)
        slots = {}
        for name, view in views.items():
            expected = self._catalogue_slots[name]
            previous = slots.setdefault(view["source"], expected)
            if previous != expected:
                raise ValueError("Ordinary TP source alias groups changed")
        schema = (
            fingerprint(regions),
            tuple(dist.get_process_group_ranks(mpu.get_data_parallel_group(with_context_parallel=True))),
            tuple(
                (
                    name,
                    self._catalogue_slots[name],
                    tuple(value.shape),
                    tuple(value.stride()),
                    value.storage_offset(),
                    str(value.dtype),
                    str(value.device),
                    value.untyped_storage().nbytes(),
                    tuple(groups[name]),
                )
                for name, value in sorted(tensors.items())
            ),
        )
        if seed:
            self._schema = schema
        elif schema != self._schema:
            raise ValueError("Ordinary TP layout/aliasing changed; native reseeding is required")
        if len(set(slots.values())) != len(slots):
            raise ValueError("Ordinary TP source storage aliasing split")
        return {"ordinary:" + slots[key]: value for key, value in sources.items()}


def local_replica_writes(records: list[dict], writes: list[dict], endpoints: list[dict], receiver: dict) -> list[dict]:
    """Choose proven ordinary DP/CP equivalents on the receiver's host.

    Preserve canonical rectangles and native dependency groups. Source storage
    slots and offsets belong to each actual owner, including fused HF views.
    """
    by_global = {record["global_rank"]: record for record in records}
    regions = [{} for _ in records]
    for record, index in zip(records, regions, strict=True):
        for region in record["replica_regions"]:
            index.setdefault(region["hf"], []).append(region)
    preferred = next(
        (
            record["replica_rank"]
            for record, endpoint in zip(records, endpoints, strict=True)
            if endpoint["boot_id"] == receiver["boot_id"] and endpoint["gpu_uuid"] == receiver["gpu_uuid"]
        ),
        0,
    )
    result = []
    equivalence = ("hf", "global_shape", "offset", "shape", "stride", "dtype", "itemsize", "group")

    def contains(region: dict, write: dict) -> bool:
        itemsize = region["itemsize"]
        stride = region["stride"] * itemsize
        width = region["shape"][1] * itemsize
        begin = region["storage_offset"] * itemsize
        end = begin + (region["shape"][0] - 1) * stride + width
        if (
            write["src_offset"] < begin
            or write["src_offset"] + (write["rows"] - 1) * write["src_stride"] + write["row_bytes"] > end
        ):
            return False
        if stride == width:
            return True
        _, column = divmod(write["src_offset"] - begin, stride)
        return (write["rows"] == 1 or write["src_stride"] == stride) and column + write["row_bytes"] <= width

    for write in writes:
        owner = records[write["source_rank"]]
        originals = [
            region
            for region in regions[owner["rank"]].get(write.get("hf"), ())
            if region["source"] == write["source"] and contains(region, write)
        ]
        original = originals[0] if len(originals) == 1 else None
        candidates = []
        if original is not None:
            for rank in owner["replica_ranks"]:
                peer = by_global[rank]
                if endpoints[peer["rank"]]["boot_id"] != receiver["boot_id"]:
                    continue
                matches = [
                    region
                    for region in regions[peer["rank"]].get(write.get("hf"), ())
                    if all(region[key] == original[key] for key in equivalence)
                ]
                if len(matches) == 1:
                    candidates.append((peer, matches[0]))
        if candidates:
            peer, region = candidates[(owner["replica_rank"] + preferred) % len(candidates)]
            offset = write["src_offset"] + (region["storage_offset"] - original["storage_offset"]) * region["itemsize"]
            if (
                offset < 0
                or offset + (write["rows"] - 1) * write["src_stride"] + write["row_bytes"] > region["capacity"]
            ):
                raise ValueError("Local ordinary replica exceeds its actual storage")
            write = dict(write, source_rank=peer["rank"], source=region["source"], src_offset=offset)
        result.append(write)
    return result
