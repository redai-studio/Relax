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
        if type(axis) is not int or not -len(shape) <= axis < len(shape):
            raise NotImplementedError("Bridge view is replicated or lacks a shard axis")
        axis %= len(shape)
        expected[axis] *= tp_size
        if (
            view.shard_group != "tp"
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
        for name, tensor in values:
            spec = dict(shape=list(tensor.shape), dtype=str(tensor.dtype))
            if name in self.native_specs and self.native_specs[name] != spec:
                raise ValueError("Native HF schema changed during TP seed")
            self.native_specs[name] = spec

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
                if not info.attrs.get("tensor_model_parallel") or info.attrs.get("parallel_mode") == "duplicated":
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
                    groups[info.name] = sorted({item["hf"] for item in items})
        return tensors, regions, groups

    def catalogue(self, weights: Mapping[str, torch.Tensor]) -> dict:
        tensors, regions, groups = self._views(weights, selected_only=False)
        self._recipes = {
            name: dict(
                key=self.iterator._vanilla_key_map[name],
                shape=tuple(value.shape),
                dtype=value.dtype,
                device=value.device,
                stride=tuple(value.stride()),
                offset=value.storage_offset(),
                nbytes=value.untyped_storage().nbytes(),
                regions=[dict(region) for region in regions if region["source"] == name],
                group=tuple(groups[name]),
            )
            for name, value in tensors.items()
        }
        if self.replica_rank != 0:
            tensors, regions, groups = {}, [], {}
        _, views = group_storage_views(tensors)
        self._catalogue_slots = {name: item["source"] for name, item in views.items()}
        for region in regions:
            region["source"] = "ordinary:" + views[region["source"]]["source"]
        return dict(
            rank=self.rank,
            global_rank=dist.get_global_rank(self.group, self.rank),
            replica_ranks=list(self.replica_ranks),
            replica_rank=self.replica_rank,
            regions=regions,
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
        if self.replica_rank != 0:
            tensors, groups = {}, {}
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
