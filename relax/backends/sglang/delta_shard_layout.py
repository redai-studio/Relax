# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validated physical destinations for target-local BF16 expert coordinates."""

from __future__ import annotations

import hashlib
import inspect
import sys
from functools import lru_cache
from itertools import chain
from typing import Any

import numpy as np
import torch

from relax.utils.sparse_tensor import storage_identity


@lru_cache(maxsize=4)
def _reviewed(codes: tuple, broadcast: bool) -> bool:
    # Hash the live code objects, including the actual Qwen dispatcher and
    # all reachable ordinary expert loader helpers, against native SGLang.
    try:
        digest = hashlib.sha256()
        for code in codes:
            digest.update(inspect.getsource(code).encode())
        expected = (
            "c860b97a950c35e3f9d66d2ceba578f3558f6b4e6adfba6e6504f040ccd60544"
            if broadcast
            else "b762ab721b208ea30d9bf23bb8759de5a2f78cbfa7171a589f9b537357b51ac0"
        )
        return digest.hexdigest() == expected
    except (OSError, TypeError):
        return False


def _runtime(model: Any, broadcast: bool = False) -> tuple | None:
    qwen = sys.modules.get("sglang.srt.models.qwen3_5" if broadcast else "sglang.srt.models.qwen3_moe")
    cls = getattr(qwen, "Qwen3_5MoeForConditionalGeneration" if broadcast else "Qwen3MoeForCausalLM", None)
    if type(model) is not cls:
        return None
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    runtime = sys.modules.get("sglang.srt.runtime_context")
    location = sys.modules.get("sglang.srt.eplb.expert_location")
    try:
        parallel = runtime.get_parallel()
        functions = (
            model.load_weights.__func__,
            FusedMoE.make_expert_params_mapping.__func__,
            *(
                getattr(FusedMoE, name)
                for name in (
                    "weight_loader",
                    "_weight_loader_physical",
                    "_weight_loader_impl",
                    "_map_global_expert_id_to_local_expert_id",
                    "_maybe_load_fp8_shared_expert_as_fp4",
                    "_load_gguf_weight",
                    "_load_model_weight_or_group_weight_scale",
                    "_load_w13",
                    "_load_w2",
                )
            ),
            UnquantizedFusedMoEMethod.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load,
            UnquantizedFusedMoEMethod.load_up_proj_weight_first.fget,
            location.ExpertLocationMetadata.logical_to_all_physical,
        )
        if broadcast:
            mapper = qwen.QWEN3_5_KV_SCALE_MAPPER
            if (
                type(mapper) is not qwen.WeightsMapper
                or vars(mapper).keys() != {"orig_to_new_prefix", "orig_to_new_substr", "orig_to_new_suffix"}
                or mapper.orig_to_new_prefix
                or mapper.orig_to_new_suffix
                or mapper.orig_to_new_substr
                != {
                    ".self_attn.k_proj.k_scale": ".attn.k_scale",
                    ".self_attn.v_proj.v_scale": ".attn.v_scale",
                    ".self_attn.k_scale": ".attn.k_scale",
                    ".self_attn.v_scale": ".attn.v_scale",
                }
            ):
                return None
            functions += (mapper.apply, mapper._map_name)
        tp = 2 if broadcast else 4
        lazy_attr = "_routed_experts_weights_of_layer" if broadcast else "routed_experts_weights_of_layer"
        if (
            not _reviewed(tuple(f.__code__ for f in functions), broadcast)
            or model.named_parameters.__func__ is not torch.nn.Module.named_parameters
            or type(vars(model).get(lazy_attr)) is not qwen.LazyValue
            or (parallel.tp_size, parallel.dp_size, parallel.pp_size, parallel.moe_ep_size, parallel.moe_tp_size)
            != (tp, 1, 1, 1, tp)
            or not 0 <= parallel.tp_rank == parallel.moe_tp_rank < tp
        ):
            return None
        return parallel.tp_rank, FusedMoE, UnquantizedFusedMoEMethod
    except (AttributeError, RuntimeError, TypeError):
        return None


def marked_ordinals(marker: dict, rank: int, count: int) -> list[int]:
    if (
        not isinstance(marker, dict)
        or set(marker) != {"format", "tp_size", "tp_rank", "entries"}
        or any(type(marker[k]) is not int for k in ("format", "tp_size", "tp_rank"))
        or (marker["format"], marker["tp_size"], marker["tp_rank"]) != (1, 4, rank)
    ):
        raise ValueError("Invalid target shard lane binding")
    ordinals = marker["entries"]
    if (
        not isinstance(ordinals, list)
        or not 0 < len(ordinals) <= count
        or any(
            type(n) is not int or not 0 <= n < count or (i and n <= ordinals[i - 1]) for i, n in enumerate(ordinals)
        )
    ):
        raise ValueError("Invalid target shard entry ordinals")
    return ordinals


def _parameter(module: Any, name: str, shape: tuple, fused: type) -> torch.Tensor | None:
    value = getattr(module, name, None)
    if (
        type(value) is not torch.nn.Parameter
        or value.dtype != torch.bfloat16
        or tuple(value.shape) != shape
        or not value.is_contiguous()
        or value.storage_offset()
        or value.untyped_storage().nbytes() != value.numel() * 2
        or value.is_neg()
        or any(
            getattr(value, flag, False)
            for flag in ("is_transposed", "weight_padded", "_sglang_require_global_experts")
        )
        or getattr(getattr(value, "weight_loader", None), "__func__", None) is not fused.weight_loader
        or getattr(value.weight_loader, "__self__", None) is not module
    ):
        return None
    return value


def _layer(model: Any, layer: int, rank: int, fused: type, method: type, geometry: tuple, tp: int) -> tuple | None:
    experts, hidden, local = geometry
    module = model.model.layers[layer].mlp.experts
    if type(module) is not fused or type(module.quant_method) is not method:
        return None
    if (
        (
            module.layer_id,
            module.moe_ep_size,
            module.moe_ep_rank,
            module._expert_storage_rank,
            module.moe_tp_size,
            module.moe_tp_rank,
        )
        != (layer, 1, 0, 0, tp, rank)
        or (module._num_local_routed, module._num_global_routed, module.num_local_experts)
        != (experts, experts, experts)
        or module.intermediate_size_per_partition != local
        or module.num_fused_shared_experts != 0
        or module.moe_runner_config.is_gated is not True
        or any(
            getattr(module, flag, None) is not False
            for flag in (
                "_has_fused_shared",
                "use_presharded_weights",
                "use_padded_loading",
                "use_flashinfer_trtllm_moe",
                "use_triton_kernels",
            )
        )
        or module.quant_method.load_up_proj_weight_first is not False
    ):
        return None
    w13 = _parameter(module, "w13_weight", (experts, 2 * local, hidden), fused)
    w2 = _parameter(module, "w2_weight", (experts, hidden, local), fused)
    if w13 is None or w2 is None:
        return None
    return w13, w2


class ExpertTargets:
    """A complete seed-admitted HF → local projection view mapping, owned by
    the existing receiver."""

    @classmethod
    def build(cls, receiver: Any, *, broadcast: bool = False) -> ExpertTargets | None:
        model = receiver.model
        runtime = _runtime(model, broadcast)
        if runtime is None or receiver.capability is None or receiver.force_full_seed:
            return None
        try:
            tp = 2 if broadcast else 4
            rank, fused, method = runtime
            layers, experts, hidden, width = (
                getattr(model.config, k)
                for k in ("num_hidden_layers", "num_experts", "hidden_size", "moe_intermediate_size")
            )
            if (
                any(type(x) is not int or x <= 0 for x in (layers, experts, hidden, width))
                or width % tp
                or (getattr(model, "start_layer", 0), getattr(model, "end_layer", layers)) != (0, layers)
                or len(model.model.layers) != layers
                or (getattr(model.model, "start_layer", 0), getattr(model.model, "end_layer", layers)) != (0, layers)
                or layers * experts * 3 > len(receiver.schema)
            ):
                return None
            geometry = experts, hidden, width // tp
            local_size = hidden * (width // tp)
            targets = {}
            owners = []
            for layer in range(layers):
                weights = _layer(model, layer, rank, fused, method, geometry, tp)
                if weights is None:
                    return None
                w13, w2 = weights
                if w13.device != w2.device or w13.device != next(model.parameters()).device:
                    return None
                owners.append(weights)
                for expert in range(experts):
                    prefix = f"model.layers.{layer}.mlp.experts.{expert}."
                    for projection, weight, offset, shape in (
                        ("gate_proj", w13, 2 * expert * local_size, (width, hidden)),
                        ("up_proj", w13, (2 * expert + 1) * local_size, (width, hidden)),
                        ("down_proj", w2, expert * local_size, (hidden, width)),
                    ):
                        name = prefix + projection + ".weight"
                        if broadcast and name not in receiver.schema:
                            name = name.replace("model.layers.", "model.language_model.layers.", 1)
                        if name in receiver.full_only or receiver.schema.get(name) != (shape, "bfloat16"):
                            return None
                        targets[name] = (weight, offset, local_size)
            # A fused or duplicate spelling can address the same physical words.
            if broadcast and any(".mlp.experts." in n and n not in targets for n in receiver.schema):
                return None
            direct = frozenset(id(p) for pair in owners for p in pair) if broadcast else None
            if not validate_shard_storage(model, receiver.destinations, (), direct):
                return None
        except (AttributeError, TypeError, IndexError, ValueError):
            return None
        plan = cls()
        plan.rank, plan.geometry, plan.runtime, plan.owners, plan.targets = rank, geometry, runtime, owners, targets
        plan.tp, plan.broadcast, plan.direct = tp, broadcast, direct
        plan.down_weights = frozenset(id(pair[1]) for pair in owners)
        return plan

    def current(self, receiver: Any, raw: torch.Tensor, status: torch.Tensor) -> bool:
        model = receiver.model
        if not validate_shard_storage(model, receiver.destinations, (raw, status), self.direct):
            return False
        if _runtime(model, self.broadcast) != self.runtime:
            return False
        for layer, saved in enumerate(self.owners):
            current = _layer(model, layer, *self.runtime, self.geometry, self.tp)
            if current is None or any(a is not b for a, b in zip(current, saved, strict=True)):
                return False
        return True

    def check(self, receiver: Any, raw: torch.Tensor, status: torch.Tensor) -> None:
        if not self.current(receiver, raw, status):
            raise ValueError("Changed or aliased target shard layout")


def _byte_span(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Enclose all addressed bytes of a strided view on its own device."""
    if tensor.device.type not in ("cpu", "cuda"):
        raise ValueError("Cannot establish target tensor address range")
    if type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.layout != torch.strided:
        # Ordinary FULL may own SGLang Parameter subclasses. Their entire
        # allocation is a conservative alias bound independent of shard layout.
        storage = tensor.untyped_storage()
        size, base = storage.nbytes(), storage.data_ptr()
        if size and not base:
            raise ValueError("Cannot establish live allocation address range")
        return (base, base + size) if size else None
    if not tensor.numel():
        return None
    strides = tensor.stride()
    if min(strides, default=0) < 0:
        raise ValueError("Cannot establish target tensor address range for negative strides")
    width = tensor.element_size()
    storage = tensor.untyped_storage()
    start = tensor.data_ptr()
    base = storage.data_ptr()
    # Holes in a strided static buffer are conservatively considered occupied.
    size = width * (1 + sum((dim - 1) * stride for dim, stride in zip(tensor.shape, strides, strict=True)))
    if (
        not base
        or not base <= start <= base + storage.nbytes() - size
        or start != base + tensor.storage_offset() * width
    ):
        raise ValueError("Ordered tensor address range exceeds its storage")
    return start, start + size


def _live_tensors(model: Any) -> list[tuple[str, torch.Tensor]]:
    return [
        *(("buffer", b) for b in model.buffers()),
        *(("param", p) for _, p in model.named_parameters(remove_duplicate=False)),
    ]


def _check_address_ranges(live: list[tuple[str, torch.Tensor]], io: tuple[torch.Tensor, ...]) -> bool:
    """Reject transport collisions and report whether live parameters are
    disjoint."""
    by_device = {}
    for kind, tensor in chain(live, (("io", t) for t in io)):
        span = _byte_span(tensor)
        if span is not None:
            by_device.setdefault(tensor.device, []).append((*span, kind))
    eligible = True
    for ranges in by_device.values():
        any_end = io_end = param_end = 0
        for start, end, kind in sorted(ranges):
            if start < io_end or (kind == "io" and start < any_end):
                raise ValueError("Target transport byte ranges overlap another input, status or live model tensor")
            if kind == "io":
                io_end = end
            if start < param_end or (kind == "param" and start < any_end):
                eligible = False
            if kind == "param":
                param_end = max(param_end, end)
            any_end = max(any_end, end)
    return eligible


def validate_transport_storage(model: Any, io: tuple[torch.Tensor, ...]) -> None:
    # Final ACK only needs safe I/O, even when the ordinary layout cannot shard.
    devices = {tensor.device for tensor in io}
    _check_address_ranges([(kind, t) for kind, t in _live_tensors(model) if t.device in devices], io)


def validate_shard_storage(
    model: Any, destinations: frozenset, io: tuple[torch.Tensor, ...], direct: frozenset | None = None
) -> bool:
    live = _live_tensors(model)
    if direct is not None:
        # Only the direct expert parameters must be disjoint. Ordinary dense
        # parameters can be tied, padded, FP32, or SGLang Parameter subclasses.
        ranges = [(kind if id(tensor) in direct else "buffer", tensor) for kind, tensor in live]
        eligible = _check_address_ranges(ranges, io)
        return (
            eligible
            and all(type(t) is torch.Tensor for t in io)
            and {storage_identity(t) for _, t in live} == destinations
        )
    if not _check_address_ranges(live, io) or any(
        type(t) is not torch.Tensor or t.layout != torch.strided for t in io
    ):
        return False
    storage = set()
    for kind, tensor in live:
        if type(tensor) not in (torch.Tensor, torch.nn.Parameter) or tensor.layout != torch.strided:
            return False
        identity = storage_identity(tensor)
        if kind == "param" and (
            tensor.dtype != torch.bfloat16
            or not tensor.is_contiguous()
            or tensor.storage_offset() != 0
            or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
            or identity in storage
        ):
            return False
        storage.add(identity)
    return storage == destinations


class ShardFrame:
    """One bounded tile directory and ordered runs sharing the original consume
    fence."""

    BLOCK = 512
    # Limit each vector of temporary int64 tile indices to 256 KiB.
    _HOST_BATCH_TILES = 32 * 1024

    @classmethod
    def ordered_workspace(cls, codes: list) -> int:
        # Six int64s per tile, validation flags and the group reduction. Include
        # CUDA's minimum allocation granularity for each retained frame.
        tiles = sum((count + cls.BLOCK - 1) // cls.BLOCK for _, count in codes)
        return ((tiles * 48 + 511) // 512 * 512 + 3 * 512) if tiles else 0

    def __init__(
        self, plan: ExpertTargets, ordinals: frozenset, entries: list, raw: torch.Tensor, ends: list[int] | None = None
    ):
        self.plan = plan
        self.raw = raw
        self.runs = []
        descriptors = np.empty((len(ordinals), 6), dtype=np.int64)
        tile_count = descriptor_count = 0
        self.local_checks = []
        selected = set(ordinals)
        self.group_runs = []
        ends = iter(ends or [len(entries)])
        end, group_start = next(ends), 0
        start = 0
        while start < len(entries):
            direct = start in selected
            stop = start + 1
            while stop < end and (stop in selected) == direct:
                stop += 1
            if not direct:
                self.runs.append((False, entries[start:stop]))
            else:
                groups = {}
                for entry, positions, values in entries[start:stop]:
                    target = plan.targets.get(entry["name"])
                    if target is None or positions is None or values.dtype != torch.bfloat16:
                        raise ValueError("A partial shard must cover a seed-admitted sparse BF16 expert")
                    weight, base, limit = target
                    if plan.broadcast:
                        limit *= plan.tp
                    if weight.device != raw.device or entry["count"] > limit:
                        raise ValueError("Target shard has an invalid physical device or coordinate count")
                    groups.setdefault(weight, []).append((entry, positions, values, base, limit))
                batch = []
                for weight, parts in groups.items():
                    first = tile_count
                    for entry, positions, values, base, limit in parts:
                        count = entry["count"]
                        if raw.device.type == "cpu":
                            self.local_checks.append((positions, limit))
                        if count:
                            descriptors[descriptor_count] = (
                                tile_count,
                                entry["offset"],
                                entry["value_offset"],
                                count,
                                base,
                                limit,
                            )
                            descriptor_count += 1
                            tile_count += (count + self.BLOCK - 1) // self.BLOCK
                    batch.append((weight, first, tile_count, parts if raw.device.type == "cpu" else None))
                self.runs.append((True, batch))
            if stop == end:
                self.group_runs.append(self.runs[group_start:])
                group_start = len(self.runs)
                end = next(ends, len(entries))
            start = stop
        self.table = self._tile_table(descriptors[:descriptor_count], tile_count, raw) if tile_count else None

    @classmethod
    def _tile_table(cls, descriptors: np.ndarray, tile_count: int, raw: torch.Tensor) -> torch.Tensor:
        host = torch.empty((tile_count, 6), dtype=torch.int64, device="cpu", pin_memory=raw.is_cuda)
        table = host.numpy()
        first, position_offset, value_offset, count, base, limit = descriptors.T
        for start in range(0, tile_count, cls._HOST_BATCH_TILES):
            stop = min(start + cls._HOST_BATCH_TILES, tile_count)
            rows = table[start:stop]
            tiles = np.arange(start, stop, dtype=np.int64)
            owner = np.searchsorted(first, tiles, side="right") - 1
            offset = (tiles - first[owner]) * cls.BLOCK
            rows[:, 0] = position_offset[owner] + 4 * offset
            rows[:, 1] = value_offset[owner] + 2 * offset
            rows[:, 2] = np.minimum(count[owner] - offset, cls.BLOCK)
            rows[:, 3] = offset != 0
            rows[:, 4] = base[owner]
            rows[:, 5] = limit[owner]
        return host.to(raw.device, non_blocking=True)

    def invalid_positions(self) -> torch.Tensor | None:
        if self.table is None:
            return None
        if self.raw.device.type == "cpu":
            return torch.stack(
                [((pos < 0) | (pos >= limit)).any() | (pos[1:] <= pos[:-1]).any() for pos, limit in self.local_checks]
            ).any()
        from .delta_shard_kernel import shard_tiles

        self.invalid = torch.zeros(1, dtype=torch.int32, device=self.raw.device)
        shard_tiles[(self.table.shape[0],)](self.raw, self.table, self.raw, self.invalid, False, self.BLOCK)
        return self.invalid[0].bool()

    def scatter(self, batch: list) -> None:
        for weight, first, end, parts in batch:
            if end == first:
                continue
            target = weight.detach().view(torch.int16).view(-1)
            _, hidden, local = self.plan.geometry
            rank, tp = self.plan.rank, self.plan.tp
            down = id(weight) in self.plan.down_weights
            if self.raw.device.type == "cpu":
                for _, pos, values, base, _ in parts:
                    words = values.view(torch.int16)
                    pos = pos.long()
                    if self.plan.broadcast:
                        axis = pos % (tp * local) if down else pos // hidden
                        keep = (axis >= rank * local) & (axis < (rank + 1) * local)
                        pos, words = pos[keep], words[keep]
                        if down:
                            pos = (pos // (tp * local)) * local + pos % (tp * local) - rank * local
                        else:
                            pos = pos - rank * local * hidden
                    target.index_copy_(0, pos + base, words)
            else:
                from .delta_shard_kernel import shard_tiles

                shard_tiles[(end - first,)](
                    self.raw,
                    self.table[first:end],
                    target,
                    self.invalid,
                    True,
                    self.BLOCK,
                    self.plan.broadcast,
                    down,
                    hidden,
                    local,
                    tp,
                    rank,
                )
