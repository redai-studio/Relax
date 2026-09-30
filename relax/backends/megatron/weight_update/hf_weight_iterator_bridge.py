# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import dataclasses
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping

import torch
import torch.distributed as dist
from megatron.core import mpu


try:
    # Megatron-Bridge >= 0.6.0 moved LoRAMerge out of peft.lora into its own module.
    from megatron.bridge.peft.lora_merge import LoRAMerge
except ImportError:  # bridge <= 0.5.x
    from megatron.bridge.peft.lora import LoRAMerge


from relax.utils import device as device_utils
from relax.utils.device import device_module
from relax.utils.logging_utils import get_logger
from relax.utils.megatron_peft_utils import (
    is_lora_adapter_mode,
    is_lora_adapter_param,
    is_lora_enabled,
    is_lora_merge_mode,
)
from relax.utils.types import ParamInfo

from .bridge_converter import BridgeConverter
from .common import all_gather_param, named_params_and_buffers
from .expert_routing import ExpertLayout, ExpertRouter
from .hf_weight_iterator_base import HfWeightIteratorBase


logger = get_logger(__name__)

_NON_BLOCKING = device_utils.use_non_blocking_copy()

# Name fragments that SGLang fuses pairwise inside a single load_weights() call,
# so both halves must land in the same sync chunk:
#   - MLA HF-style : q_a_proj + kv_a_proj_with_mqa -> fused_qkv_a_proj_with_mqa
#   - MLA DSv4-native: wq_a + wkv -> wqkv_a (deepseek_v4.py `fuse_wqa_wkv`)
#   - DSv4 compressor: wkv + wgate (deepseek_v4.py COMPRESSOR_PART)
# Each of those asserts its pending-pair cache is empty when the call ends.
#
# Matching replaces the fragment with a sentinel, so the REST of the name -- the
# `.weight` vs `.weight_scale_inv` tail included -- becomes the pairing key. That
# matters: quantize_params_fp8 emits weight and scale adjacently, so keying on a
# stripped suffix alone would pair wq_a.weight with wq_a.weight_scale_inv instead
# of with its wkv counterpart.
#
# Longest fragment wins, because ".wkv." is a substring of ".compressor.wkv.".
_FUSED_PAIR_FRAGMENTS = (
    (".q_a_proj.", ".kv_a_proj_with_mqa."),
    (".wq_a.", ".wkv."),
    (".compressor.wkv.", ".compressor.wgate."),
)
_PAIR_SENTINEL = "\x00"


def _fused_pair_key(name: str) -> str | None:
    """Key that both halves of a fused pair map to, or None if `name` is not
    one.

    The sentinel carries the pair index, because two different families can
    leave identical text around their fragment: replacing ".wq_a." in
    `layers.N.attn.wq_a.weight` and ".compressor.wkv." in
    `layers.N.attn.compressor.wkv.weight` both yield `layers.N.attn<S>weight`,
    which would let a q_a half pair with a compressor half.
    """
    best = None  # (fragment, pair_index)
    for index, pair in enumerate(_FUSED_PAIR_FRAGMENTS):
        for fragment in pair:
            if fragment in name and (best is None or len(fragment) > len(best[0])):
                best = (fragment, index)
    if best is None:
        return None
    fragment, index = best
    return name.replace(fragment, f"{_PAIR_SENTINEL}{index}{_PAIR_SENTINEL}", 1)


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bridge_converter = BridgeConverter(
            args=self.args, model=self.model, quantization_config=self.quantization_config
        )
        # All ranks must initialize Bridge tasks before buffer filtering because
        # task construction contains PP collectives.
        self._bridge_converter.init_tasks()
        self.lora_merge_mode = is_lora_enabled(self.args) and is_lora_merge_mode(self.args)
        self.lora_adapter_mode = is_lora_enabled(self.args) and is_lora_adapter_mode(self.args)
        collect_adapters = self.lora_merge_mode or self.lora_adapter_mode
        buckets_result = _build_param_info_buckets(
            self.args,
            self.model,
            buffer_is_mapped=self._bridge_converter.can_convert,
            collect_adapters=collect_adapters,
        )
        self._expert_buckets, self._non_expert_buckets, self._vanilla_key_map, self._adapter_map = buckets_result
        self._expert_broadcast_caches = [({}, {}) for _ in self._expert_buckets]
        self._expert_router: ExpertRouter | None = None
        if self.lora_merge_mode:
            self.lora_alpha = self.args.lora_alpha
            self.lora_dim = self.args.lora_rank
            # Expert LoRA merge selects a per-expert adapter slice locally on the owning EP
            # rank (no ETP collective). Expert-TP > 1 would need an ETP all-gather that only
            # the single owning rank reaches -> deadlock; fail loud rather than mis-merge.
            # MoE LoRA merge is therefore supported only with expert-tensor-parallel-size=1
            # (expert-model-parallel-size / EP may be > 1). Attention/dense LoRA is unaffected.
            assert mpu.get_expert_tensor_parallel_world_size() == 1, (
                "MoE LoRA merge mode requires --expert-tensor-parallel-size 1 "
                f"(got {mpu.get_expert_tensor_parallel_world_size()}). Set ETP=1 (EP may stay > 1), "
                "or use --lora-adapter-mode: in colocate the reference stays in-process (no fold), "
                "so colocate adapter mode is unaffected by ETP. (Fully-async off-policy adapter mode "
                "DOES fold expert deltas for actor_fwd and so also needs ETP=1.)"
            )

    def configure_expert_routing(
        self, layout: ExpertLayout, *, payload_group: dist.ProcessGroup, metadata_group: dist.ProcessGroup
    ) -> None:
        if (
            mpu.get_expert_tensor_parallel_world_size() != 1
            or mpu.get_expert_model_parallel_world_size() * mpu.get_pipeline_model_parallel_world_size()
            != layout.world_size
        ):
            raise ValueError("Expert routing requires ETP=1 and unique owners across PP*EP=world")
        # Derive owners from actual local backup keys, not a PP/EP relay rank.
        # Exchange once so every rank has exactly the same source buckets.
        rank = dist.get_rank(metadata_group)
        owned = [
            dataclasses.replace(info, src_rank=rank)
            for bucket in self._expert_buckets
            for info in bucket
            if info.name in self._vanilla_key_map
        ]
        all_owned = [None] * layout.world_size
        dist.all_gather_object(all_owned, owned, group=metadata_group)
        infos = sorted((info for entries in all_owned for info in entries), key=lambda info: info.name)
        if not infos or len({info.name for info in infos}) != len(infos):
            raise ValueError("Expert routing requires nonempty, globally unique expert owners")
        self._expert_buckets = _bucket_experts_by_owner(infos, self.args.update_weight_buffer_size, layout)
        self._expert_router = ExpertRouter(layout, payload_group=payload_group, metadata_group=metadata_group)
        if rank == 0:
            logger.info("[Bridge Expert Routing] configured %d parallel source rounds", len(self._expert_buckets))

    def _iter_routed_expert_chunks(
        self, megatron_local_weights: Mapping[str, torch.Tensor]
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        router = self._expert_router
        router.reset_metrics()
        self._bridge_converter.init_tasks()
        self._bridge_converter.broadcast_and_apply_configs()
        device = device_utils.make_current_torch_device()
        convert_seconds = route_seconds = 0.0
        pending = []
        pending_bytes = [0] * router.layout.world_size
        rounds = 0
        ipc_chunks = 0
        # Reserve space for the route packing buffer and previous in-flight IPC.
        ipc_budget = max(1, self.args.update_weight_buffer_size // 3)
        for bucket_index, bucket_infos in enumerate(self._expert_buckets):
            start = time.monotonic()
            converted = []
            error = None
            # ETP=1: only owners load weights; copies and quantization run in
            # stream order, without non-owner allocations or a device sync.
            try:
                for info in bucket_infos:
                    if info.src_rank == router.rank:
                        tensor = megatron_local_weights[self._vanilla_key_map[info.name]].to(
                            device=device, non_blocking=_NON_BLOCKING
                        )
                        converted.extend(self._bridge_converter.convert(info.name, tensor))
                        del tensor
                raw_bytes = sum(info.size for info in bucket_infos if info.src_rank == router.rank)
                if sum(t.numel() * t.element_size() for _, t in converted) > raw_bytes:
                    raise ValueError("Routed MXFP4 tensors exceed their source-byte budget")
            except Exception as exc:
                error = str(exc)
            convert_seconds += time.monotonic() - start
            start = time.monotonic()
            result = router.exchange(bucket_index, converted, device, error=error, receive_budget=ipc_budget)
            route_seconds += time.monotonic() - start
            # Every rank has the same destination vector from the schema plan.
            # Accumulate per destination before taking the maximum: summing
            # individual bucket maxima overcounts buckets sent to different EP
            # shards and creates many mostly empty IPC requests.
            combined_bytes = [a + b for a, b in zip(pending_bytes, router.destination_bytes, strict=True)]
            if rounds and max(combined_bytes) > ipc_budget:
                yield pending
                pending = []
                combined_bytes = list(router.destination_bytes)
                rounds = 0
                ipc_chunks += 1
            pending.extend(result)
            pending_bytes = combined_bytes
            rounds += 1
            del converted, result
        if rounds:
            yield pending
            ipc_chunks += 1
        if router.rank == 0:
            logger.info(
                "[Bridge Expert Routing] buckets=%d ipc_chunks=%d convert=%.2fs route=%.2fs "
                "payload_bytes=%d full_replica_bytes=%d",
                len(self._expert_buckets),
                ipc_chunks,
                convert_seconds,
                route_seconds,
                router.payload_bytes,
                router.replicated_bytes,
            )

    def get_hf_weight_chunks(self, megatron_local_weights):
        if self._expert_router is not None:
            yield from self._iter_routed_expert_chunks(megatron_local_weights)
        yield from _chunk_with_mla_pairing(
            self._iter_hf_params(megatron_local_weights, include_experts=self._expert_router is None),
            chunk_size=self.args.update_weight_buffer_size,
        )

    def _iter_hf_params(self, megatron_local_weights, *, include_experts=True):
        """Yield individual (name, tensor) pairs for all params.

        Expert weights: load → TP gather + convert (src_rank only) →
        PP+EP broadcast via _broadcast_converted_bucket.

        Non-expert weights: PP/EP broadcast (BF16) → TP all-gather →
        bridge convert.
        """
        param_count = 0
        t_bcast_total = 0.0
        t_gather_total = 0.0
        t_convert_total = 0.0
        t_start = time.monotonic()
        device = device_utils.make_current_torch_device()
        rank = dist.get_rank()
        # Eagerly init bridge converter so all ranks are ready before broadcast.
        self._bridge_converter.init_tasks()
        self._bridge_converter.broadcast_and_apply_configs()

        # In LoRA merge mode, fold each adapter into its base weight at load time
        # (on the owning rank), so the rest of the pipeline sees a single merged model.
        merge_fn = self._merge_base_with_adapter if self.lora_merge_mode else None

        # --- Expert weights: quantize-before-broadcast path ---
        expert_buckets = self._expert_buckets if include_experts else []
        caches = self._expert_broadcast_caches if include_experts else []
        for bucket_infos, phase_caches in zip(expert_buckets, caches, strict=True):
            t_c0 = time.monotonic()
            params = _load_to_gpu(
                bucket_infos, megatron_local_weights, self._vanilla_key_map, device, rank, merge_fn=merge_fn
            )
            all_converted = []
            for info, param in zip(bucket_infos, params, strict=True):
                gathered = all_gather_param(self.args, info.name, param)
                if rank == info.src_rank:
                    all_converted.append(self._bridge_converter.convert(info.name, gathered))
                else:
                    all_converted.append(None)
                del gathered
            del params
            t_convert_total += time.monotonic() - t_c0

            t_b0 = time.monotonic()
            results = _broadcast_converted_bucket(bucket_infos, all_converted, device, phase_caches=phase_caches)
            t_bcast_total += time.monotonic() - t_b0
            param_count += len(results)
            yield from results
            del all_converted, results

        # --- Non-expert weights: original path ---
        for bucket_infos in self._non_expert_buckets:
            t_b0 = time.monotonic()
            params = _load_and_broadcast(
                bucket_infos, megatron_local_weights, self._vanilla_key_map, device, rank, merge_fn=merge_fn
            )
            t_b1 = time.monotonic()
            t_bcast_total += t_b1 - t_b0

            for info, param in zip(bucket_infos, params, strict=True):
                t_g0 = time.monotonic()
                gathered = all_gather_param(self.args, info.name, param)
                t_g1 = time.monotonic()
                t_gather_total += t_g1 - t_g0

                converted = self._bridge_converter.convert(info.name, gathered)
                t_convert_total += time.monotonic() - t_g1

                param_count += len(converted)
                yield from converted
                del gathered, converted

            del params

        if rank == 0:
            logger.info(
                "[Bridge Fast] params=%d | bcast=%.1fs | tp_gather=%.1fs | convert=%.1fs | total=%.1fs",
                param_count,
                t_bcast_total,
                t_gather_total,
                t_convert_total,
                time.monotonic() - t_start,
            )

    def _merge_base_with_adapter(self, info, param, megatron_local_weights, device):
        """Fold this param's LoRA adapter into the base weight, in Megatron-
        sharded space.

        Returns a new Parameter ``base + (alpha/dim)·(B @ A)`` with the SAME shape,
        dtype and TP attributes as ``param`` — so the downstream gather/convert path
        treats it identically to a non-LoRA base weight. Returns ``param`` unchanged
        for weights that have no paired adapter (layernorms, router, embeddings, ...).

        Only called on the rank that owns ``param`` (``rank == info.src_rank``). The
        merge math (incl. the TP all-gather of the sharded adapter dim) is delegated to
        ``megatron.bridge.peft.lora.LoRAMerge`` — the same implementation the upstream
        bridge export uses — to avoid drifting from its conventions.
        """
        slot = self._adapter_map.get(_base_param_prefix(info.name))
        if slot is None or "in" not in slot or "out" not in slot:
            return param

        linear_in = megatron_local_weights[slot["in"]]
        linear_out = megatron_local_weights[slot["out"]]

        if ".experts." in info.name:
            # Grouped experts: base is per-expert (``...weight{N}``) while the adapter is a
            # single grouped tensor on this EP rank. Select expert N's local slice.
            if linear_in.ndim > 2:
                ep_size = mpu.get_expert_model_parallel_world_size()
                ep_rank = mpu.get_expert_model_parallel_rank()
                global_idx = int(re.search(r"weight(\d+)$", info.name).group(1))
                local_idx = global_idx - ep_rank * self.args.num_experts // ep_size
                # Slice on CPU before moving to device: only this expert's slice is used.
                linear_in = linear_in[local_idx]
                linear_out = linear_out[local_idx]
            tp_size = mpu.get_expert_tensor_parallel_world_size()
            tp_group = mpu.get_expert_tensor_parallel_group()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()
            tp_group = mpu.get_tensor_model_parallel_group()

        linear_in = linear_in.to(device=device).float()
        linear_out = linear_out.to(device=device).float()

        merged = (
            LoRAMerge()
            .merge(
                param.data.float(),
                linear_out,
                linear_in,
                self.lora_alpha,
                self.lora_dim,
                tp_size=tp_size,
                tp_group=tp_group,
            )
            .to(param.dtype)
        )

        merged_param = torch.nn.Parameter(merged, requires_grad=False)
        for key, value in info.attrs.items():
            setattr(merged_param, key, value)
        return merged_param


def _base_param_prefix(name):
    """Strip the LoRA base suffix ``.to_wrap.weight`` (and any expert
    ``weight{N}``) so it matches the key produced by
    ``_adapter_base_prefix``."""
    return re.sub(r"\.to_wrap\.weight\d*$", "", name)


def _adapter_base_prefix(name):
    """Strip the LoRA adapter suffix so it matches ``_base_param_prefix``."""
    return name.replace(".adapter.linear_in.weight", "").replace(".adapter.linear_out.weight", "")


def _build_param_info_buckets(args, model, buffer_is_mapped: Callable[[str], bool], collect_adapters=False):
    """Build ParamInfo buckets and vanilla-key mapping at init time.

    Exchanges parameter metadata across PP/EP ranks so every rank knows about
    all params.  Also records the vanilla-key (TensorBackuper dict key) for
    each param owned by the current rank.

    When ``collect_adapters`` is set (LoRA merge mode), LoRA adapter params are
    pulled OUT of the conversion buckets (they have no standalone bridge mapping)
    and returned in ``adapter_map`` keyed by base prefix, so the iterator can
    splice them into their base weight at load time.

    Returns:
        expert_buckets: list of ParamInfo lists for expert params
        non_expert_buckets: list of ParamInfo lists for non-expert params
        vanilla_key_map: dict mapping global_name -> vanilla_key (only for
            params owned by this PP rank)
        adapter_map: dict base_prefix -> {"in": vanilla_key, "out": vanilla_key}
            for LoRA adapters owned by this rank (empty unless collect_adapters)
    """
    rank = dist.get_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    vanilla_iter = named_params_and_buffers(args, model, convert_to_global_name=False, include_persistent_buffers=True)
    global_iter = named_params_and_buffers(args, model, convert_to_global_name=True, include_persistent_buffers=True)

    local_infos = {}
    vanilla_key_map = {}
    adapter_map: dict[str, dict[str, str]] = {}
    for (v_name, v_param), (g_name, _g_param) in zip(vanilla_iter, global_iter, strict=True):
        if not isinstance(v_param, torch.nn.Parameter) and not buffer_is_mapped(g_name):
            continue
        if collect_adapters and is_lora_adapter_param(g_name):
            # LoRA adapter param: keep it out of the conversion buckets (no standalone
            # bridge mapping) and record its vanilla key for load-time merging.
            slot = adapter_map.setdefault(_adapter_base_prefix(g_name), {})
            if ".linear_in." in g_name:
                slot["in"] = v_name
            elif ".linear_out." in g_name:
                slot["out"] = v_name
            continue
        local_infos[g_name] = ParamInfo(
            name=g_name,
            dtype=v_param.dtype,
            shape=v_param.shape,
            attrs={
                "tensor_model_parallel": getattr(v_param, "tensor_model_parallel", False),
                "partition_dim": getattr(v_param, "partition_dim", -1),
                "partition_stride": getattr(v_param, "partition_stride", 1),
                "parallel_mode": getattr(v_param, "parallel_mode", None),
            },
            size=v_param.numel() * v_param.element_size(),
            src_rank=rank,
        )
        vanilla_key_map[g_name] = v_name

    # Exchange across PP so every rank has all PP stages' param infos.
    if pp_size > 1:
        pp_infos_list: list[None | tuple[int, dict]] = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, local_infos),
            object_list=pp_infos_list,
            group=mpu.get_pipeline_model_parallel_group(),
        )
        for src_rank, infos in pp_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                if name in local_infos:
                    if local_infos[name].src_rank > src_rank:
                        local_infos[name] = info
                else:
                    local_infos[name] = info

    # Exchange across EP so every rank has all expert indices.
    # Only expert params need src_rank update — non-expert params are
    # replicated across EP and already have the correct PP-local src_rank.
    if ep_size > 1:
        ep_infos_list: list[None | tuple[int, dict]] = [None] * ep_size
        dist.all_gather_object(
            obj=(rank, local_infos),
            object_list=ep_infos_list,
            group=mpu.get_expert_model_parallel_group(),
        )
        for src_rank, infos in ep_infos_list:
            for name, info in infos.items():
                if name not in local_infos:
                    local_infos[name] = dataclasses.replace(info, src_rank=src_rank)
                elif ".experts." in name and info.src_rank < local_infos[name].src_rank:
                    local_infos[name] = dataclasses.replace(local_infos[name], src_rank=info.src_rank)

    # Sort deterministically and split expert / non-expert.
    all_infos = sorted(local_infos.values(), key=lambda info: info.name)
    expert_infos = [i for i in all_infos if ".experts." in i.name]
    non_expert_infos = [i for i in all_infos if ".experts." not in i.name]

    expert_buckets = _bucket_by_size(expert_infos, args)
    non_expert_buckets = _bucket_by_size(non_expert_infos, args)

    return expert_buckets, non_expert_buckets, vanilla_key_map, adapter_map


def _bucket_experts_by_owner(infos: list[ParamInfo], buffer_size: int, layout: ExpertLayout) -> list[list[ParamInfo]]:
    """Run owner-local buckets concurrently with conservative BF16 byte
    bounds."""
    by_owner: dict[int, list[ParamInfo]] = {}
    destination_owners: dict[int, set[int]] = {}
    for info in infos:
        match = re.search(r"\.experts\.linear_fc[12]\.weight(\d+)$", info.name)
        if match is None or not 0 <= info.src_rank < layout.world_size:
            raise ValueError(f"Unsupported routed Megatron expert: {info.name}")
        name = f"model.layers.0.block_sparse_moe.experts.{match[1]}.w1.weight_packed"
        for destination in layout.destinations(name):
            destination_owners.setdefault(destination, set()).add(info.src_rank)
        by_owner.setdefault(info.src_rank, []).append(info)
    fanin = max(len(owners) for owners in destination_owners.values())
    # MXFP4 packed+scale bytes must not exceed the original BF16 bytes (checked
    # during conversion). Each destination receives at most `fanin` such buckets.
    budget = min(buffer_size // len(layout.engine_offsets), buffer_size // 3 // fanin)
    owner_buckets = []
    for owner in sorted(by_owner):
        buckets: list[list[ParamInfo]] = [[]]
        used = 0
        for info in sorted(by_owner[owner], key=lambda item: item.name):
            if not 0 < info.size <= budget:
                raise ValueError(f"Expert {info.name} has {info.size} bytes, exceeding owner budget {budget}")
            if used + info.size > budget:
                buckets.append([])
                used = 0
            buckets[-1].append(info)
            used += info.size
        owner_buckets.append(buckets)
    return [
        [info for buckets in owner_buckets if index < len(buckets) for info in buckets[index]]
        for index in range(max(map(len, owner_buckets)))
    ]


def _bucket_by_size(infos, args, *, buffer_size: int | None = None):
    if not infos:
        return []
    buffer_size = args.update_weight_buffer_size if buffer_size is None else buffer_size
    buckets: list[list[ParamInfo]] = [[]]
    bucket_bytes = 0
    for info in infos:
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()
        param_size = info.size * tp_size

        if bucket_bytes + param_size > buffer_size and buckets[-1]:
            buckets.append([])
            bucket_bytes = 0
        buckets[-1].append(info)
        bucket_bytes += param_size
    return buckets


def _load_to_gpu(bucket_infos, megatron_local_weights, vanilla_key_map, device, rank, merge_fn=None):
    """Load params from CPU dict to GPU.

    No broadcast. When ``merge_fn`` is given (LoRA merge mode), each owned
    param is passed through it to fold in its LoRA adapter before the
    gather/convert pipeline.
    """
    params = []
    for info in bucket_infos:
        if rank == info.src_rank:
            vanilla_key = vanilla_key_map[info.name]
            cpu_tensor = megatron_local_weights[vanilla_key]
            gpu_tensor = cpu_tensor.to(device=device, non_blocking=_NON_BLOCKING)
            param = torch.nn.Parameter(gpu_tensor, requires_grad=False)
        else:
            param = torch.nn.Parameter(torch.empty(info.shape, dtype=info.dtype, device=device), requires_grad=False)
        for key, value in info.attrs.items():
            setattr(param, key, value)
        if merge_fn is not None and rank == info.src_rank:
            param = merge_fn(info, param, megatron_local_weights, device)
        params.append(param)
    device_module.synchronize()
    return params


def _pp_broadcast(bucket_infos, params):
    """PP-broadcast params in-place."""
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    if pp_size <= 1:
        return
    handles = []
    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_ranks = dist.get_process_group_ranks(pp_group)
    for info, param in zip(bucket_infos, params, strict=True):
        if info.src_rank in pp_ranks:
            handles.append(dist.broadcast(param, src=info.src_rank, group=pp_group, async_op=True))
    for handle in handles:
        handle.wait()


def _ep_broadcast(bucket_infos, params):
    """EP-broadcast expert params in-place."""
    ep_size = mpu.get_expert_model_parallel_world_size()
    if ep_size <= 1:
        return
    handles = []
    ep_group = mpu.get_expert_model_parallel_group()
    ep_ranks = dist.get_process_group_ranks(ep_group)
    rank = dist.get_rank()
    for info, param in zip(bucket_infos, params, strict=True):
        if ".experts." in info.name:
            src = info.src_rank if info.src_rank in ep_ranks else rank
            handles.append(dist.broadcast(param, src=src, group=ep_group, async_op=True))
    for handle in handles:
        handle.wait()


def _load_and_broadcast(bucket_infos, megatron_local_weights, vanilla_key_map, device, rank, merge_fn=None):
    """Load params from CPU dict, PP-broadcast, EP-broadcast.

    After this call every rank holds all params from all PP stages and all EP
    shards (still TP-sharded).  Mirrors the broadcast logic in
    ``HfWeightIteratorDirect._get_megatron_full_params``.

    LoRA adapters are merged at load time (before broadcast) so the merged base
    weight is what propagates to the non-owning PP/EP ranks.
    """
    params = _load_to_gpu(bucket_infos, megatron_local_weights, vanilla_key_map, device, rank, merge_fn=merge_fn)
    _pp_broadcast(bucket_infos, params)
    _ep_broadcast(bucket_infos, params)
    return params


def _broadcast_converted_bucket(bucket_infos, all_converted, device, phase_caches: tuple[dict, dict] | None = None):
    """Broadcast converted expert tensors across PP and EP groups.

    ``all_converted[i]`` is ``bridge_converter.convert()`` output for
    ``bucket_infos[i]`` on the owning rank, or ``None`` on non-owners.

    Two-phase NCCL broadcast: PP first, then EP.
    """
    rank = dist.get_rank()

    pp_size = mpu.get_pipeline_model_parallel_world_size()
    if pp_size > 1:
        all_converted = _broadcast_converted_phase(
            bucket_infos,
            all_converted,
            device,
            rank,
            group=mpu.get_pipeline_model_parallel_group(),
            plan_cache=phase_caches[0] if phase_caches is not None else None,
        )

    ep_size = mpu.get_expert_model_parallel_world_size()
    if ep_size > 1:
        all_converted = _broadcast_converted_phase(
            bucket_infos,
            all_converted,
            device,
            rank,
            group=mpu.get_expert_model_parallel_group(),
            plan_cache=phase_caches[1] if phase_caches is not None else None,
        )

    out: list[tuple[str, torch.Tensor]] = []
    for converted in all_converted:
        if converted is not None:
            out.extend(converted)
    return out


# dtype ↔ int encoding for NCCL metadata tensor
_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.int8: 5,
    torch.uint8: 6,
    torch.float8_e4m3fn: 7,
    torch.float8_e5m2: 8,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


def _compute_slot_size(all_converted, bucket_infos):
    """Compute the fixed int count per slot for metadata encoding.

    Every slot (including empty ones) must use the same number of ints so that
    allreduce(SUM) aligns correctly across ranks.
    """
    max_ints = 2  # header: [src+1, n_tensors]
    for converted in all_converted:
        if converted is None:
            continue
        n = 2
        for name, tensor in converted:
            n += 1 + len(name.encode("utf-8")) + 1 + tensor.ndim + 1
        max_ints = max(max_ints, n)
    return max_ints


def _encode_metadata(all_converted, bucket_infos, group_ranks_set, rank, slot_size=0):
    """Encode converted tensor metadata into a fixed-width int64 tensor.

    Each slot occupies exactly ``slot_size`` ints (zero-padded), making the
    total length ``len(bucket_infos) * slot_size``.  This enables correct
    allreduce(SUM) when only one rank has data per slot.

    Format per slot (padded to slot_size):
      [src_rank+1, n_tensors, (name_len, *name_bytes, ndim, *shape, dtype_code) × N, 0...]
    Empty slots: all zeros.
    """
    if slot_size == 0:
        slot_size = _compute_slot_size(all_converted, bucket_infos)
    n_slots = len(bucket_infos)
    buf = [0] * (n_slots * slot_size)
    for i, (info, converted) in enumerate(zip(bucket_infos, all_converted)):
        base = i * slot_size
        if converted is None:
            continue
        src = info.src_rank if info.src_rank in group_ranks_set else rank
        pos = base
        buf[pos] = src + 1
        pos += 1
        buf[pos] = len(converted)
        pos += 1
        for name, tensor in converted:
            name_bytes = name.encode("utf-8")
            buf[pos] = len(name_bytes)
            pos += 1
            for b in name_bytes:
                buf[pos] = b
                pos += 1
            buf[pos] = tensor.ndim
            pos += 1
            for s in tensor.shape:
                buf[pos] = s
                pos += 1
            buf[pos] = _DTYPE_TO_CODE[tensor.dtype]
            pos += 1
    return torch.tensor(buf, dtype=torch.int64, device="cpu")


def _decode_metadata(meta_tensor, slot_size):
    """Decode fixed-width int64 metadata tensor back to per-slot results.

    Each slot occupies ``slot_size`` ints.  src_rank is stored as src_rank+1; 0
    means empty slot.
    """
    data = meta_tensor.tolist()
    n_slots = len(data) // slot_size
    slots = []
    for i in range(n_slots):
        base = i * slot_size
        src_encoded = data[base]
        n_tensors = data[base + 1]
        if src_encoded == 0:
            slots.append(None)
            continue
        src = src_encoded - 1
        pos = base + 2
        tensors_meta = []
        for _ in range(n_tensors):
            name_len = data[pos]
            pos += 1
            name_bytes = bytes(data[pos : pos + name_len])
            pos += name_len
            name = name_bytes.decode("utf-8")
            ndim = data[pos]
            pos += 1
            shape = tuple(data[pos : pos + ndim])
            pos += ndim
            dtype_code = data[pos]
            pos += 1
            tensors_meta.append((name, shape, _CODE_TO_DTYPE[dtype_code]))
        slots.append((src, tensors_meta))
    return slots


def _broadcast_converted_phase(bucket_infos, all_converted, device, rank, group, plan_cache: dict | None = None):
    """Single-group broadcast of converted tensors using only NCCL.

    1. MAX agrees on slot size and (when caching) whether any rank needs a refresh.
    2. On refresh, SUM merges non-overlapping metadata contributions. Warm
       caches skip this large exchange and reuse CPU descriptors and byte sizes.
    3. Fresh data buffers are broadcast from their owners on every call.
    """
    group_ranks = dist.get_process_group_ranks(group)
    group_ranks_set = set(group_ranks)

    # Cache only CPU descriptors, never tensors, process groups or Work handles.
    # Ownership/schema/topology changes must invalidate on EVERY group member;
    # piggyback the miss bit on the existing size MAX, keeping collective order.
    signature = (
        tuple(group_ranks),
        rank,
        tuple(
            (
                info.src_rank,
                None if converted is None else tuple((name, tuple(t.shape), t.dtype) for name, t in converted),
            )
            for info, converted in zip(bucket_infos, all_converted, strict=True)
        ),
    )
    cache_hit = plan_cache is not None and plan_cache.get("signature") == signature
    local_size = plan_cache["local_size"] if cache_hit else _compute_slot_size(all_converted, bucket_infos)
    if plan_cache is None:
        slot_size_t = torch.tensor([local_size], dtype=torch.int64, device=device)
        dist.all_reduce(slot_size_t, op=dist.ReduceOp.MAX, group=group)
        slot_size = slot_size_t.item()
        refresh = True
    else:
        agreement = torch.tensor([local_size, int(not cache_hit)], dtype=torch.int64, device=device)
        dist.all_reduce(agreement, op=dist.ReduceOp.MAX, group=group)
        # Replaces the pre-existing scalar D2H synchronization; no extra sync.
        slot_size, refresh = agreement.cpu().tolist()

    if refresh:
        local_meta_tensor = _encode_metadata(all_converted, bucket_infos, group_ranks_set, rank, slot_size)
        meta_buf = local_meta_tensor.to(device)
        dist.all_reduce(meta_buf, op=dist.ReduceOp.SUM, group=group)
        merged_slots = _decode_metadata(meta_buf.cpu(), slot_size)
        src_to_slots: dict[int, list] = {}
        src_bytes: dict[int, int] = {}
        element_sizes = {dtype: torch.empty(0, dtype=dtype).element_size() for dtype in _DTYPE_TO_CODE}
        for i, slot in enumerate(merged_slots):
            if slot is None:
                continue
            src, param_meta = slot
            sized_meta = [
                (name, shape, dtype, element_sizes[dtype] * torch.Size(shape).numel())
                for name, shape, dtype in param_meta
            ]
            src_to_slots.setdefault(src, []).append((i, sized_meta))
            src_bytes[src] = src_bytes.get(src, 0) + sum(entry[3] for entry in sized_meta)
        if plan_cache is not None:
            plan_cache.update(signature=signature, local_size=local_size, slots=src_to_slots, sizes=src_bytes)
    else:
        src_to_slots = plan_cache["slots"]
        src_bytes = plan_cache["sizes"]

    result = list(all_converted)
    handles = []
    unpack_tasks: list[tuple[int, torch.Tensor, list[tuple[int, list]]]] = []

    for src, slot_list in src_to_slots.items():
        is_owner = rank == src
        total_bytes = src_bytes[src]

        if is_owner:
            parts = []
            for i, param_meta in slot_list:
                for j, (_name, _shape, _dtype, _n_bytes) in enumerate(param_meta):
                    parts.append(all_converted[i][j][1].contiguous().flatten().view(torch.uint8))
            buf = torch.cat(parts).to(device) if parts else torch.empty(0, dtype=torch.uint8, device=device)
        else:
            buf = torch.empty(total_bytes, dtype=torch.uint8, device=device)

        if total_bytes:
            handles.append(dist.broadcast(buf, src=src, group=group, async_op=True))
        unpack_tasks.append((src, buf, slot_list))

    for h in handles:
        h.wait()

    # Unpack buffers back into named tensors
    for src, buf, slot_list in unpack_tasks:
        is_owner = rank == src
        offset = 0
        for i, param_meta in slot_list:
            tensors: list[tuple[str, torch.Tensor]] = []
            for j, (name, shape, dtype, n_bytes) in enumerate(param_meta):
                if is_owner:
                    tensor = all_converted[i][j][1]
                else:
                    tensor = buf[offset : offset + n_bytes].view(dtype).reshape(shape)
                offset += n_bytes
                tensors.append((name, tensor))
            result[i] = tensors

    return result


def _chunk_with_mla_pairing(named_params, chunk_size):
    """Chunk weights by size while keeping fused weight pairs together.

    SGLang fuses ``q_a_proj`` + ``kv_a_proj_with_mqa`` into
    ``fused_qkv_a_proj_with_mqa``, and the DSv4 compressor's ``wkv`` + ``wgate``
    into one param, each via a per-call cache dict. Every chunk triggers a
    separate ``load_weights`` call, so both halves **must** be in the same chunk
    -- DSv4 even asserts its cache is empty when the call ends.

    Strategy: buffer any unpaired half and flush it together with its partner
    when the partner arrives.  All other weights pass through to the normal
    size-based chunking logic.
    """
    bucket: list[tuple[str, torch.Tensor]] = []
    bucket_size = 0
    pending_pairs: OrderedDict[str, tuple[str, torch.Tensor]] = OrderedDict()

    for name, tensor in named_params:
        pair_key = _fused_pair_key(name)

        if pair_key is not None:
            if pair_key in pending_pairs:
                partner_name, partner_tensor = pending_pairs.pop(pair_key)
                pair = [(partner_name, partner_tensor), (name, tensor)]
                pair_size = partner_tensor.nbytes + tensor.nbytes

                if bucket and (bucket_size + pair_size) >= chunk_size:
                    yield bucket
                    bucket = []
                    bucket_size = 0

                bucket.extend(pair)
                bucket_size += pair_size
            else:
                pending_pairs[pair_key] = (name, tensor)
        else:
            obj_size = tensor.nbytes
            if bucket and (bucket_size + obj_size) >= chunk_size:
                yield bucket
                bucket = []
                bucket_size = 0

            bucket.append((name, tensor))
            bucket_size += obj_size

    for pair_key, (name, tensor) in pending_pairs.items():
        if dist.get_rank() == 0:
            logger.warning("[Bridge Export] Unpaired fused weight: %s (pair_key=%r)", name, pair_key)
        obj_size = tensor.nbytes
        if bucket and (bucket_size + obj_size) >= chunk_size:
            yield bucket
            bucket = []
            bucket_size = 0
        bucket.append((name, tensor))
        bucket_size += obj_size

    if bucket:
        yield bucket
