# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron/TrainActor integration: receive, prepare and replay model
decisions."""

import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import torch

from relax.utils import device as device_utils
from relax.utils.env import Envs
from relax.utils.logging_utils import get_logger

from ..config import references_enabled, rollout_kinds
from ..consumer import (
    Consumer,
    ConsumerBatchLayout,
    ConsumerLayerBinding,
    ConsumerModelLayout,
    ConsumerReadSession,
    ConsumerRecord,
    ConsumerTensorCache,
)
from ..core import (
    ConsumerChannel,
    ConsumerPhase,
    ReplayManager,
    ReplayRegion,
    replay_data_manager,
)
from ..ref_data import REFERENCE_FIELD, MoERouteSpan, ReplayRef, selection_route_spans


P = ParamSpec("P")
R = TypeVar("R")
ROUTING_REPLAY = None
_VALID_STAGES = {"fallthrough", "replay_forward", "replay_backward"}
logger = get_logger(__name__)


class MegatronConsumer(Consumer):
    """Relax Actor preparation; the shared Consumer owns only receive/read
    semantics."""

    def prepare(
        self,
        actor: Any,
        data_iterator: Any,
        num_microbatches: Any,
        batch: dict[str, Any],
        *,
        region: ReplayRegion | None = None,
        cache: ConsumerTensorCache | None = None,
    ) -> ConsumerTensorCache | None:
        self.manager._ensure_open()
        if region is not None and region.manager is not self.manager:
            raise ValueError("Replay region belongs to another manager")
        if "moe" in self.kinds:
            if REFERENCE_FIELD in batch:
                if cache is None:
                    cache = ConsumerTensorCache(self.manager.config.materialization_bytes)
                fill_reference_replay(actor, data_iterator, num_microbatches, "moe", region or self.manager, cache)
            else:
                fill_full_moe_replay(actor, data_iterator, num_microbatches, batch)
        self.bind_batch_inputs(actor.model, batch, region)
        return cache

    def bind_batch_inputs(self, models: Any, batch: dict[str, Any], region: ReplayRegion | None) -> None:
        """Rebind inputs that follow the batch, including after Hybrid merges
        its sub-batches.

        R3 bindings follow the microbatch schedule and survive the merge.
        """
        if "mm_embedding" in self.kinds:
            from .mm_megatron import prepare_mm_replay_models

            if region is None:
                raise ValueError("MM replay requires a queue partition consumption region")
            prepare_mm_replay_models(self.manager, models, batch, region)

    def streaming_binding(self, args: Any, models: Any, partition: str) -> Callable[[Any, dict], None]:
        """Bind only the current pipeline chunk after the schedule fetches a
        microbatch."""
        routers = {id(model): [] for model in models}
        has_media = "mm_embedding" in self.kinds
        has_moe = "moe" in self.kinds
        if has_moe:
            for chunk, layer, router in local_moe_routers(models, args.num_layers):
                routers[id(models[chunk])].append((chunk, layer, router))

        if not self.manager.uses_references:

            def bind_full(model: Any, batch: dict) -> None:
                bind_full_moe_batch(args, batch, routers[id(model)], batch["unconcat_tokens"])

            return bind_full

        region = self.manager.region(partition)
        if has_media:
            from .mm_megatron import bind_mm_replay_batch

            def bind_media(model: Any, batch: dict) -> None:
                self.bind_batch_inputs(models, batch, region)
                bind_mm_replay_batch(batch, model)

            if not has_moe:
                return bind_media

        layouts = {model_id: _moe_layout(args, layers) for model_id, layers in routers.items()}
        cache = ConsumerTensorCache(self.manager.config.materialization_bytes)
        device = device_utils.make_current_torch_device()

        next_microbatch = {model_id: 0 for model_id in layouts}

        def bind(model: Any, batch: dict) -> None:
            if has_media:
                bind_media(model, batch)
            if not has_moe:
                return
            model_id = id(model)
            _bind_reference_batch(
                args,
                batch,
                batch["unconcat_tokens"],
                layouts[model_id],
                region,
                cache,
                device,
                next_microbatch[model_id],
            )
            next_microbatch[model_id] += 1

        return bind


def log_reference_replay_metrics(cache: ConsumerTensorCache | None, args: Any, rollout_id: int) -> None:
    """Report foreground CPU waits without adding CUDA syncs or collectives.

    Every rank logs its own counters; ClearML uses the same primary rank as the
    existing training performance report, not a cross-rank maximum.
    """
    from .mm_megatron import log_mm_replay_metrics

    log_mm_replay_metrics(args, rollout_id)
    if cache is None:
        return
    ahead = cache._read_ahead
    if not ahead.read_calls:
        return
    from megatron.core import mpu

    from relax.utils import tracking_utils
    from relax.utils.metrics.metric_utils import compute_rollout_step
    from relax.utils.timer import Timer

    Timer().add("replay_read_cpu", ahead.read_seconds)
    Timer().add("replay_consumer_wait", ahead.demand_wait_seconds)
    Timer().add("replay_prefetch_read", ahead.prefetch_read_seconds)
    metrics = {
        "replay/read_calls": ahead.read_calls,
        "replay/cold_reads": ahead.cold_reads,
        "replay/prefetch_hits": ahead.prefetch_hits,
        "replay/prefetch_ready_hits": ahead.prefetch_ready_hits,
        "replay/prefetch_windows": ahead.prefetch_windows,
        "replay/cross_group_windows": ahead.cross_group_windows,
        "replay/prefetched_layers": ahead.prefetched_layers,
        "replay/max_window_layers": ahead.max_window_layers,
        "replay/cache_hits": cache.hits,
        "replay/cache_evictions": cache.evictions,
        "replay/cache_peak_bytes": cache.peak_bytes,
        "replay/prefetch_peak_bytes": cache.prefetch_peak_bytes,
    }
    coordinates = {
        "tp": mpu.get_tensor_model_parallel_rank(),
        "pp": mpu.get_pipeline_model_parallel_rank(),
        "cp": mpu.get_context_parallel_rank(),
        "dp": mpu.get_data_parallel_rank(with_context_parallel=True),
    }
    logger.info(
        "replay_consumer %s: %s",
        rollout_id,
        {
            **coordinates,
            **metrics,
            "read_cpu_seconds": ahead.read_seconds,
            "demand_wait_seconds": ahead.demand_wait_seconds,
        },
    )
    if coordinates["tp"] == 0 and mpu.is_pipeline_last_stage() and coordinates["dp"] == 0:
        tracking_utils.log_direct(args, compute_rollout_step(args, rollout_id), metrics)


def pack_token_rows(
    args: Any,
    rows: list[torch.Tensor],
    lengths: list[int],
    pad: Callable[[torch.Tensor, int], torch.Tensor],
    *,
    sequence_parallel: bool,
    has_mm_inputs: bool = False,
) -> torch.Tensor:
    """The existing token layout, independent of what each row contains."""
    from megatron.core import mpu

    from relax.backends.megatron.cp_utils import maybe_padded_total_lengths, slice_with_cp

    tp = mpu.get_tensor_model_parallel_world_size()
    cp = mpu.get_context_parallel_world_size()
    unsplit = has_mm_inputs or getattr(args, "is_vl_model", False) or getattr(args, "uses_unsplit_forward", False)
    padded = maybe_padded_total_lengths(lengths, args.qkv_format, unsplit)
    if padded is not None:
        rows = [pad(value, padded[i] - value.shape[0]) for i, value in enumerate(rows)]
        packed = torch.cat([slice_with_cp(value, pad) for value in rows], dim=0)
    else:
        rows = [pad(value, 1) for value in rows]
        alignment = tp * args.data_pad_size_multiplier
        if args.allgather_cp and cp > 1:
            packed = torch.cat(rows, dim=0)
            count = (-packed.shape[0]) % (cp * alignment)
            if count:
                packed = pad(packed, count)
            packed = packed.chunk(cp, dim=0)[mpu.get_context_parallel_rank()]
        else:
            packed = torch.cat([slice_with_cp(value, pad) for value in rows], dim=0)
            count = (-packed.shape[0]) % alignment
            if count:
                packed = pad(packed, count)
    if sequence_parallel:
        assert packed.shape[0] % tp == 0
        packed = packed.chunk(tp, dim=0)[mpu.get_tensor_model_parallel_rank()]
    return packed


def consumer_batch_layout(
    args: Any,
    refs: list[ReplayRef],
    lengths: list[int],
    *,
    sequence_parallel: bool,
    sample_spans: list[tuple[MoERouteSpan, ...]] | None = None,
    has_mm_inputs: bool = False,
) -> ConsumerBatchLayout:
    import numpy as np

    canonical = {}
    for index, ref in enumerate(refs):
        if len(ref.record) != 32 or ref.rows >= 2**31 or len(bytes.fromhex(ref.record)) != 16:
            raise ValueError("Replay record cannot be represented by token handles")
        if ref.record in canonical:
            previous, _ = canonical[ref.record]
            if previous != ref:
                raise ValueError("Conflicting manifests for one replay record")
        else:
            canonical[ref.record] = (ref, index)
    if sample_spans is None:
        sample_spans = [(MoERouteSpan(ref, 0, ref.rows),) for ref in refs]
    rows = []
    for spans, length in zip(sample_spans, lengths, strict=True):
        if not spans or spans[-1].stop != length - 1:
            raise ValueError("Replay reference row count differs from training tokens")
        coordinates = np.empty((length - 1, 2), dtype=np.int64)
        for span in spans:
            coordinates[span.start : span.stop, 0] = canonical[span.ref.record][1]
            coordinates[span.start : span.stop, 1] = np.arange(span.source_start, span.ref.rows, dtype=np.int64)
        rows.append(torch.from_numpy(coordinates))

    def pad(value: torch.Tensor, count: int) -> torch.Tensor:
        # Each padding operation restarts its offset, exactly as the original
        # MoE padding formula does. Do not replace this with a global offset.
        padding = np.empty((count, 2), dtype=np.int64)
        padding[:, 0] = -1
        padding[:, 1] = np.arange(count, dtype=np.int64)
        return torch.cat((value, torch.from_numpy(padding)), dim=0)

    coordinates = pack_token_rows(
        args, rows, lengths, pad, sequence_parallel=sequence_parallel, has_mm_inputs=has_mm_inputs
    ).numpy()
    return ConsumerBatchLayout.from_coordinates(coordinates)


def _local_layer_ranges(models: Any) -> Iterator[tuple[Any, range]]:
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
    from megatron.core.utils import get_model_config

    for vp_stage, model in enumerate(models):
        config = get_model_config(model)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        yield config, range(offset, offset + get_num_layers_to_build(config, vp_stage=vp_stage))


def local_moe_routers(models: Any, num_layers: int) -> Iterator[tuple[int, int, "RoutingReplay"]]:
    """Match captured columns to the final global layer numbers assigned by
    Megatron."""
    from megatron.core.transformer.moe.router import TopKRouter

    seen = set()
    for chunk, model in enumerate(models):
        for module in model.modules():
            replay = getattr(module, "routing_replay", None)
            if getattr(module, "is_mtp_layer", False):
                continue
            if replay is None:
                if isinstance(module, TopKRouter):
                    raise RuntimeError("Megatron MoE router has no Relax routing replay hook")
                continue
            layer = module.layer_number - 1
            if not 0 <= layer < num_layers or layer in seen:
                raise ValueError(f"Invalid or duplicate MoE replay layer number: {module.layer_number}")
            seen.add(layer)
            yield chunk, layer, replay


def bind_full_moe_batch(args: Any, batch: dict, routers: list, tokens: list[torch.Tensor]) -> None:
    """Bind dense capture rows with exactly the same token packing as
    references."""
    if not routers:
        return
    rows = [
        row.reshape(row.shape[0], args.num_layers, args.moe_router_topk) for row in batch["rollout_routed_experts"]
    ]
    lengths = [len(sample) for sample in tokens]
    if any(row.shape[0] != size - 1 for row, size in zip(rows, lengths, strict=True)):
        raise ValueError("MoE replay row count differs from training tokens")

    def pad(value: torch.Tensor, count: int) -> torch.Tensor:
        shape = (count, args.num_layers, args.moe_router_topk)
        padding = torch.arange(count * args.num_layers * args.moe_router_topk, device=value.device, dtype=value.dtype)
        return torch.cat((value, padding.reshape(shape) % args.num_experts))

    packed = pack_token_rows(
        args,
        rows,
        lengths,
        pad,
        sequence_parallel=args.sequence_parallel,
        has_mm_inputs=batch.get("multimodal_train_inputs") is not None,
    )
    for _, layer, router in routers:
        router.record(packed[:, layer])


def _moe_layout(args: Any, routers: Iterable[tuple[int, int, "RoutingReplay"]]) -> ConsumerModelLayout:
    bindings = tuple(ConsumerLayerBinding(layer, router.channel) for _, layer, router in routers)
    return ConsumerModelLayout("moe", args.moe_router_topk, args.num_layers, bindings)


def _bind_microbatch(
    layout: ConsumerModelLayout,
    reader: ConsumerReadSession,
    batch_layout: ConsumerBatchLayout,
    microbatch: int,
    device: Any,
    num_experts: int,
    cache: ConsumerTensorCache | None = None,
) -> None:
    from ..consumer import ConsumerReadTask, bind_read_task

    bind_read_task(ConsumerReadTask(layout, batch_layout, microbatch, device, num_experts), reader, cache)


def _bind_reference_batch(
    args: Any,
    batch: dict,
    tokens: list[torch.Tensor],
    layout: ConsumerModelLayout,
    readers: ReplayManager | ReplayRegion,
    cache: ConsumerTensorCache,
    device: torch.device,
    microbatch: int,
) -> None:
    if not layout.bindings:
        return
    kind = layout.kind
    sample_spans = [selection_route_spans(ref[kind], kind) for ref in batch[REFERENCE_FIELD]]
    refs = list(dict.fromkeys(span.ref for spans in sample_spans for span in spans))
    if any(ref.layers != layout.capture_layers or ref.topk != layout.topk for ref in refs):
        raise ValueError("Replay reference shape does not match model")
    batch_layout = consumer_batch_layout(
        args,
        refs,
        [len(row) for row in tokens],
        sequence_parallel=kind == "moe" and args.sequence_parallel,
        sample_spans=sample_spans,
        has_mm_inputs=batch.get("multimodal_train_inputs") is not None,
    )
    with readers.reader(refs) as reader:
        _bind_microbatch(
            layout, reader, batch_layout, microbatch, device, args.num_experts if kind == "moe" else 0, cache
        )


def streaming_moe_replay(args: Any, models: Any, partition: str) -> Callable[[Any, dict], None]:
    """Compatibility entry point for callers that historically requested R3
    implicitly."""
    kinds = rollout_kinds(args)
    if not kinds and not hasattr(args, "use_rollout_routing_replay"):
        kinds = {"moe"}
    return replay_data_manager.register_consumer(kinds).streaming_binding(args, models, partition)


def fill_full_moe_replay(actor: Any, data_iterator: list, num_microbatches: list[int], batch: dict) -> None:
    if "rollout_routed_experts" not in batch:
        raise ValueError("rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set.")
    routers = list(local_moe_routers(actor.model, actor.args.num_layers))
    keys = ["rollout_routed_experts", "tokens"]
    if "multimodal_train_inputs" in batch:
        keys.append("multimodal_train_inputs")
    for iterator in data_iterator:
        iterator.reset()
    try:
        for _ in range(sum(num_microbatches)):
            microbatch = data_iterator[0].get_next(keys)
            bind_full_moe_batch(actor.args, microbatch, routers, microbatch["tokens"])
        del batch["rollout_routed_experts"]
    finally:
        for iterator in data_iterator:
            iterator.reset()


def fill_reference_replay(
    actor: Any,
    data_iterator: list,
    num_microbatches: list[int],
    kind: str,
    readers: ReplayManager | ReplayRegion,
    cache: ConsumerTensorCache,
) -> None:
    if kind != "moe":
        raise ValueError("The built-in selection consumer handles MoE routes only")
    layout = _moe_layout(actor.args, local_moe_routers(actor.model, actor.args.num_layers))
    device = device_utils.make_current_torch_device()
    for iterator in data_iterator:
        iterator.reset()
    try:
        # Hybrid accumulates several forward sub-batches before a single training replay.
        start = len(layout.bindings[0].channel.entries) if layout.bindings else 0
        for microbatch in range(start, start + sum(num_microbatches)):
            keys = [REFERENCE_FIELD, "tokens"]
            if "multimodal_train_inputs" in getattr(data_iterator[0], "rollout_data", {}):
                keys.append("multimodal_train_inputs")
            batch = data_iterator[0].get_next(keys)
            _bind_reference_batch(actor.args, batch, batch["tokens"], layout, readers, cache, device, microbatch)
    except BaseException as primary_error:
        try:
            RoutingReplay.clear_all()
        except Exception as cleanup_error:
            raise primary_error from cleanup_error
        raise
    finally:
        for iterator in data_iterator:
            iterator.reset()


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


class RoutingReplay:
    all_routing_replays = []
    phase = ConsumerPhase(environment=os.environ, key="ROUTING_REPLAY_STAGE")

    def __init__(self):
        self.channel = ConsumerChannel("moe")
        RoutingReplay.all_routing_replays.append(self)

    @property
    def top_indices_list(self) -> list[torch.Tensor]:
        return self.channel.entries

    @top_indices_list.setter
    def top_indices_list(self, entries: list[torch.Tensor]) -> None:
        self.channel.entries = entries
        self.channel.microbatch_ids = {}

    @property
    def forward_index(self) -> int:
        return self.channel.forward_index

    @forward_index.setter
    def forward_index(self, value: int) -> None:
        self.channel.forward_index = value

    @property
    def backward_index(self) -> int:
        return self.channel.backward_index

    @backward_index.setter
    def backward_index(self, value: int) -> None:
        self.channel.backward_index = value

    def record(self, top_indices: torch.Tensor) -> None:
        # Exactly the original placement/copy policy for CPU and CUDA inputs.
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.channel.bind(ConsumerRecord("moe", len(self.top_indices_list), buf))

    def pop_forward(self):
        values = self.channel.consume("replay_forward")
        return values.to(device=device_utils.make_current_torch_device(), dtype=torch.long)

    def pop_backward(self):
        values = self.channel.consume("replay_backward")
        return values.to(device=device_utils.make_current_torch_device(), dtype=torch.long)

    def clear(self):
        self.channel.clear()

    def clear_forward(self):
        self.channel.reset_forward()

    @classmethod
    def set_stage(cls, stage: str) -> None:
        cls.phase.set(stage)

    @classmethod
    def get_stage(cls) -> str:
        return cls.phase.get()

    @classmethod
    @contextmanager
    def forward_stage(cls) -> Iterator[None]:
        previous = cls.phase.enter_forward()
        try:
            yield
        finally:
            cls.phase.set(previous)

    @staticmethod
    def clear_all():
        error = None
        for replay in RoutingReplay.all_routing_replays:
            try:
                replay.clear()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()


def _align_top_indices(top_indices, scores_len):
    """Slice or pad top_indices along dim-0 to match scores_len.

    VLM bridge models with CP may use different sequence alignment than
    fill_routing_replay, causing a padding mismatch.  The extra (or missing)
    entries correspond to padding tokens that are zeroed out by the loss mask,
    so truncating or zero-padding is safe.
    """
    if top_indices.shape[0] == scores_len:
        return top_indices
    if top_indices.shape[0] > scores_len:
        return top_indices[:scores_len]
    pad_rows = scores_len - top_indices.shape[0]
    return torch.nn.functional.pad(top_indices, (0, 0, 0, pad_rows), value=0)


def get_routing_replay_compute_topk(old_compute_topk):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        # ROUTING_REPLAY is None for routers that opt out of replay (e.g. MTP),
        # in which case we fall through to the original implementation.
        if Envs.ENABLE_ROUTING_REPLAY and ROUTING_REPLAY is not None:
            routing_replay_stage = RoutingReplay.get_stage()
            if routing_replay_stage == "fallthrough":
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if routing_replay_stage == "record":
                probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                ROUTING_REPLAY.record(top_indices)
            elif routing_replay_stage == "replay_forward":
                top_indices = ROUTING_REPLAY.pop_forward()
                top_indices = _align_top_indices(top_indices, scores.shape[0])
                assert top_indices.shape[1] == topk, (
                    f"[{torch.distributed.get_rank()}] top_indices topk {top_indices.shape[1]} does not match expected topk {topk}"
                )
                probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_backward":
                top_indices = ROUTING_REPLAY.pop_backward()
                top_indices = _align_top_indices(top_indices, scores.shape[0])
                assert top_indices.shape[1] == topk, (
                    f"top_indices topk {top_indices.shape[1]} does not match expected topk {topk}"
                )
                probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    return compute_topk


def register_routing_replay(module):
    if not Envs.ENABLE_ROUTING_REPLAY:
        return

    # MTP routers exist in training but rollout (sglang) doesn't run MTP, so
    # there's nothing to record/replay against.  Install a pre-hook that clears
    # the global ROUTING_REPLAY (compute_topk falls through) and skip
    # registration in `all_routing_replays` so fill_routing_replay's per-layer
    # accounting stays consistent.
    if getattr(module, "is_mtp_layer", False):

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(None)

        module.register_forward_pre_hook(pre_forward_hook)
        return

    module.routing_replay = RoutingReplay()

    def pre_forward_hook(*args, **kwargs):
        set_routing_replay(module.routing_replay)

    module.register_forward_pre_hook(pre_forward_hook)


def reference_execution(function: Callable[P, R]) -> Callable[P, R]:
    """Release local leases on errors too; preserve inline exception
    semantics."""

    @wraps(function)
    def execute(*args: P.args, **kwargs: P.kwargs) -> R:
        if not references_enabled():
            return function(*args, **kwargs)
        primary_error = None
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            errors = []
            for replay in RoutingReplay.all_routing_replays:
                try:
                    replay.clear()
                except Exception as exc:
                    errors.append(exc)
            if args and getattr(getattr(args[0], "args", None), "use_rollout_mm_embedding_replay", False):
                from .mm_megatron import close_mm_replay

                try:
                    close_mm_replay()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                cleanup_error = errors[0]
                if hasattr(cleanup_error, "add_note"):
                    for other in errors[1:]:
                        cleanup_error.add_note(f"Additional replay cleanup failure: {other!r}")
                if primary_error is not None:
                    # Keep the training failure primary; expose cleanup failure
                    # as its cause, and retain failed sources for a later retry.
                    raise primary_error from cleanup_error
                raise cleanup_error

    return execute
