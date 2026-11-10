# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Megatron/TrainActor integration: receive, prepare and replay model
decisions."""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import torch

from relax.utils import device as device_utils
from relax.utils.env import Envs

from ..config import references_enabled
from ..consumer import (
    Consumer,
    ConsumerBatchLayout,
    ConsumerChannelKey,
    ConsumerLayerBinding,
    ConsumerModelLayout,
    ConsumerReadSession,
    ConsumerRecord,
    ConsumerTensorCache,
)
from ..core import ConsumerConsumptionError, ConsumerPhase, ReplayManager, ReplayRegion, replay_data_manager
from ..ref_data import REFERENCE_FIELD, ReplayRef, TokenHandle


P = ParamSpec("P")
R = TypeVar("R")
ROUTING_REPLAY = None
_VALID_STAGES = {"fallthrough", "replay_forward", "replay_backward"}


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
    ) -> None:
        self.manager._ensure_open()
        if region is not None and region.manager is not self.manager:
            raise ValueError("Replay region belongs to another manager")
        # Actor retains full-tensor packing; share one cache across reference kinds.
        cache = ConsumerTensorCache(self.manager.config.materialization_bytes)
        if "moe" in self.kinds:
            actor.fill_routing_replay(
                data_iterator,
                num_microbatches,
                batch,
                delivery_manager=self.manager,
                region=region,
                cache=cache,
            )
        if "indexer" in self.kinds:
            actor.fill_indexer_replay(
                data_iterator,
                num_microbatches,
                batch,
                delivery_manager=self.manager,
                region=region,
                cache=cache,
            )


def pack_token_rows(
    args: Any,
    rows: list[torch.Tensor],
    lengths: list[int],
    pad: Callable[[torch.Tensor, int], torch.Tensor],
    *,
    sequence_parallel: bool,
) -> torch.Tensor:
    """The existing token layout, independent of what each row contains."""
    from megatron.core import mpu

    from relax.backends.megatron.cp_utils import maybe_padded_total_lengths, slice_with_cp

    tp = mpu.get_tensor_model_parallel_world_size()
    cp = mpu.get_context_parallel_world_size()
    padded = maybe_padded_total_lengths(lengths, args.qkv_format, getattr(args, "is_vl_model", False))
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
) -> ConsumerBatchLayout:
    if any(m.rows != length - 1 for m, length in zip(refs, lengths, strict=True)):
        raise ValueError("Replay reference row count differs from training tokens")
    rows = [torch.from_numpy(TokenHandle.expand(ref)) for ref in refs]

    def pad(value: torch.Tensor, count: int) -> torch.Tensor:
        # Each padding operation restarts its offset, exactly as the original
        # MoE padding formula does. Do not replace this with a global offset.
        padding = torch.from_numpy(TokenHandle.padding(count))
        return torch.cat((value, padding), dim=0)

    coordinates = pack_token_rows(args, rows, lengths, pad, sequence_parallel=sequence_parallel).numpy()
    return ConsumerBatchLayout.from_coordinates(TokenHandle.coordinates(coordinates, refs))


def _local_layer_ranges(actor: Any) -> Iterator[tuple[Any, range]]:
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    for vp_stage, model in enumerate(actor.model):
        config = model.module.config
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        yield config, range(offset, offset + get_num_layers_to_build(config, vp_stage=vp_stage))


def _moe_layout(actor: Any) -> ConsumerModelLayout:
    local_layers = []
    for config, layers in _local_layer_ranges(actor):
        frequency = config.moe_layer_freq
        local_layers.extend(
            layer
            for layer in layers
            if (layer % frequency == 0 if isinstance(frequency, int) else frequency[layer] != 0)
        )
    routers = RoutingReplay.all_routing_replays
    if len(local_layers) != len(routers):
        raise ValueError("MoE reference layout does not match registered routers")
    bindings = tuple(
        ConsumerLayerBinding(layer, router.manager, router.channel_key)
        for layer, router in zip(local_layers, routers, strict=True)
    )
    return ConsumerModelLayout("moe", actor.args.moe_router_topk, actor.args.num_layers, bindings)


def _indexer_layout(actor: Any) -> ConsumerModelLayout:
    local_layers = []
    topk = None
    capture_layers = None
    for config, layers in _local_layer_ranges(actor):
        ratios = getattr(config, "csa_compress_ratios", None)
        if ratios is None or not hasattr(config, "dsa_indexer_topk") or getattr(config, "dsa_indexer_loss_coeff", 0):
            raise RuntimeError("Indexer reference replay requires the supported DSV4 C4 configuration")
        c4_layers = [layer for layer, ratio in enumerate(ratios[: config.num_layers]) if ratio == 4]
        if not c4_layers or len(ratios) < config.num_layers:
            raise ValueError("Invalid C4 layer map")
        if topk is not None and (topk != config.dsa_indexer_topk or capture_layers != c4_layers):
            raise ValueError("Inconsistent Indexer capture layout")
        topk = config.dsa_indexer_topk
        capture_layers = c4_layers
        local_layers.extend((layer, c4_layers.index(layer)) for layer in layers if layer in c4_layers)
    if topk is None or capture_layers is None:
        raise ValueError("No Indexer model configuration")
    IndexerReplay.begin_step([layer + 1 for layer, _ in local_layers])
    for model in actor.model:
        model.module.config.relax_use_indexer_replay = True
    bindings = tuple(
        ConsumerLayerBinding(slot, IndexerReplay._layers[layer + 1].manager, IndexerReplay._layers[layer + 1].key)
        for layer, slot in local_layers
    )
    return ConsumerModelLayout("indexer", topk, len(capture_layers), bindings)


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


def fill_reference_replay(
    actor: Any,
    data_iterator: list,
    num_microbatches: list[int],
    *,
    kind: str,
    delivery_manager: ReplayManager | None = None,
    region: ReplayRegion | None = None,
    cache: ConsumerTensorCache | None = None,
) -> None:

    if kind not in {"moe", "indexer"}:
        raise ValueError(f"Unsupported replay content: {kind}")
    if cache is None:
        manager = region.manager if region is not None else (delivery_manager or replay_data_manager)
        cache = ConsumerTensorCache(manager.config.materialization_bytes)
    layout = _moe_layout(actor) if kind == "moe" else _indexer_layout(actor)
    device = device_utils.make_current_torch_device()
    for iterator in data_iterator:
        iterator.reset()
    try:
        # Hybrid prepares several sub-batches before training. Append exactly as
        # the full-tensor RoutingReplay.record path does; do not reset its cursors.
        first = layout.bindings[0] if layout.bindings else None
        offset = len(first.manager._channel(first.channel).entries) if first is not None else 0
        for microbatch in range(offset, offset + sum(num_microbatches)):
            batch = data_iterator[0].get_next([REFERENCE_FIELD, "tokens"])
            refs = [ReplayRef.from_dict(ref[kind]) for ref in batch[REFERENCE_FIELD]]
            if any(m.layers != layout.capture_layers or m.topk != layout.topk for m in refs):
                raise ValueError("Replay reference shape does not match model")
            batch_layout = consumer_batch_layout(
                actor.args,
                refs,
                [len(tokens) for tokens in batch["tokens"]],
                sequence_parallel=kind == "moe" and actor.args.sequence_parallel,
            )
            # All local layers share this microbatch's registered reader holds.
            # Exiting the builder leaves the bound sources in charge of them.
            readers = region if region is not None else (delivery_manager or replay_data_manager)
            with readers.reader(refs) as reader:
                _bind_microbatch(
                    layout,
                    reader,
                    batch_layout,
                    microbatch,
                    device,
                    actor.args.num_experts if kind == "moe" else 0,
                    cache,
                )
    except BaseException as primary_error:
        try:
            if kind == "moe":
                RoutingReplay.clear_all()
            else:
                IndexerReplay.clear()
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
        self.manager = ReplayManager(phase=self.phase)
        self.channel_key = ConsumerChannelKey("moe", 0, "local_registration")
        self._channel = self.manager.register_channel(self.channel_key)
        RoutingReplay.all_routing_replays.append(self)

    @property
    def top_indices_list(self) -> list[torch.Tensor]:
        return self._channel.entries

    @top_indices_list.setter
    def top_indices_list(self, entries: list[torch.Tensor]) -> None:
        self._channel.entries = entries
        self._channel.microbatch_ids = {}

    @property
    def forward_index(self) -> int:
        return self._channel.forward_index

    @forward_index.setter
    def forward_index(self, value: int) -> None:
        self._channel.forward_index = value

    @property
    def backward_index(self) -> int:
        return self._channel.backward_index

    @backward_index.setter
    def backward_index(self, value: int) -> None:
        self._channel.backward_index = value

    def record(self, top_indices: torch.Tensor) -> None:
        # Exactly the original placement/copy policy for CPU and CUDA inputs.
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.manager.bind(self.channel_key, ConsumerRecord("moe", len(self.top_indices_list), buf))

    def pop_forward(self):
        values = self.manager.consume(self.channel_key, "replay_forward")
        return values.to(device_utils.make_current_torch_device())

    def pop_backward(self):
        values = self.manager.consume(self.channel_key, "replay_backward")
        return values.to(device_utils.make_current_torch_device())

    def clear(self):
        self.manager.clear()

    def clear_forward(self):
        self.manager.reset_forward(self.channel_key)

    @classmethod
    def set_stage(cls, stage: str) -> None:
        cls.phase.set(stage)

    @classmethod
    def get_stage(cls) -> str:
        return cls.phase.get()

    @classmethod
    @contextmanager
    def forward_stage(cls, *, enabled: bool = True) -> Iterator[None]:
        if not enabled:
            yield
            return
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


class _LayerIndexerReplay:
    def __init__(self, layer_number: int, phase: ConsumerPhase) -> None:
        self.manager = ReplayManager(phase=phase)
        self.key = ConsumerChannelKey("indexer", layer_number - 1)
        self._channel = self.manager.register_channel(self.key)

    @property
    def entries(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return self._channel.entries

    @property
    def forward_index(self) -> int:
        return self._channel.forward_index

    @forward_index.setter
    def forward_index(self, value: int) -> None:
        self._channel.forward_index = value

    @property
    def backward_index(self) -> int:
        return self._channel.backward_index

    @backward_index.setter
    def backward_index(self, value: int) -> None:
        self._channel.backward_index = value

    def record(self, topk: torch.Tensor, row_valid: torch.Tensor) -> None:
        if topk.ndim != 2 or topk.dtype != torch.int32:
            raise ValueError(f"indexer top-k must be 2-D int32, got shape={tuple(topk.shape)}, dtype={topk.dtype}")
        if row_valid.ndim != 1 or row_valid.dtype != torch.bool or row_valid.shape[0] != topk.shape[0]:
            raise ValueError(
                "indexer replay row-valid mask must be 1-D bool with the same row count as top-k, "
                f"got topk={tuple(topk.shape)}, row_valid={tuple(row_valid.shape)}/{row_valid.dtype}"
            )
        self.manager.bind(self.key, ConsumerRecord("indexer", len(self.entries), (topk, row_valid)))

    def pop(self, stage: str) -> tuple[torch.Tensor, torch.Tensor]:
        if stage not in ("replay_forward", "replay_backward"):
            raise ValueError(f"Cannot pop indexer replay in stage {stage!r}")
        index = self.forward_index if stage == "replay_forward" else self.backward_index
        try:
            return self.manager.consume(self.key, stage, advance_before_lookup=True)
        except IndexError:
            if index < len(self.entries):
                raise
            raise RuntimeError(
                f"Indexer replay {stage} consumed more microbatches than recorded: index={index}, "
                f"recorded={len(self.entries)}"
            ) from None


class IndexerReplay:
    """Step-local DSV4 C4 indexer choices, keyed by global one-based layer
    number."""

    _enabled = False
    _phase = ConsumerPhase()
    _layers: dict[int, _LayerIndexerReplay] = {}

    @classmethod
    def begin_step(cls, layer_numbers: list[int]) -> None:
        cls._release_sources()
        cls._enabled = True
        cls._phase.set("fallthrough")
        cls._layers = {number: _LayerIndexerReplay(number, cls._phase) for number in dict.fromkeys(layer_numbers)}

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def get_stage(cls) -> str:
        return cls._phase.get()

    @classmethod
    def set_stage(cls, stage: str) -> None:
        if stage not in _VALID_STAGES:
            raise ValueError(f"Invalid indexer replay stage {stage!r}; expected one of {sorted(_VALID_STAGES)}")
        cls._phase.set(stage)

    @classmethod
    @contextmanager
    def forward_stage(cls) -> Iterator[None]:
        """Use the forward cursor during a train forward, then restore backward
        replay."""
        old_stage = cls._phase.get()
        switch_stage = cls._enabled and old_stage == "replay_backward"
        if switch_stage:
            cls._phase.enter_forward()
        try:
            yield
        finally:
            if switch_stage:
                cls._phase.set(old_stage)

    @classmethod
    def record(
        cls,
        layer_number: int,
        topk: torch.Tensor,
        row_valid: torch.Tensor,
    ) -> None:
        try:
            replay = cls._layers[layer_number]
        except KeyError as exc:
            raise RuntimeError(f"Unexpected DSV4 indexer replay layer {layer_number}") from exc
        replay.record(topk, row_valid)

    @classmethod
    def replay(cls, layer_number: int, computed_topk: torch.Tensor | None) -> torch.Tensor | None:
        if not cls._enabled or cls._phase.get() == "fallthrough":
            return computed_topk
        try:
            replay = cls._layers[layer_number]
        except KeyError as exc:
            raise RuntimeError(f"Missing rollout indexer replay for DSV4 layer {layer_number}") from exc

        rollout_topk, row_valid = replay.pop(cls._phase.get())
        if computed_topk is None:
            return None
        if rollout_topk.shape != computed_topk.shape:
            raise RuntimeError(
                f"DSV4 indexer replay shape mismatch at layer {layer_number}: "
                f"rollout={tuple(rollout_topk.shape)}, trainer={tuple(computed_topk.shape)}"
            )
        if rollout_topk.device != computed_topk.device or row_valid.device != computed_topk.device:
            raise RuntimeError(
                f"DSV4 indexer replay device mismatch at layer {layer_number}: "
                f"rollout={rollout_topk.device}, mask={row_valid.device}, trainer={computed_topk.device}"
            )
        return torch.where(row_valid.unsqueeze(1), rollout_topk, computed_topk)

    @classmethod
    def reset_forward(cls, require_consumed: bool = True) -> None:
        for layer_number, replay in cls._layers.items():
            try:
                replay.manager.reset_forward(replay.key, require_consumed=require_consumed)
            except ConsumerConsumptionError as exc:
                raise RuntimeError(
                    f"DSV4 layer {layer_number} consumed {exc.consumed}/{exc.recorded} "
                    "forward indexer replay microbatches"
                ) from None

    @classmethod
    def finish_step(cls, expect_backward: bool) -> None:
        for layer_number, replay in cls._layers.items():
            try:
                replay.manager.check_consumed(replay.key, expect_backward=expect_backward)
            except ConsumerConsumptionError as exc:
                phase = "training-forward" if exc.phase == "replay_forward" else "recompute-backward"
                raise RuntimeError(
                    f"DSV4 layer {layer_number} consumed {exc.consumed}/{exc.recorded} "
                    f"{phase} indexer replay microbatches"
                ) from None
        cls.clear()

    @classmethod
    def clear(cls) -> None:
        cls._release_sources()
        cls._enabled = False
        cls._phase.set("fallthrough")
        cls._layers = {}

    @classmethod
    def _release_sources(cls) -> None:
        error = None
        for layer in cls._layers.values():
            try:
                layer.manager.release_sources()
            except Exception as exc:
                error = error or exc
        if error is not None:
            raise error


def maybe_replay_indexer_topk(
    layer_number: int,
    computed_topk: torch.Tensor | None,
) -> torch.Tensor | None:
    return IndexerReplay.replay(layer_number, computed_topk)


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
            try:
                IndexerReplay.clear()
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
