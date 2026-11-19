# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Frozen Qwen3.8 vision outputs consumed through reference replay."""

from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist

from ..ref_data import ReplayRef, validate_mm_envelope


_active_reader: ContextVar[Any] = ContextVar("mm_embedding_replay_reader", default=None)
_vision_input: ContextVar[Any] = ContextVar("mm_embedding_replay_input", default=None)
_INPUT_KEY = "_relax_mm_embedding_replay"


@dataclass(frozen=True)
class VisionReplayInput:
    embeddings: torch.Tensor
    media: tuple[dict[str, Any], ...]


def _model_module(model: Any) -> Any:
    while hasattr(model, "module"):
        model = model.module
    return model


def install_vision_replay(model: Any) -> None:
    """Install instance-local vision bypass; unmarked calls keep the original
    path."""
    if getattr(model, "_relax_mm_replay_installed", False):
        return
    vision = getattr(model, "vision_model", None)
    if vision is None:
        raise ValueError("MM replay requires a vision module on the input pipeline stage")
    if any(parameter.requires_grad for parameter in vision.parameters()):
        raise ValueError("MM replay requires the entire VE and merger/projection to be frozen")
    config = getattr(model, "vision_transformer_config", None)
    if config is None or getattr(config, "deepstack_visual_indexes", None):
        raise ValueError("Qwen3.8 MM replay requires a vision config without DeepStack outputs")
    if model.config.hidden_size != 2560 or model.config.context_parallel_size != 1:
        raise ValueError("Qwen3.8 MM replay requires hidden size 2560 and CP1")

    import importlib

    implementation = importlib.import_module(type(model).__module__)
    if not hasattr(implementation, "reorganize_inputs"):
        raise RuntimeError("Installed Qwen3.8 Bridge lacks the visual input-layout integration point")
    original_layout = implementation.reorganize_inputs
    if not getattr(original_layout, "_relax_mm_replay", False):

        @wraps(original_layout)
        def reorganize(input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            replay = _vision_input.get()
            if replay is None:
                return original_layout(input_ids, *args, **kwargs)
            return replay_layout(input_ids, replay, model.image_token_id, model.video_token_id)

        reorganize._relax_mm_replay = True
        implementation.reorganize_inputs = reorganize

    original_vision = vision.forward

    @wraps(original_vision)
    def vision_forward(_vision: Any, *args: Any, **kwargs: Any) -> Any:
        replay = _vision_input.get()
        if replay is None:
            return original_vision(*args, **kwargs)
        return replay.embeddings, []

    vision.forward = MethodType(vision_forward, vision)
    original_forward = model.forward

    @wraps(original_forward)
    def forward(_model: Any, *args: Any, **kwargs: Any) -> Any:
        replay = kwargs.pop(_INPUT_KEY, None)
        has_media = kwargs.get("image_grid_thw") is not None or kwargs.get("video_grid_thw") is not None
        if replay is None and has_media:
            raise RuntimeError("MM replay is enabled but this visual batch has no replay embeddings")
        token = _vision_input.set(replay)
        try:
            return original_forward(*args, **kwargs)
        finally:
            _vision_input.reset(token)

    model.forward = MethodType(forward, model)
    model._relax_mm_replay_installed = True


def replay_layout(
    input_ids: torch.Tensor, replay: VisionReplayInput, image_token_id: int, video_token_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the original visual geometry without transporting pixel tensors.

    The zero-width carrier holds patch row counts only; the intercepted VE
    consumes the replay tensor instead. No fake pixels reach an encoder.
    """
    grids = [entry["grid"] for entry in replay.media]
    patches = sum(t * h * w for t, h, w in grids)
    rows = sum(t * h * w // 4 for t, h, w in grids)
    if replay.embeddings.shape != (rows, 2560) or replay.embeddings.dtype != torch.bfloat16:
        raise ValueError("Replay embedding shape/dtype does not match the visual grid")
    device = replay.embeddings.device
    host_grid = torch.tensor(grids, dtype=torch.long, pin_memory=device.type == "cuda")
    grid = host_grid.to(device, non_blocking=True)
    mask = (input_ids == image_token_id) | (input_ids == video_token_id)
    return torch.empty((patches, 0), dtype=torch.bfloat16, device=device), grid, mask


class MMReplayReader:
    """One update's bounded immutable embeddings and explicit TQ reader
    leases."""

    def __init__(self, manager: Any, batch: dict[str, Any], region: Any, model: Any) -> None:
        self.store = manager.store()
        self.capacity_bytes = manager.config.materialization_bytes
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.cache_bytes = 0
        self.metrics = dict.fromkeys(
            (
                "logical_bytes",
                "unique_bytes",
                "read_bytes",
                "cache_hits",
                "cache_misses",
                "cache_evictions",
                "cache_peak_bytes",
                "read_cpu_seconds",
                "tp_broadcast_cpu_seconds",
            ),
            0,
        )
        self.refs: list[ReplayRef] = []
        self.leases: list[str] = []
        self.enabled = model is not None
        self.tp = None if model is None else model.pg_collection.tp
        if self.enabled and self.tp is None:
            raise ValueError("MM replay requires an explicit tensor-parallel process group")
        self.tp_rank = dist.get_rank(self.tp) if self.enabled and dist.is_initialized() else 0
        self.tp_size = dist.get_world_size(self.tp) if self.enabled and dist.is_initialized() else 1
        self.tp_source = dist.get_global_rank(self.tp, 0) if self.tp_size > 1 else None
        self.tp_control = self.tp
        self.device = torch.device("cuda", torch.cuda.current_device()) if self.enabled else torch.device("cpu")
        if not self.enabled:
            return
        install_vision_replay(model)
        seen = {}
        for manifests in batch["rollout_replay_refs"]:
            envelope = manifests.get("mm_embedding")
            validate_mm_envelope(envelope)
            for entry in envelope["media"]:
                ref = ReplayRef.from_dict(entry["ref"])
                self.metrics["logical_bytes"] += ref.rows * 5120
                if ref.record in seen and seen[ref.record] != ref:
                    raise ValueError("Conflicting MM replay manifests for one body")
                seen[ref.record] = ref
        self.metrics["unique_bytes"] = sum(ref.rows * 5120 for ref in seen.values())
        if self.tp_rank == 0:
            self.refs = list(seen.values())
        if seen:
            if self.tp_size > 1 and dist.get_backend(self.tp) != "gloo":
                # Only the disjoint input-stage TP groups participate. Keep
                # the control group for the model's lifetime, across updates.
                control = getattr(model, "_relax_mm_replay_control_group", None)
                if control is None:
                    control = dist.new_group(
                        ranks=dist.get_process_group_ranks(self.tp), backend="gloo", use_local_synchronization=True
                    )
                    model._relax_mm_replay_control_group = control
                self.tp_control = control
            error = None
            if self.tp_rank == 0:
                try:
                    self.leases = self.store.acquire_many(self.refs, partition=region.partition)
                except Exception as exc:
                    error = exc
            try:
                self._check_source_error(error)
            except BaseException as primary:
                try:
                    self.close()
                except Exception as cleanup:
                    primary.add_note(f"MM replay lease release failed: {cleanup!r}")
                raise
        self._leases_by_record = {ref.record: lease for ref, lease in zip(self.refs, self.leases, strict=True)}

    def _check_source_error(self, error: Exception | None) -> None:
        """Check source I/O on CPU before entering any payload collective."""
        if self.tp_size > 1:
            failed = torch.tensor([error is not None], dtype=torch.uint8, device="cpu")
            dist.broadcast(failed, src=self.tp_source, group=self.tp_control)
            if failed[0]:
                details = [None if error is None else f"{type(error).__name__}: {error}"]
                dist.broadcast_object_list(details, src=self.tp_source, group=self.tp_control, device=failed.device)
                if error is None:
                    raise RuntimeError(f"MM replay source I/O failed: {details[0]}")
        if error is not None:
            raise error

    def load(self, media: list[dict[str, Any]]) -> torch.Tensor:
        from ..mm_embedding import read_mm_embedding

        outputs = []
        pending = []
        error = None
        for entry in media:
            ref = ReplayRef.from_dict(entry["ref"])
            tensor = self.cache.get(ref.record)
            if tensor is None:
                self.metrics["cache_misses"] += 1
                if self.tp_rank == 0:
                    started = perf_counter()
                    try:
                        tensor = read_mm_embedding(
                            self.store, entry, device=self.device, lease=self._leases_by_record[ref.record]
                        )
                    except Exception as exc:
                        error = exc
                        break
                    self.metrics["read_cpu_seconds"] += perf_counter() - started
                    self.metrics["read_bytes"] += ref.rows * 5120
                else:
                    tensor = torch.empty((ref.rows, 2560), device=self.device, dtype=torch.bfloat16)
                pending.append(tensor)
                size = tensor.numel() * tensor.element_size()
                if size <= self.capacity_bytes:
                    while self.cache and self.cache_bytes + size > self.capacity_bytes:
                        _, removed = self.cache.popitem(last=False)
                        self.metrics["cache_evictions"] += 1
                        self.cache_bytes -= removed.numel() * removed.element_size()
                    self.cache[ref.record] = tensor
                    self.cache_bytes += size
                    self.metrics["cache_peak_bytes"] = max(self.metrics["cache_peak_bytes"], self.cache_bytes)
            elif ref.record in self.cache:
                self.metrics["cache_hits"] += 1
                self.cache.move_to_end(ref.record)
            outputs.append(tensor)
        if pending or error is not None:
            # All ranks traverse the same media and cache order. Confirm
            # every cold read once per microbatch before sharing embeddings.
            try:
                self._check_source_error(error)
            except BaseException:
                self.cache.clear()
                self.cache_bytes = 0
                raise
            if self.tp_size > 1:
                started = perf_counter()
                for tensor in pending:
                    dist.broadcast(tensor, src=self.tp_source, group=self.tp)
                self.metrics["tp_broadcast_cpu_seconds"] += perf_counter() - started
        return torch.cat(outputs, dim=0)

    def close(self) -> None:
        self.cache.clear()
        self.cache_bytes = 0
        if self.leases:
            self.store.release_many(self.refs, self.leases)
            self.leases = []


def prepare_mm_replay(manager: Any, actor: Any, batch: dict[str, Any], region: Any) -> None:
    close_mm_replay()
    models = [_model_module(model) for model in actor.model]
    inputs = [model for model in models if getattr(model, "pre_process", False)]
    if len(inputs) > 1:
        raise ValueError("MM replay does not support multiple input pipeline chunks")
    reader = MMReplayReader(manager, batch, region, inputs[0] if inputs else None)
    _active_reader.set(reader)


def close_mm_replay() -> None:
    reader = _active_reader.get()
    if reader is not None:
        reader.close()
        _active_reader.set(None)


def bind_mm_replay_batch(batch: dict[str, Any]) -> None:
    reader = _active_reader.get()
    if reader is None:
        raise RuntimeError("MM replay batch has no active reference consumption region")
    if not reader.enabled:
        return
    manifests = batch.get("rollout_replay_refs")
    if manifests is None:
        raise ValueError("MM replay batch is missing reference metadata")
    media = []
    for refs in manifests:
        envelope = refs.get("mm_embedding")
        validate_mm_envelope(envelope)
        media.extend(envelope["media"])
    if not media:
        if batch.get("multimodal_train_inputs"):
            grids = batch["multimodal_train_inputs"]
            if any(key in grids and grids[key].shape[0] for key in ("image_grid_thw", "video_grid_thw")):
                raise ValueError("Visual inputs have no MM replay references")
        return
    if batch.get("multimodal_train_inputs") is None:
        raise ValueError("MM replay requires retained visual layout metadata")
    batch["multimodal_train_inputs"][_INPUT_KEY] = VisionReplayInput(reader.load(media), tuple(media))


def log_mm_replay_metrics(args: Any, rollout_id: int) -> None:
    """Report local counters; CPU durations include enqueue/waits, not GPU
    elapsed time."""
    reader = _active_reader.get()
    if reader is None or not reader.enabled:
        return
    from megatron.core import mpu

    from relax.utils import tracking_utils
    from relax.utils.logging_utils import get_logger
    from relax.utils.metrics.metric_utils import compute_rollout_step

    metrics = {f"replay/mm_{key}": value for key, value in reader.metrics.items()}
    get_logger(__name__).info("mm_replay_consumer %s tp=%s: %s", rollout_id, reader.tp_rank, metrics)
    if reader.tp_rank == 0 and mpu.get_data_parallel_rank(with_context_parallel=True) == 0:
        # Input pipeline stage owns VE replay; the usual tracking rank is on
        # the last stage, so initialize its lightweight service adapter here.
        if getattr(args, "use_metrics_service", False):
            from relax.utils.metrics.metrics_service_adapter import (
                get_metrics_service_adapter,
                init_metrics_service_adapter,
            )

            if get_metrics_service_adapter() is None:
                init_metrics_service_adapter(args)
            tracking_utils.log_direct(args, compute_rollout_step(args, rollout_id), metrics)
        # Without the shared metrics service, keep local logs instead of
        # creating another ClearML/W&B task on the input pipeline stage.
