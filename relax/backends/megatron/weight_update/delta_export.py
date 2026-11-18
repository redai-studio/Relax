# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Backend-owned acknowledged shard snapshots and lossless HF delta export.

The existing Bridge/raw iterators remain responsible for naming and geometry.
Converting both acknowledged and pending shards also handles numerical
converters without NaN probes or a resident complete HF GPU snapshot. Transport
receives only final HF coordinates and does not interpret Megatron layouts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from copy import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from relax.distributed.weight_sync.config import RETAINED_DIRECTORY_BYTES, DeltaSyncConfig
from relax.utils.logging_utils import get_logger


logger = get_logger(__name__)


def delta_sync_config(args: Any, quantization_config: Any) -> DeltaSyncConfig | None:
    path = getattr(args, "memory_weight_sync_config", None)
    if not path:
        return None
    config = DeltaSyncConfig.load(path)
    # These change the actual exported/loaded state contract. Keep their
    # existing FULL implementation rather than banning an execution mode.
    incompatible = bool(
        quantization_config
        or getattr(args, "use_lora", False)
        or getattr(args, "lora_rank", 0)
        or getattr(args, "sglang_speculative_algorithm", None)
        or getattr(args, "mtp_num_layers", 0)
        or getattr(args, "rollout_external", False)
        or getattr(args, "sglang_pp_size", 1) != 1
        or getattr(args, "sglang_dp_size", 1) != 1
        or getattr(args, "rollout_num_gpus_per_engine", 1) > getattr(args, "num_gpus_per_node", 8)
    )
    if incompatible:
        logger.info(
            "Delta export retains original FULL for quantized, adapter, draft or external/inference-PP/DP state"
        )
        return None
    return config


def create_delta_exporter(args: Any, model: Any, *, model_name: str, quantization_config: Any) -> Any:
    config = delta_sync_config(args, quantization_config)
    if config is None:
        return None
    # Rank zero may concurrently materialize a CPU tensor and immutable TCP
    # bytes, including the same-source-GPU fallback of an NCCL connection.
    # Leave bounded control metadata room as well; do not spend this same host
    # budget in full independently in snapshots and delivery.
    snapshots = config.host_budget_bytes - 2 * config.staging_budget_bytes - (16 << 20)
    if config.receive_overlap:
        snapshots -= RETAINED_DIRECTORY_BYTES
    if snapshots <= 0:
        raise ValueError("Delta host budget must cover snapshots, TCP staging copies and bounded control metadata")
    return ShardDeltaExporter.create(
        args,
        model,
        model_name=model_name,
        quantization_config=quantization_config,
        snapshot_budget_bytes=snapshots,
        config=config,
    )


@dataclass(frozen=True)
class DeltaWeight:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    positions: torch.Tensor | None
    values: torch.Tensor

    @property
    def full(self) -> bool:
        return self.positions is None


class ShardDeltaExporter:
    """One pending snapshot, committed only after all recipients finish.

    Two existing iterator instances keep mutable conversion state independent
    (e.g. multi-input alpha buffers and PLE generator process-group scopes).
    Their temporary buckets are live concurrently; the caller must account for
    their actual allocations rather than assume a single decode buffer.
    """

    def __init__(
        self,
        current_iterator: Any,
        baseline_iterator: Any,
        *,
        snapshot_budget_bytes: int,
        config: DeltaSyncConfig | None = None,
    ) -> None:
        if current_iterator is baseline_iterator:
            raise ValueError("Current and baseline conversions require independent iterator state")
        if type(snapshot_budget_bytes) is not int or snapshot_budget_bytes <= 0:
            raise ValueError("The local shard snapshot budget must be a positive integer")
        self.current_iterator = current_iterator
        self.baseline_iterator = baseline_iterator
        self.snapshot_budget_bytes = snapshot_budget_bytes
        self._acknowledged: dict[str, torch.Tensor] = {}
        self._pending: dict[str, torch.Tensor] | None = None
        self._exported = False
        self._force_full = True
        self._attempt: object | None = None
        self._stream: Iterator | None = None
        self.config = config
        self.transfer: Any = None
        self.full_names: set[str] = set()
        self._version: str | None = None
        self._sequence = 0
        self._pending_version: str | None = None
        self._seed_after_commit = False
        self.failed = False

    @classmethod
    def create(cls, args: Any, model: Any, *, model_name: str, quantization_config: Any, **kwargs: Any) -> Any:
        from .hf_weight_iterator_base import HfWeightIteratorBase

        # Both constructions may contain collectives: every source rank enters
        # them in the same order, including ranks that never send rollout data.
        config = kwargs.get("config")
        if config is not None:
            args = copy(args)
            args.update_weight_buffer_size = min(args.update_weight_buffer_size, config.bucket_bytes)
        current, baseline = [
            HfWeightIteratorBase.create(args, model, model_name=model_name, quantization_config=quantization_config)
            for _ in range(2)
        ]
        return cls(current, baseline, **kwargs)

    @staticmethod
    def supports_recipients(engines: list, gpu_counts: list[int], *, group: dist.ProcessGroup) -> bool:
        """Query actual SG topology before the original pause boundary."""
        import ray

        result = [None]
        if dist.get_rank(group=group) == 0:
            try:
                layouts = ray.get([engine.get_delta_weight_sync_layout.remote() for engine in engines], timeout=60)
                result[0] = (
                    all(
                        layout["nnodes"] == 1
                        and layout["pp_size"] == 1
                        and layout["dp_size"] == 1
                        and layout["tp_size"] == count
                        for layout, count in zip(layouts, gpu_counts, strict=True)
                    ),
                    None,
                )
            except Exception as exc:
                result[0] = (False, str(exc))
        dist.broadcast_object_list(result, src=dist.get_process_group_ranks(group)[0], group=group)
        supported, error = result[0]
        if error is not None:
            raise RuntimeError(f"Cannot query actual delta recipient topology: {error}")
        return supported

    def synchronize(
        self,
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        engines: list | None,
        gpu_counts: list[int] | None,
        version: str,
        group: dist.ProcessGroup,
        force_full: bool = False,
        before_first_apply: Callable[[], None] | None = None,
    ) -> dict:
        """Export/deliver under the original caller's update lock.

        Every training rank enters conversion. Only source group rank zero
        sends the global HF stream. In particular, DCS's actor_fwd/reference
        raw path remains in its existing owner and does not pass through this
        code. The caller commits only after its remaining lifecycle gates
        succeed. An optional owner barrier defers its pause until bounded
        immutable reception finishes; only that owner knows how to pause.
        """
        from relax.distributed.weight_sync.codec import encode
        from relax.distributed.weight_sync.transport import DeltaTransfer

        if self.config is None or self.failed:
            raise RuntimeError("Delta synchronization requires an active configured backend exporter")
        rank = dist.get_rank(group=group)
        world = dist.get_world_size(group=group)
        source_rank = dist.get_process_group_ranks(group)[0]
        allocation_base = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocation_base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        def workspace_peak() -> int:
            return 0 if allocation_base is None else max(0, torch.cuda.max_memory_allocated() - allocation_base)

        def agree(error: Exception | None) -> None:
            errors = [None] * world
            dist.all_gather_object(errors, None if error is None else f"{type(error).__name__}: {error}", group=group)
            if any(errors):
                raise RuntimeError(f"Delta weight synchronization failed across source ranks: {errors}")

        try:
            if before_first_apply is not None:
                # Seed/reconnect may allocate a communicator or a fresh arena.
                # Keep those allocations inside the original pause, rather
                # than admitting them against a transient generation trough.
                early_pause, error = [None], None
                if rank == 0:
                    try:
                        early_pause[0] = bool(
                            not self.config.receive_overlap
                            or force_full
                            or self._force_full
                            or self.config.full_only
                            or self._sequence % self.config.anchor_interval == 0
                            or self.transfer is None
                            or not self.transfer.can_reuse(engines, gpu_counts)
                        )
                    except Exception as exc:
                        error = exc
                agree(error)
                dist.broadcast_object_list(early_pause, src=source_rank, group=group)
                if early_pause[0]:
                    before_first_apply()
                    before_first_apply = None
            status = [None]
            error = None
            if rank == 0:
                try:
                    if self.transfer is None:
                        from relax.utils.device import make_current_torch_device

                        self.transfer = DeltaTransfer(self.config, make_current_torch_device())
                    changed = self.transfer.connect(engines, gpu_counts, reconnect=force_full)
                    status[0] = bool(
                        changed
                        or force_full
                        or self._force_full
                        or self.config.full_only
                        or self._sequence % self.config.anchor_interval == 0
                    )
                except Exception as exc:
                    error = exc
            agree(error)
            dist.broadcast_object_list(status, src=source_rank, group=group)
            seed = status[0]
            applying = before_first_apply is None
            overlap = bool(self.config.receive_overlap and not seed and not force_full and not applying)
            constraints = set()

            def enter_apply() -> None:
                nonlocal applying
                if applying:
                    return
                # Every source rank enters the owner's same original pause,
                # flush and session barrier exactly once. No converter is
                # suspended across the separate DCS raw-weight iteration.
                before_first_apply()
                error = None
                if rank == 0:
                    try:
                        self.transfer.activate()
                        constraints.update(self.transfer.apply())
                    except Exception as exc:
                        error = exc
                agree(error)
                applying = True

            error = None
            try:
                if seed:
                    self.invalidate()
                self.begin(weights_getter())
            except Exception as exc:
                error = exc
            agree(error)
            error = None
            if rank == 0:
                try:
                    self.transfer.begin(version=version, base=self._version, seed=seed, defer_apply=not applying)
                except Exception as exc:
                    error = exc
            agree(error)
            if not overlap:
                enter_apply()

            stream = self.export(full=seed, full_names=frozenset(self.full_names), emit=rank == 0)
            while True:
                error, chunk = None, None
                try:
                    chunk = next(stream, None)
                    if workspace_peak() > self.config.gpu_budget_bytes:
                        raise ValueError("Measured source conversion workspace exceeds its configured GPU budget")
                except Exception as exc:
                    error = exc
                agree(error)
                finished = [None] * world
                dist.all_gather_object(finished, chunk is None, group=group)
                if any(finished):
                    if not all(finished):
                        raise ValueError("Source converters disagree on the complete HF group directory")
                    break
                error = None
                if rank == 0:
                    try:
                        metadata, raw = encode(
                            ((item.name, item.shape, item.dtype, item.positions, item.values) for item in chunk),
                            max_bytes=self.config.staging_budget_bytes,
                        )
                        if workspace_peak() > self.config.gpu_budget_bytes:
                            raise ValueError("Measured source conversion/encoding workspace exceeds its GPU budget")
                    except Exception as exc:
                        error = exc
                agree(error)
                if not applying:
                    admission, error = [None], None
                    if rank == 0:
                        try:
                            admission[0] = self.transfer.can_stage(metadata, raw)
                        except Exception as exc:
                            error = exc
                    agree(error)
                    dist.broadcast_object_list(admission, src=source_rank, group=group)
                    if not admission[0]:
                        enter_apply()
                error = None
                if rank == 0:
                    try:
                        self.transfer.stage(metadata, raw)
                        if applying:
                            constraints.update(self.transfer.apply())
                    except Exception as exc:
                        error = exc
                agree(error)
                del chunk
            enter_apply()
            source_peaks = [None] * world
            dist.all_gather_object(source_peaks, workspace_peak(), group=group)
            status = [None]
            error = None
            if rank == 0:
                try:
                    result = self.transfer.finalize()
                    status[0] = dict(
                        result,
                        full=seed,
                        version=version,
                        full_names=sorted(constraints),
                        force_full_seed=any(reply["force_full_seed"] for reply in result["receivers"]),
                        source_torch_workspace_peak_bytes=source_peaks,
                        receive_overlap_requested=self.config.receive_overlap,
                        receive_overlap_eligible=overlap,
                    )
                except Exception as exc:
                    error = exc
            agree(error)
            dist.broadcast_object_list(status, src=source_rank, group=group)
            self.full_names.update(status[0]["full_names"])
            self._pending_version = version
            self._seed_after_commit = status[0]["force_full_seed"]
            if rank == 0:
                logger.info("Delta weight data delivery complete (pending lifecycle commit): %s", status[0])
            return status[0]
        except BaseException:
            self.failed = True
            self.abort()
            raise

    def invalidate(self) -> None:
        """Recipient recreation or weight residency loss requires a FULL
        seed."""
        if self._pending is not None:
            raise RuntimeError("Cannot invalidate an in-flight delta export")
        self._force_full = True

    def disconnect(self, *, group: dist.ProcessGroup) -> None:
        """Join the original trainer sleep boundary after a completed
        update."""
        if self._pending is not None:
            raise RuntimeError("Cannot disconnect an unacknowledged delta update")
        error = [None]
        if dist.get_rank(group=group) == 0:
            try:
                if self.transfer is not None:
                    self.transfer.close()
                    self.transfer = None
            except Exception as exc:
                error[0] = str(exc)
        dist.broadcast_object_list(error, src=dist.get_process_group_ranks(group)[0], group=group)
        if error[0] is not None:
            raise RuntimeError(f"Delta disconnect failed: {error[0]}")
        self.invalidate()

    def begin(self, local_shards: Mapping[str, torch.Tensor]) -> None:
        if self._pending is not None:
            raise RuntimeError("A delta snapshot is already pending acknowledgement")
        required = sum(t.numel() * t.element_size() for t in local_shards.values())
        required += sum(t.numel() * t.element_size() for t in self._acknowledged.values())
        if required > self.snapshot_budget_bytes:
            raise ValueError("Acknowledged and pending local shard snapshots exceed their host budget")
        # Blocking copies establish a stable target before a converter or the
        # optimizer can reuse the source. No pending tensor aliases live state.
        self._pending = {name: tensor.detach().to(device="cpu", copy=True) for name, tensor in local_shards.items()}
        self._attempt = object()
        self._stream = None
        self._exported = False
        if self._acknowledged and self._schema(self._pending) != self._schema(self._acknowledged):
            self._force_full = True

    @staticmethod
    def _schema(shards: Mapping[str, torch.Tensor]) -> tuple:
        return tuple((name, tuple(value.shape), value.dtype) for name, value in shards.items())

    def export(
        self, *, full: bool = False, full_names: frozenset[str] = frozenset(), emit: bool = True
    ) -> Iterator[list[DeltaWeight]]:
        if self._pending is None or self._exported or self._stream is not None:
            raise RuntimeError("Export requires a fresh pending snapshot")
        self._stream = self._export_pending(self._pending, self._attempt, full=full, full_names=full_names, emit=emit)
        return self._stream

    @torch.no_grad()
    def _export_pending(
        self,
        pending: Mapping[str, torch.Tensor],
        attempt: object,
        *,
        full: bool,
        full_names: frozenset[str],
        emit: bool,
    ) -> Iterator[list[DeltaWeight]]:
        if self._attempt is not attempt:
            raise RuntimeError("The delta export attempt is no longer active")
        current = iter(self.current_iterator.get_hf_weight_chunks(pending))
        full = full or self._force_full or not self._acknowledged
        baseline = iter(()) if full else iter(self.baseline_iterator.get_hf_weight_chunks(self._acknowledged))
        try:
            while True:
                # Check before advancing either collective-bearing iterator.
                if self._attempt is not attempt:
                    raise RuntimeError("The delta export attempt is no longer active")
                current_chunk = next(current, None)
                baseline_chunk = next(baseline, None)
                if current_chunk is None and baseline_chunk is None:
                    break
                if current_chunk is None or (not full and baseline_chunk is None):
                    raise ValueError("Current and baseline HF conversion directories differ")
                if not full and len(current_chunk) != len(baseline_chunk):
                    raise ValueError("Current and baseline HF conversion buckets differ")
                yield self._encode_chunk(current_chunk, baseline_chunk, full=full, full_names=full_names, emit=emit)
            if self._attempt is not attempt:
                raise RuntimeError("A retired export cannot complete a new pending snapshot")
            self._exported = True
        finally:
            # In particular, restore converters' cross-yield process-group scopes.
            for iterator in (current, baseline):
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()

    @staticmethod
    def _encode_chunk(
        current: list, baseline: list | None, *, full: bool, full_names: frozenset[str], emit: bool
    ) -> list[DeltaWeight]:
        entries = [None] * len(current)
        sparse = {}
        for index, (name, tensor) in enumerate(current):
            if not full:
                old_name, old = baseline[index]
                if old_name != name or old.shape != tensor.shape or old.dtype != tensor.dtype:
                    raise ValueError("Current and baseline HF tensor schemas differ")
            if not emit:
                continue
            tensor = tensor.detach().contiguous()
            sparse_capable = tensor.dtype in (torch.bfloat16, torch.float32) and tensor.numel() < 2**31
            if full or name in full_names or not sparse_capable:
                entries[index] = DeltaWeight(name, tuple(tensor.shape), tensor.dtype, None, tensor.flatten().clone())
            else:
                sparse.setdefault(tensor.dtype, []).append((index, name, tensor, old.detach().contiguous()))
        if not emit:
            return []
        # Dynamic nonzero and small tensor-boundary metadata synchronize once
        # per dtype bucket, not once for every expert projection. Only one
        # elected sender encodes the globally exported HF values.
        for dtype, group in sparse.items():
            words = torch.int16 if dtype == torch.bfloat16 else torch.int32
            current_words = torch.cat([tensor.view(words).flatten() for _, _, tensor, _ in group])
            baseline_words = torch.cat([old.view(words).flatten() for _, _, _, old in group])
            positions = (current_words != baseline_words).nonzero().flatten()
            values = current_words.index_select(0, positions).view(dtype)
            offsets = [0]
            for _, _, tensor, _ in group:
                offsets.append(offsets[-1] + tensor.numel())
            boundaries = torch.tensor(offsets, dtype=torch.int64, device=positions.device)
            cuts = torch.searchsorted(positions, boundaries).cpu().tolist()
            for slot, (index, name, tensor, _old) in enumerate(group):
                left, right = cuts[slot : slot + 2]
                if (right - left) * (4 + tensor.element_size()) >= tensor.numel() * tensor.element_size():
                    entries[index] = DeltaWeight(name, tuple(tensor.shape), dtype, None, tensor.flatten().clone())
                else:
                    local_positions = (positions[left:right] - offsets[slot]).to(torch.int32)
                    entries[index] = DeltaWeight(name, tuple(tensor.shape), dtype, local_positions, values[left:right])
        return entries

    def commit(self) -> None:
        """Called once after the final common receiver completion gate."""
        if self._pending is None or not self._exported:
            raise RuntimeError("Cannot commit an incomplete or absent delta export")
        self._acknowledged = self._pending
        self._pending = None
        self._exported = False
        self._force_full = self._seed_after_commit
        self._attempt = None
        self._stream = None
        if self._pending_version is not None:
            self._version = self._pending_version
            self._pending_version = None
            self._sequence += 1
        self._seed_after_commit = False

    def abort(self) -> None:
        """Discard the failed target; keep the last acknowledged baseline."""
        self._attempt = None
        if self._stream is not None:
            self._stream.close()
        self._stream = None
        self._pending = None
        self._exported = False
        self._force_full = True
        self._pending_version = None
        self._seed_after_commit = False
