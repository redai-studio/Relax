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

from relax.distributed.weight_sync.config import DeltaSyncConfig
from relax.utils.logging_utils import get_logger

from . import target_shard
from .expert_gather import OwnerExpert
from .shard_snapshot import CudaShardBackup, CudaShardInputs, shard_layout


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
    target_lane_counts: tuple[int, ...] | None = None

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

    # Private frozen-pair A/B binding, installed on every source rank.
    _owner_local_diff = False

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
        self._live_inputs: CudaShardInputs | None = None
        self._backup: CudaShardBackup | None = None
        self._live_read_done: torch.cuda.Event | None = None
        self._ack_bank: torch.Tensor | None = None
        self._spare_bank: torch.Tensor | None = None
        self._supports_live_shards = False
        self._exported = False
        self._force_full = True
        self._attempt: object | None = None
        self._stream: Iterator | None = None
        self.config = config
        self.transfer: Any = None
        self.full_names: set[str] = set()
        self._target_shards = False
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
        exporter = cls(current, baseline, **kwargs)
        exporter._supports_live_shards = (
            getattr(args, "fully_async", False)
            and not getattr(args, "hybrid", False)
            and args.megatron_to_hf_mode == "bridge"
        )
        return exporter

    @property
    def sender_pipeline_active(self) -> bool:
        return self._live_inputs is not None

    def _can_read_live(self, shards: Mapping[str, torch.Tensor]) -> bool:
        if not self._supports_live_shards or not shards:
            return False
        device = next(iter(shards.values())).device
        return (
            device.type == "cuda"
            and device.index == torch.cuda.current_device()
            and all(
                value.device == device and value.layout == torch.strided and not value.is_quantized
                for value in shards.values()
            )
        )

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
    ) -> dict:
        """Export/deliver under the original caller's update lock.

        Every training rank enters conversion. Only source group rank zero
        sends the global HF stream. In particular, DCS's actor_fwd/reference
        raw path remains in its existing owner and does not pass through this
        code. The caller commits only after its remaining lifecycle gates
        succeed.
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
            status = [None, False]
            error = None
            if rank == 0:
                try:
                    if self.transfer is None:
                        from relax.utils.device import make_current_torch_device

                        self.transfer = DeltaTransfer(self.config, make_current_torch_device())
                    changed = self.transfer.connect(engines, gpu_counts, reconnect=force_full)
                    if self.config.target_shard_delta:
                        status[1] = (
                            self.transfer.target_shard_capable and self.transfer.target_shard_version == self._version
                        )
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
            self._target_shards = status[1] is True and not seed
            constraints = set()

            def drain_previous_frame() -> None:
                # Called by every rank at the same point in the export. A TCP
                # consumer error must reach the whole group before we admit
                # another frame or publish the final version.
                error = None
                if rank == 0:
                    try:
                        constraints.update(self.transfer.drain_frame())
                    except Exception as exc:
                        error = exc
                agree(error)

            error = None
            try:
                if seed:
                    self.invalidate()
                shards = weights_getter()
                live = bool(getattr(self.config, "sender_pipeline", False)) and self._can_read_live(shards)
            except Exception as exc:
                error = exc
            if getattr(self.config, "sender_pipeline", False):
                agree(error)
                eligible = [None] * world
                dist.all_gather_object(eligible, live, group=group)
                live = all(eligible)
            if error is None:
                try:
                    self.begin(shards, live_locked=live)
                except Exception as exc:
                    error = exc
            agree(error)
            error = None
            if rank == 0:
                try:
                    self.transfer.begin(version=version, base=self._version, seed=seed)
                except Exception as exc:
                    error = exc
            agree(error)

            stream = self.export(full=seed, full_names=frozenset(self.full_names), emit=rank == 0, _owner_group=group)
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
                        if self._target_shards:
                            metadata, raw = target_shard.encode_frame(chunk, self.config.staging_budget_bytes)
                        else:
                            metadata, raw = encode(
                                ((item.name, item.shape, item.dtype, item.positions, item.values) for item in chunk),
                                max_bytes=self.config.staging_budget_bytes,
                            )
                        if workspace_peak() > self.config.gpu_budget_bytes:
                            raise ValueError("Measured source conversion/encoding workspace exceeds its GPU budget")
                    except Exception as exc:
                        error = exc
                agree(error)
                if self.sender_pipeline_active:
                    drain_previous_frame()
                error = None
                if rank == 0:
                    try:
                        if self.sender_pipeline_active:
                            constraints.update(self.transfer.submit_frame(metadata, raw))
                        else:
                            constraints.update(self.transfer.send(metadata, raw, defer_ordered=True))
                    except Exception as exc:
                        error = exc
                agree(error)
                if rank == 0:
                    del raw, metadata
                del chunk
            if self.sender_pipeline_active:
                drain_previous_frame()
            source_peaks = [None] * world
            dist.all_gather_object(source_peaks, workspace_peak(), group=group)
            status = [None]
            error = None
            if rank == 0:
                try:
                    if self.transfer.ordered_target_consume:
                        constraints.update(self.transfer.apply())
                    result = self.transfer.finalize()
                    status[0] = dict(
                        result,
                        full=seed,
                        version=version,
                        full_names=sorted(constraints),
                        force_full_seed=any(reply["force_full_seed"] for reply in result["receivers"]),
                        source_torch_workspace_peak_bytes=source_peaks,
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
            if rank == 0 and self.sender_pipeline_active and self.transfer is not None:
                try:
                    self.transfer.drain_frame()
                except Exception:
                    logger.exception("Pending TCP frame also failed during delta cleanup")
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

    def begin(self, local_shards: Mapping[str, torch.Tensor], *, live_locked: bool = False) -> None:
        if self._pending is not None:
            raise RuntimeError("A delta snapshot is already pending acknowledgement")
        acknowledged_bytes = (
            self._ack_bank.numel()
            if self._ack_bank is not None
            else sum(t.numel() * t.element_size() for t in self._acknowledged.values())
        )
        available = self.snapshot_budget_bytes - acknowledged_bytes
        if live_locked:
            # The synchronous Actor publication owns the model through commit/abort.
            # Keep the converter's mutable temporaries separate from these inputs.
            entries, size = shard_layout(local_shards.items())
            if self._spare_bank is not None and self._spare_bank.numel() != size:
                self._spare_bank = None
            self._backup = CudaShardBackup(entries, size, self._spare_bank, available, self.config.bucket_bytes)
            self._spare_bank = None
            self._pending = self._backup.targets
            self._live_inputs = CudaShardInputs(local_shards, next(iter(local_shards.values())).device)
        else:
            self._spare_bank = None
            if sum(t.numel() * t.element_size() for t in local_shards.values()) > available:
                raise ValueError("Acknowledged and pending local shard snapshots exceed their host budget")
            self._pending = {
                name: tensor.detach().to(device="cpu", copy=True) for name, tensor in local_shards.items()
            }
        self._attempt = object()
        self._stream = None
        self._exported = False
        if self._acknowledged and self._schema(self._pending) != self._schema(self._acknowledged):
            self._force_full = True

    @staticmethod
    def _schema(shards: Mapping[str, torch.Tensor]) -> tuple:
        return tuple((name, tuple(value.shape), value.dtype) for name, value in shards.items())

    def export(
        self, *, full: bool = False, full_names: frozenset[str] = frozenset(), emit: bool = True, _owner_group=None
    ) -> Iterator[list[DeltaWeight]]:
        if self._pending is None or self._exported or self._stream is not None:
            raise RuntimeError("Export requires a fresh pending snapshot")
        self._stream = self._export_pending(
            self._pending, self._attempt, full=full, full_names=full_names, emit=emit, metadata_group=_owner_group
        )
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
        metadata_group: dist.ProcessGroup | None,
    ) -> Iterator[list[DeltaWeight]]:
        if self._attempt is not attempt:
            raise RuntimeError("The delta export attempt is no longer active")
        full = full or self._force_full or not self._acknowledged
        payload_group = None
        owner_diff = self._target_shards or self._owner_local_diff or self.sender_pipeline_active
        if owner_diff and not full and metadata_group is not None:
            error = None
            selector = "_target_shard_group" if self._target_shards else "_owner_delta_group"
            try:
                choose = getattr(self.current_iterator, selector, None)
                payload_group = choose(metadata_group) if choose else None
                if (
                    payload_group is not None
                    and getattr(self.baseline_iterator, selector)(metadata_group) is not payload_group
                ):
                    payload_group = None
            except Exception as exc:
                error = exc
            # No rank enters the collective-bearing owner path alone, including
            # when a local converter rejects the group or its layout.
            states = [None] * dist.get_world_size(metadata_group)
            dist.all_gather_object(
                states, (payload_group is not None, str(error) if error else None), group=metadata_group
            )
            if any(message is not None for _, message in states):
                raise RuntimeError(f"Owner delta source eligibility failed: {states}") from error
            if not all(ok for ok, _ in states):
                payload_group = None
        current_shards = self._live_inputs if self.sender_pipeline_active else pending
        previous = self._acknowledged
        if self.sender_pipeline_active and not full:
            # Two slots share at most half the exporter GPU budget; the caller's
            # measured peak gate still covers conversion, encoding, and transport.
            limit = min(self.config.bucket_bytes, self.config.gpu_budget_bytes // 4)
            previous = CudaShardInputs(previous, self._live_inputs.device, limit)
        if payload_group is None:
            current = iter(self.current_iterator.get_hf_weight_chunks(current_shards))
            baseline = iter(()) if full else iter(self.baseline_iterator.get_hf_weight_chunks(previous))
        else:
            options = {"target_shards": True} if self._target_shards else {}
            current = iter(self.current_iterator._owner_delta_chunks(current_shards, metadata_group, **options))
            baseline = iter(self.baseline_iterator._owner_delta_chunks(previous, metadata_group, **options))
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
                if payload_group is not None and any(
                    isinstance(tensor, OwnerExpert) for chunk in (current_chunk, baseline_chunk) for _, tensor in chunk
                ):
                    yield self._encode_owner_chunk(
                        current_chunk,
                        baseline_chunk,
                        full_names,
                        payload_group,
                        metadata_group,
                        target_lanes=self._target_shards,
                    )
                else:
                    yield self._encode_chunk(
                        current_chunk, baseline_chunk, full=full, full_names=full_names, emit=emit
                    )
            if self._attempt is not attempt:
                raise RuntimeError("A retired export cannot complete a new pending snapshot")
        finally:
            # In particular, restore converters' cross-yield process-group scopes.
            try:
                for iterator in (current, baseline):
                    close = getattr(iterator, "close", None)
                    if close is not None:
                        close()
            finally:
                if self._live_inputs is not None:
                    compute = torch.cuda.current_stream(self._live_inputs.device)
                    if isinstance(previous, CudaShardInputs):
                        previous.join_prefetch(compute)
                    self._live_read_done = compute.record_event()
        # This final iterator advance is inside synchronize()'s existing
        # all-rank error agreement. A delayed copy failure must reach every
        # rank before receiver acknowledgement and the local bank swaps.
        if self._backup is not None:
            self._finish_live_reads()
        self._exported = True

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

    @classmethod
    def _encode_owner_chunk(cls, current, baseline, full_names, payload_group, metadata_group, *, target_lanes=False):
        """Diff whole owned slots, then gather just the selected representation
        to the publisher."""
        rank = dist.get_rank(payload_group)
        world = dist.get_world_size(payload_group)
        slots, ordinary, owned = [], [], []
        qualified = set()
        error = None
        # Counts are already known on the host after the existing diff. A negative
        # count denotes the unchanged dense threshold or a loader-required FULL.
        width = 5 if target_lanes else 1
        counts = torch.zeros(len(current) * width + 1, dtype=torch.int64, device="cpu")
        try:
            for i, ((name, new), (old_name, old)) in enumerate(zip(current, baseline, strict=True)):
                if (name, new.shape, new.dtype) != (old_name, old.shape, old.dtype):
                    raise ValueError("Current and baseline HF tensor schemas differ")
                slot = new if isinstance(new, OwnerExpert) else old
                if not isinstance(slot, OwnerExpert):
                    ordinary.append(i)
                    continue
                slots.append((i, slot))
                if isinstance(new, OwnerExpert) and isinstance(old, OwnerExpert):
                    if new.owner != old.owner:
                        raise ValueError("Current and baseline HF slot owners differ")
                    if target_lanes and target_shard.split_axis(name, slot.shape) is not None:
                        qualified.add(i)
                if slot.owner == rank:
                    owned.append(
                        (
                            i,
                            new.tensor if isinstance(new, OwnerExpert) else new,
                            old.tensor if isinstance(old, OwnerExpert) else old,
                        )
                    )
            ordinary_entries = cls._encode_chunk(
                [current[i] for i in ordinary],
                [baseline[i] for i in ordinary],
                full=False,
                full_names=full_names,
                emit=rank == 0,
            )
            local = cls._encode_chunk(
                [(current[i][0], new) for i, new, _ in owned],
                [(current[i][0], old) for i, _, old in owned],
                full=False,
                full_names=full_names,
                emit=True,
            )
            routed = target_shard.partition_owner_entries(owned, local, qualified) if qualified else {}
            for j, ((i, _, _), entry) in enumerate(zip(owned, local, strict=True)):
                counts[i * width] = -1 if entry.full else entry.values.numel()
                if i in routed:
                    per_lane, positions, values = routed[i]
                    counts[i * width + 1 : (i + 1) * width] = torch.tensor(per_lane)
                    local[j] = DeltaWeight(entry.name, entry.shape, entry.dtype, positions, values)
            device = slots[0][1].device
            positions = [entry.positions for entry in local if not entry.full]
            send_positions = torch.cat(positions) if positions else torch.empty(0, dtype=torch.int32, device=device)
            send_values = (
                torch.cat([entry.values for entry in local])
                if local
                else torch.empty(0, dtype=torch.bfloat16, device=device)
            )
        except Exception as exc:
            error = exc
            counts[-1] = 1
        dist.all_reduce(counts, group=metadata_group)
        sizes = counts.tolist()
        if sizes[-1]:
            raise RuntimeError("Owner delta comparison failed on a source rank") from error

        pos_sizes, value_sizes = [0] * world, [0] * world
        for i, slot in slots:
            count = sizes[i * width]
            value_sizes[slot.owner] += slot.nbytes // 2 if count < 0 else count
            pos_sizes[slot.owner] += max(count, 0)

        # Root allocation can fail independently of the owners. Agree before
        # any peer enters the first variable-size payload collective.
        try:
            pos = send_positions.new_empty(sum(pos_sizes) if rank == 0 else 0)
            vals = send_values.new_empty(sum(value_sizes) if rank == 0 else 0)
        except Exception as exc:
            error = exc
        failed = torch.tensor(int(error is not None), dtype=torch.int32, device="cpu")
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=metadata_group)
        if int(failed):
            raise RuntimeError("Cannot allocate owner delta gather buffers") from error
        for send, receive, lengths in ((send_positions, pos, pos_sizes), (send_values, vals, value_sizes)):
            if any(lengths):
                dist.all_to_all_single(
                    receive,
                    send,
                    output_split_sizes=lengths if rank == 0 else [0] * world,
                    input_split_sizes=[send.numel()] + [0] * (world - 1),
                    group=payload_group,
                )
        if rank != 0:
            return []
        pos_parts = pos.split(pos_sizes)
        val_parts = vals.split(value_sizes)
        pos_offset, val_offset = [0] * world, [0] * world
        result = [None] * len(current)
        for i, entry in zip(ordinary, ordinary_entries, strict=True):
            result[i] = entry
        for i, slot in slots:
            owner, count = slot.owner, sizes[i * width]
            nvalues = slot.nbytes // 2 if count < 0 else count
            positions = None if count < 0 else pos_parts[owner].narrow(0, pos_offset[owner], count)
            values = val_parts[owner].narrow(0, val_offset[owner], nvalues)
            name = current[i][0]
            lane_counts = None
            if i in qualified and count >= 0:
                lane_counts = tuple(sizes[i * width + 1 : (i + 1) * width])
                if sum(lane_counts) != count:
                    raise ValueError("Owner lane counts differ from the acknowledged sparse payload")
            result[i] = DeltaWeight(name, slot.shape, slot.dtype, positions, values, lane_counts)
            pos_offset[owner] += max(count, 0)
            val_offset[owner] += nvalues
        return result

    def commit(self) -> None:
        """Called once after the final common receiver completion gate."""
        if self._pending is None or not self._exported:
            raise RuntimeError("Cannot commit an incomplete or absent delta export")
        if self._backup is not None:
            self._spare_bank, self._ack_bank = self._ack_bank, self._backup.storage
        else:
            self._ack_bank = None
        self._backup = self._live_inputs = None
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

    def _finish_live_reads(self) -> None:
        try:
            self._backup.finish()
        finally:
            if self._live_read_done is not None:
                self._live_read_done.synchronize()
                self._live_read_done = None

    def abort(self) -> None:
        """Discard the failed target; keep the last acknowledged baseline."""
        self._attempt = None
        try:
            try:
                if self._stream is not None:
                    self._stream.close()
            finally:
                if self._backup is not None:
                    self._finish_live_reads()
                    self._spare_bank = self._backup.storage
        finally:
            self._backup = self._live_inputs = self._live_read_done = None
            self._stream = None
            self._pending = None
            self._exported = False
            self._force_full = True
            self._pending_version = None
            self._seed_after_commit = False
