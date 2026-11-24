# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Acknowledged training-shard snapshots and verl sparse HF export.

The native FULL export defines the SGLang loader directory. In steady state
Bridge projects each contributor's local raw-bit changes to those slots.
Transport receives only final HF coordinates and does not interpret Megatron
layouts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from relax.distributed.weight_sync.config import RETAINED_DIRECTORY_BYTES, DeltaSyncConfig, supports_delta_layout
from relax.utils.logging_utils import get_logger

from .shard_snapshot import (
    CudaShardBackup,
    CudaShardInputs,
    PinnedShardBank,
    preallocate_spare_bank,
    raw_shard,
    shard_bank_layout,
)


logger = get_logger(__name__)


def delta_sync_config(args: Any, quantization_config: Any) -> DeltaSyncConfig | None:
    path = getattr(args, "memory_weight_sync_config", None)
    if not path:
        return None
    config = DeltaSyncConfig.load(path)
    if getattr(args, "megatron_to_hf_mode", "bridge") != "bridge":
        logger.info("The selected converter uses its native FULL publication")
        return None
    # These change the actual exported/loaded state contract. Keep their
    # existing FULL implementation rather than banning an execution mode.
    incompatible = bool(
        quantization_config
        or getattr(args, "use_lora", False)
        or getattr(args, "lora_rank", 0)
        or getattr(args, "sglang_speculative_algorithm", None)
        or getattr(args, "mtp_num_layers", 0)
        or getattr(args, "rollout_external", False)
        or getattr(args, "sglang_dp_size", 1) != 1
        or getattr(args, "rollout_num_gpus_per_engine", 1) > getattr(args, "num_gpus_per_node", 8)
    )
    if incompatible:
        logger.info(
            "Delta export retains original FULL for quantized, adapter, draft or external/multinode/inference-DP state"
        )
        return None
    if getattr(args, "sglang_config", None) and (
        getattr(args, "fully_async", False) or getattr(args, "hybrid", False)
    ):
        from relax.distributed.ray.rollout import _resolve_sglang_config

        # DCS publishes to all registered models. The delta handle lookup
        # addresses the first model, so keep native DCS for multi-model YAML.
        if len(_resolve_sglang_config(args).models) > 1:
            logger.info("Multi-model DCS publication retains original FULL transport")
            return None
    return config


def create_delta_exporter(args: Any, model: Any, *, model_name: str, quantization_config: Any) -> Any:
    config = delta_sync_config(args, quantization_config)
    if config is None:
        return None
    # Rank zero may hold two pipelined host slots or a CPU tensor and its
    # immutable bytes on the mixed NCCL/TCP fallback.
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
    """Keep one pending local snapshot until every recipient acknowledges
    publication."""

    def __init__(
        self,
        current_iterator: Any,
        *,
        snapshot_budget_bytes: int,
        config: DeltaSyncConfig | None = None,
    ) -> None:
        if type(snapshot_budget_bytes) is not int or snapshot_budget_bytes <= 0:
            raise ValueError("The local shard snapshot budget must be a positive integer")
        from .sharded_source import BridgeShardSource

        self.sharded = BridgeShardSource(current_iterator)
        self._in_publication = False
        self.recipients_confirmed = True
        self.snapshot_budget_bytes = snapshot_budget_bytes
        self._acknowledged: dict[str, torch.Tensor] = {}
        self._pending: dict[str, torch.Tensor] | None = None
        self._live_inputs: CudaShardInputs | None = None
        self._backup: CudaShardBackup | None = None
        self._live_read_done: torch.cuda.Event | None = None
        self._ack_bank: PinnedShardBank | None = None
        self._spare_bank: PinnedShardBank | None = None
        self._supports_live_shards = False
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

        config = kwargs.get("config")
        if config is not None:
            args = copy(args)
            args.update_weight_buffer_size = min(args.update_weight_buffer_size, config.bucket_bytes)
        current = HfWeightIteratorBase.create(
            args, model, model_name=model_name, quantization_config=quantization_config
        )
        exporter = cls(current, **kwargs)
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
                        supports_delta_layout(layout, count) for layout, count in zip(layouts, gpu_counts, strict=True)
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
        from relax.distributed.weight_sync.codec import WireBucket, encode
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

        # Both production callers pass their Gloo metadata group. Keep control
        # on the host there, even when the conversion itself runs on CUDA.
        control_device = "cuda" if world > 1 and dist.get_backend(group) == "nccl" else "cpu"

        def reduce_control(error: Exception | None, count: bool = False, pause: bool = False) -> tuple[int, bool]:
            failed, total, needs_pause = int(error is not None), int(count), int(pause)
            if world > 1:
                state = torch.tensor([failed, total, needs_pause], dtype=torch.int32, device=control_device)
                dist.all_reduce(state, op=dist.ReduceOp.SUM, group=group)
                failed, total, needs_pause = state.tolist()
            if failed:
                agree(error)
            return total, bool(needs_pause)

        warm_pool = warm_spare = None
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
                shards = weights_getter()
                live = bool(getattr(self.config, "sender_pipeline", False)) and self._can_read_live(shards)
            except Exception as exc:
                error = exc
            if getattr(self.config, "sender_pipeline", False):
                agree(error)
                eligible = [None] * world
                dist.all_gather_object(eligible, live, group=group)
                live = all(eligible)
            cold_banks = self._ack_bank is None and not self._acknowledged and self._spare_bank is None
            if error is None:
                try:
                    self.begin(shards, live_locked=live)
                    if self._force_full and not seed:
                        raise ValueError("Source shard schema changed during an acknowledged delta stream")
                except Exception as exc:
                    error = exc
            agree(error)
            first_live_seed = seed and live and self._sequence == 0
            cold_socket_seed = first_live_seed and self.config.transport == "nccl_socket"
            after_submit = [False]
            error = None
            if rank == 0:
                try:
                    self.transfer.begin(version=version, base=self._version, seed=seed, defer_apply=not applying)
                    if cold_socket_seed:
                        after_submit[0] = changed and self.transfer.pipelined_socket_install
                except Exception as exc:
                    error = exc
            agree(error)
            if cold_socket_seed:
                dist.broadcast_object_list(after_submit, src=source_rank, group=group)
            defer_spare = after_submit[0]

            def start_spare() -> None:
                nonlocal warm_pool, warm_spare
                error = None
                try:
                    size = self._backup.storage.nbytes
                    if cold_banks and 0 < size <= self.snapshot_budget_bytes // 2:
                        warm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shard-spare")
                        warm_spare = warm_pool.submit(
                            preallocate_spare_bank, self._backup.storage.sizes, self._live_inputs.device
                        )
                except Exception as exc:
                    error = exc
                agree(error)

            if first_live_seed and not defer_spare:
                start_spare()
            if not overlap:
                enter_apply()

            stream = self.export(full=seed, full_names=frozenset(self.full_names), emit=rank == 0, _owner_group=group)
            pack = self.config.transport == "nccl_socket"
            bucket = (
                WireBucket(min(self.config.staging_budget_bytes, self.config.gpu_budget_bytes // 8)) if pack else None
            )
            while True:
                error, chunk = None, None
                try:
                    chunk = next(stream, None)
                    if workspace_peak() > self.config.gpu_budget_bytes:
                        raise ValueError("Measured source conversion workspace exceeds its configured GPU budget")
                except Exception as exc:
                    error = exc
                finished, _ = reduce_control(error, chunk is None)
                if finished:
                    if finished != world:
                        raise ValueError("Source converters disagree on the complete HF group directory")
                    if not pack:
                        break
                error, frame, needs_pause = None, None, False
                if rank == 0:
                    try:
                        if chunk is None:
                            frame = bucket.finish()
                        else:
                            frame = encode(
                                ((item.name, item.shape, item.dtype, item.positions, item.values) for item in chunk),
                                max_bytes=self.config.staging_budget_bytes,
                            )
                            if pack:
                                frame = bucket.offer(*frame)
                        if workspace_peak() > self.config.gpu_budget_bytes:
                            raise ValueError("Measured source conversion/encoding workspace exceeds its GPU budget")
                        if frame is not None:
                            metadata, raw = frame
                            if self.sender_pipeline_active:
                                self.transfer.prepare_frame(raw)
                                constraints.update(self.transfer.drain_frame())
                            if not applying:
                                needs_pause = not self.transfer.can_stage(metadata, raw)
                    except Exception as exc:
                        error = exc
                # Only the publisher encodes, drains the previous ACK, and checks admission.
                # Every rank learns the resulting frame/pause decision in the same collective.
                available, needs_pause = reduce_control(error, frame is not None, needs_pause)
                if not available:
                    if chunk is None:
                        break
                    continue
                if needs_pause:
                    enter_apply()
                error = None
                if rank == 0:
                    try:
                        if self.sender_pipeline_active:
                            constraints.update(self.transfer.submit_frame(metadata, raw, applying=applying))
                        elif applying:
                            constraints.update(self.transfer.send(metadata, raw))
                        else:
                            self.transfer.stage(metadata, raw)
                    except Exception as exc:
                        error = exc
                reduce_control(error)
                if defer_spare:
                    # The FULL bucket has fenced its broadcast and handed the
                    # ordinary receiver ACK to the install worker on rank zero.
                    start_spare()
                    defer_spare = False
                if rank == 0:
                    del raw, metadata
                if chunk is None:
                    break
                del frame, chunk
            if defer_spare:
                # A converter can finish without yielding any wire frame.
                start_spare()
            if self.sender_pipeline_active:
                # The final submitted frame must acknowledge before we pause or finalize.
                error = None
                if rank == 0:
                    try:
                        constraints.update(self.transfer.drain_frame())
                    except Exception as exc:
                        error = exc
                agree(error)
            enter_apply()
            if first_live_seed:
                try:
                    if warm_spare is not None:
                        self._spare_bank = warm_spare.result()
                except Exception as exc:
                    error = exc
                agree(error)
            source_peaks = [None] * world
            dist.all_gather_object(source_peaks, (workspace_peak(), self.sharded.metrics), group=group)
            status = [None]
            error = None
            if rank == 0:
                try:
                    result = self.transfer.finalize()
                    constraints.update(result.pop("final_full_names"))
                    status[0] = dict(
                        result,
                        full=seed,
                        version=version,
                        full_names=sorted(constraints),
                        force_full_seed=any(reply["force_full_seed"] for reply in result["receivers"]),
                        source_torch_workspace_peak_bytes=[peak for peak, _ in source_peaks],
                        receive_overlap_requested=self.config.receive_overlap,
                        receive_overlap_eligible=overlap,
                        shard_gather=[metrics for _, metrics in source_peaks],
                    )
                except Exception as exc:
                    error = exc
            agree(error)
            dist.broadcast_object_list(status, src=source_rank, group=group)
            if seed:
                self.full_names.clear()
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
                try:
                    self.transfer.discard_prepared()
                except Exception:
                    logger.exception("Unsubmitted TCP frame also failed during delta cleanup")
            self.abort()
            raise
        finally:
            if warm_pool is not None:
                warm_pool.shutdown(wait=True)

    def invalidate(self) -> None:
        """Recipient recreation or weight residency loss requires a FULL
        seed."""
        if self._pending is not None:
            raise RuntimeError("Cannot invalidate an in-flight delta export")
        self._force_full = True

    def disconnect(self, *, group: dist.ProcessGroup) -> None:
        """Join the original trainer sleep boundary after a completed
        update."""
        pending = self._pending is not None
        if pending and not (self._in_publication and self._exported and self._pending_version is not None):
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
        if pending:
            self._seed_after_commit = True
        else:
            self.invalidate()

    def begin(self, local_shards: Mapping[str, torch.Tensor], *, live_locked: bool = False) -> None:
        if self._pending is not None:
            raise RuntimeError("A delta snapshot is already pending acknowledgement")
        acknowledged_bytes = (
            self._ack_bank.nbytes
            if self._ack_bank is not None
            else sum(t.numel() * t.element_size() for t in self._acknowledged.values())
        )
        available = self.snapshot_budget_bytes - acknowledged_bytes
        if live_locked:
            # The synchronous Actor publication owns the model through commit/abort.
            # Keep the converter's mutable temporaries separate from these inputs.
            entries, size = shard_bank_layout(local_shards.items(), self.config.bucket_bytes)
            if self._spare_bank is not None and self._spare_bank.sizes != tuple(slab.capacity for slab in entries):
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
                name: raw_shard(tensor).to(device="cpu", copy=True).view(tensor.dtype)
                for name, tensor in local_shards.items()
            }
        self._attempt = object()
        self._stream = None
        self._exported = False
        if not self._force_full and self._schema(self._pending) != self._schema(self._acknowledged):
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
            self._attempt, full=full, full_names=full_names, emit=emit, metadata_group=_owner_group
        )
        return self._stream

    @torch.no_grad()
    def _export_pending(
        self,
        attempt: object,
        *,
        full: bool,
        full_names: frozenset[str],
        emit: bool,
        metadata_group: dist.ProcessGroup | None,
    ) -> Iterator[list[DeltaWeight]]:
        if self._attempt is not attempt:
            raise RuntimeError("The delta export attempt is no longer active")
        if metadata_group is None:
            raise RuntimeError("Sharded Bridge export requires the Actor metadata group")
        full = full or self._force_full
        from relax.utils.device import make_current_torch_device

        current = self._live_inputs if self.sender_pipeline_active else self._pending
        previous = self._acknowledged
        if self.sender_pipeline_active and not full:
            limit = min(self.config.bucket_bytes, self.config.gpu_budget_bytes // 4)
            previous = CudaShardInputs(previous, self._live_inputs.device, limit, bank=self._ack_bank)
        stream = (
            self.sharded.seed(current)
            if full
            else self.sharded.chunks(
                current,
                previous,
                full_names=full_names,
                metadata_group=metadata_group,
                device=make_current_torch_device(),
                max_round_bytes=self.config.bucket_bytes,
                staging_budget_bytes=self.config.staging_budget_bytes,
            )
        )
        try:
            for entries in stream:
                if full and emit:
                    yield [
                        DeltaWeight(
                            name, tuple(value.shape), value.dtype, None, value.detach().contiguous().reshape(-1)
                        )
                        for name, value in entries
                    ]
                else:
                    yield entries if emit else []
            if full:
                self.sharded.prepare(metadata_group)
        finally:
            stream.close()
            if self.sender_pipeline_active:
                compute = torch.cuda.current_stream(self._live_inputs.device)
                if isinstance(previous, CudaShardInputs):
                    previous.join_prefetch(compute)
                self._live_read_done = compute.record_event()
        if self._backup is not None:
            self._finish_live_reads()
        self._exported = True

    @contextmanager
    def publication(self, group: dist.ProcessGroup):
        """Keep the previous bank until the outer Actor call completes on every
        rank."""
        if self._in_publication:
            yield
            return
        self._in_publication = True
        self.recipients_confirmed = True
        try:
            error = None
            try:
                yield
            except Exception as exc:
                error = exc
            reports = [None] * dist.get_world_size(group)
            dist.all_gather_object(
                reports,
                (str(error) if error is not None else None, self._pending is not None, self.recipients_confirmed),
                group=group,
            )
            if error is not None:
                raise error
            if (
                any(message is not None for message, _, _ in reports)
                or len({pending for _, pending, _ in reports}) != 1
            ):
                raise RuntimeError(f"Delta Actor publication did not complete on all source ranks: {reports}")
            if self._pending is not None:
                if all(confirmed for _, _, confirmed in reports):
                    self.commit()
                else:
                    # The original Actor retries router registration on its next
                    # publication. A partially visible version needs a new seed.
                    self.abort()
        finally:
            self._in_publication = False
            if self._pending is not None:
                self.abort()

    def finish_delivery(self) -> None:
        """Standalone backends finish here; the Actor also owns later lifecycle
        steps."""
        if not self._in_publication:
            self.commit()

    def commit(self) -> None:
        """Called once after the final common receiver completion gate."""
        if self._pending is None or not self._exported:
            raise RuntimeError("Cannot commit an incomplete or absent delta export")
        if self._backup is not None:
            if self._ack_bank is not None:
                self._spare_bank = self._ack_bank
            self._ack_bank = self._backup.storage
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
