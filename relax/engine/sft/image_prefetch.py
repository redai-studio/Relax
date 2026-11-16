# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Bounded CPU image decoding and rank-side reconstruction for SFT."""

from __future__ import annotations

from argparse import Namespace
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, current_thread, main_thread
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist

from relax.engine.sft.image import ImageBatchSampler, assign_image_batch
from relax.utils.data.image_refs import SFT_IMAGE_REFS_FIELD
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger
from relax.utils.timer import Timer


if TYPE_CHECKING:
    from relax.utils.data.processor_pool import ProcessorPool


logger = get_logger(__name__)


class ImagePrefetch:
    def __init__(
        self,
        dataset: Any,
        sampler: ImageBatchSampler,
        rank: int,
        world: int,
        start: int,
        stop: int,
        depth: int = 4,
        workers: int = 2,
    ) -> None:
        if depth <= 0 or workers <= 0 or not 0 <= rank < world or start < 0 or stop < start:
            raise ValueError("Invalid image prefetch bounds")
        if sampler.batch_size % world:
            raise ValueError("Image prefetch requires equal samples per rank")
        self.dataset = dataset
        # Do not advance the trainer's mutable epoch cache.
        self.sampler = ImageBatchSampler(sampler.size, sampler.batch_size, sampler.seed)
        self.rank, self.world, self.stop = rank, world, stop
        self.step, self.micro = start, 0
        self.depth = depth
        self.pending: deque[tuple[int, int, int, Future]] = deque()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="image-prefetch")
        self.closed = False
        self._fill()

    def _fill(self) -> None:
        while not self.closed and len(self.pending) < self.depth and self.step < self.stop:
            ids = assign_image_batch(
                self.sampler.batch(self.step), getattr(self.dataset, "config", {}), step=self.step, world=self.world
            )[self.rank :: self.world]
            sample_id = ids[self.micro]
            future = self.pool.submit(self.dataset.__getitem__, sample_id)
            self.pending.append((self.step, self.micro, sample_id, future))
            self.micro += 1
            if self.micro == len(ids):
                self.step, self.micro = self.step + 1, 0

    def get(self, step: int, micro: int, sample_id: int) -> Any:
        if self.closed or not self.pending:
            raise ValueError("Image prefetch is closed or exhausted")
        expected = self.pending[0]
        if expected[:3] != (step, micro, sample_id):
            raise ValueError("Image prefetch cursor differs from the verified training batch")
        self.pending.popleft()
        try:
            sample = expected[3].result()
        except BaseException:
            self.close()
            raise
        self._fill()
        return sample

    def close(self) -> None:
        # Cancel queued reads. Python cannot interrupt an already running NFS
        # read; process termination remains the fallback for a hung filesystem.
        self.closed = True
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.pending.clear()


# Only samples selected for the current microbatch reach this bounded pool;
# retaining a complete rollout of decoded multi-image samples exhausts host RAM.
# Each worker limits its own torch intra-op threads.
_REBUILD_POOL_SIZE = 8


def _raise_if_sft_image_resolution_error(local_error: Exception | None, *, rollout_id: int) -> None:
    """Propagate rollout-level image metadata failures across the actor
    world."""
    error_flag = torch.tensor([int(local_error is not None)], dtype=torch.int32, device="cpu")
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX, group=get_gloo_group())
    if int(error_flag.item()) != 0:
        if local_error is not None:
            raise local_error
        raise RuntimeError(
            f"SFT rank-side image resolution failed on a peer rank for rollout_id={rollout_id}; "
            "aborting before model collectives."
        )


class LazySFTImageInput(dict):
    """Keep rollout-sized image payloads as metadata until one microbatch
    consumes them.

    Decoded tensors are returned in a fresh dictionary, never cached on this
    object or its owning rollout. Iterator replay can reconstruct them again;
    pipeline/recompute lifetimes remain owned by the resulting model tensors.
    """

    def __init__(
        self, metadata: dict, descriptor: dict, owner: "SFTImagePrefetch", rollout_id: int, sample_index: int
    ) -> None:
        super().__init__(metadata)
        self.descriptor = descriptor
        self.owner = owner
        self.rollout_id = rollout_id
        self.sample_index = sample_index

    def materialize(self) -> dict:
        from relax.utils.data.image_rebuild import build_batch_image_features

        try:
            rebuilt, timings = build_batch_image_features(self.owner._ensure_pool(), [self.descriptor])
            features = rebuilt[0]
            if features is None:
                raise ValueError("image descriptor did not produce pixel tensors")
            producer_grid = self.get("image_grid_thw")
            rebuilt_grid = features.get("image_grid_thw")
            if producer_grid is not None and rebuilt_grid is not None:
                if torch.as_tensor(producer_grid).tolist() != torch.as_tensor(rebuilt_grid).tolist():
                    raise ValueError(
                        "rank-rebuilt image grid does not match the producer grid; "
                        "the consumer processor config diverged from the producer"
                    )
            # CPU prepacking may call this helper from a background thread.
            # Its timers must not mutate the training thread's Timer state.
            if current_thread() is main_thread():
                for name, elapsed in timings.items():
                    Timer().add(name, elapsed)
            return {**self, "pixel_values": features["pixel_values"]}
        except Exception as exc:
            raise RuntimeError(
                f"rollout_id={self.rollout_id} sample {self.sample_index}: "
                f"rank-side microbatch image reconstruction failed: {exc}"
            ) from exc


def resolve_sft_image_micro_batch(batch: dict) -> None:
    """Decode only this selected microbatch without retaining pixels in the
    rollout.

    No distributed collectives belong here: pipeline stages request different
    microbatches at different times. Resolution errors retain their sample
    context and propagate through the training task's ordinary error handling.
    """
    inputs = batch.get("multimodal_train_inputs")
    if inputs is not None and any(isinstance(value, LazySFTImageInput) for value in inputs):
        batch["multimodal_train_inputs"] = [
            value.materialize() if isinstance(value, LazySFTImageInput) else value for value in inputs
        ]


class SFTImagePrefetch:
    def __init__(self, args: Namespace, fetch_data: Callable[..., tuple[list, float]]) -> None:
        self.args = args
        self.fetch_data = fetch_data
        self.pool: ProcessorPool | None = None
        self.features: dict[int, tuple[list, dict[str, float]]] = {}

    def discard(self, rollout_id: int) -> None:
        """Drop a completed stale prefetch after its future has been joined."""
        self.features.pop(rollout_id, None)

    def close(self) -> None:
        """Release image resources after the fetch executor has stopped."""
        pool, self.pool = self.pool, None
        self.features.clear()
        if pool is not None:
            pool.shutdown(wait=True)

    def owns_vision(self, model: Any) -> bool:
        """Whether a local model chunk builds the vision tower.

        Decided from the chunks' own ``pre_process`` capability (VPP-aware)
        rather than a fixed PP rank: non-vision PP stages keep only the small
        multimodal metadata and never rebuild pixels.
        """
        if not getattr(self.args, "is_vl_model", False):
            return False
        chunks = model if isinstance(model, list) else [model]
        for chunk in chunks:
            module = chunk
            for _ in range(3):  # DDP / Float16Module wrappers
                if module is None:
                    break
                if getattr(module, "pre_process", False):
                    return True
                module = getattr(module, "module", None)
        return False

    def _ensure_pool(self) -> "ProcessorPool":
        """Pool of rebuild workers for the vision ranks; each loads the
        processor once."""
        if self.pool is None:
            from relax.utils.data.processor_pool import ProcessorPool
            from relax.utils.multimodal.config import MultimodalConfig

            self.pool = ProcessorPool(
                self.args.hf_checkpoint,
                pool_size=_REBUILD_POOL_SIZE,
                trust_remote_code=True,
                # Same image-token limits as the producer side, so the rebuilt
                # grid matches the descriptor grid exactly.
                multimodal_config=MultimodalConfig.from_args(self.args),
            )
        return self.pool

    def fetch(
        self,
        rollout_id: int,
        *,
        model: Any,
        data_fields: list[str],
        batch_size: int,
        partition_id: str,
        task_name: str,
        sampling_config: dict,
        cancel_event: Event | None = None,
    ) -> tuple[list, float]:
        """Prefetch CPU tokens, image references and grids, without decoding
        pixels.

        A full DP-local rollout can contain thousands of multi-image samples.
        Pixel reconstruction belongs to microbatch consumption, not this
        rollout-sized lookahead slot.
        """
        fetch_kwargs = dict(
            data_fields=data_fields,
            batch_size=batch_size,
            partition_id=partition_id,
            task_name=task_name,
            sampling_config=sampling_config,
            batch_index=0,
        )
        if cancel_event is None:
            prefetched = self.fetch_data(**fetch_kwargs)
        else:
            from relax.utils.data.stream_dataloader import _prefetch_data_from_transfer_queue

            prefetched = _prefetch_data_from_transfer_queue(
                tq_client=None, fetch_data=self.fetch_data, cancel_event=cancel_event, **fetch_kwargs
            )
        return prefetched

    def resolve_across_ranks(
        self, rollout_data: dict | None, rollout_id: int, *, model: Any, log: bool = False
    ) -> bool:
        """Resolve one available batch and propagate failures actor-wide.

        A DP replica that still has no batch returns immediately and keeps
        polling. Replicas with data wait in the world-wide Gloo reduction until
        every slower DP replica obtains its cached TQ slice and joins once.
        """
        if rollout_data is None:
            return False
        local_error: Exception | None = None
        try:
            self.resolve(rollout_data, rollout_id, model=model, log=log)
        except Exception as exc:  # noqa: BLE001
            local_error = exc
        _raise_if_sft_image_resolution_error(local_error, rollout_id=rollout_id)
        return True

    def resolve(self, rollout_data: dict, rollout_id: int, *, model: Any, log: bool = False) -> None:
        """Attach lazy image inputs, preserving the producer's tokens and
        grids."""
        descriptors = rollout_data.pop(SFT_IMAGE_REFS_FIELD, None)
        if descriptors is None:
            return
        mm_inputs = rollout_data.get("multimodal_train_inputs")
        if mm_inputs is None:
            raise RuntimeError(
                f"rollout_id={rollout_id}: image-ref descriptors present but multimodal_train_inputs is missing."
            )
        owns_vision = self.owns_vision(model)
        self.features.pop(rollout_id, None)
        if len(mm_inputs) != len(descriptors):
            raise ValueError(f"rollout_id={rollout_id}: image reference and multimodal input counts differ")
        resolved_samples = 0
        for index, descriptor in enumerate(descriptors):
            if descriptor is None:
                continue
            if not owns_vision:
                continue
            mm = mm_inputs[index]
            if mm is None:
                raise RuntimeError(f"rollout_id={rollout_id} sample {index}: image references without grid metadata.")
            mm_inputs[index] = LazySFTImageInput(mm, descriptor, self, rollout_id, index)
            resolved_samples += 1
        if resolved_samples and log:
            logger.info(
                "Prepared lazy rank-side image rebuild for rollout_id=%d: %d sample(s); "
                "pixels are decoded only for each consumed microbatch.",
                rollout_id,
                resolved_samples,
            )
