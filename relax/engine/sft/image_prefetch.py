# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""CPU image reconstruction and bounded cache for SFT prefetch."""

from argparse import Namespace
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist

from relax.utils.data.image_refs import SFT_IMAGE_REFS_FIELD, extract_image_ref_descriptors
from relax.utils.distributed_utils import get_gloo_group
from relax.utils.logging_utils import get_logger
from relax.utils.timer import Timer


if TYPE_CHECKING:
    from relax.utils.data.processor_pool import ProcessorPool


logger = get_logger(__name__)

# Samples of one prefetched batch are rebuilt in parallel — one processor call
# per sample (a sample carries ~20 images), submitted together by
# ``build_batch_image_features``. The vision rank is the prefetch's tail, so
# size the pool like the producer's ``--sft-prefetch-num-workers`` instead of a
# single worker: only one vision rank per node rebuilds (VPP-aware
# ``pre_process`` check), and its workers cap their own torch intra-op threads.
_REBUILD_POOL_SIZE = 8


def _raise_if_sft_image_resolution_error(local_error: Exception | None, *, rollout_id: int) -> None:
    """Propagate rank-local image rebuild failures across the actor world."""
    error_flag = torch.tensor([int(local_error is not None)], dtype=torch.int32, device="cpu")
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX, group=get_gloo_group())
    if int(error_flag.item()) != 0:
        if local_error is not None:
            raise local_error
        raise RuntimeError(
            f"SFT rank-side image resolution failed on a peer rank for rollout_id={rollout_id}; "
            "aborting before model collectives."
        )


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
    ) -> tuple[list, float]:
        """Background SFT prefetch with the rank-side pixel rebuild.

        Extends the CPU-only TQ read of ``--sft-train-data-prefetch`` with the
        ``--sft-image-preprocess-on-rank`` reconstruction: the payload carries
        image references instead of pixel tensors, so this thread reads the
        referenced files and rebuilds CPU ``pixel_values`` in the processor
        worker process. No collectives and no GPU work happen here; the main
        thread re-attaches the finished tensors in ``resolve``.
        """
        prefetched = self.fetch_data(
            data_fields=data_fields,
            batch_size=batch_size,
            partition_id=partition_id,
            task_name=task_name,
            sampling_config=sampling_config,
            batch_index=0,
        )
        raw_batch = prefetched[0][0]
        if raw_batch is None or not self.owns_vision(model):
            return prefetched
        descriptors = extract_image_ref_descriptors(raw_batch)
        if descriptors is None:
            return prefetched
        from relax.utils.data.image_rebuild import build_batch_image_features

        features = build_batch_image_features(self._ensure_pool(), descriptors)
        # Keep only the newest entry so the stash stays bounded by one batch;
        # stale prefetches are discarded in _take_sft_train_prefetch.
        self.features = {rollout_id: features}
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
        """Re-attach rank-rebuilt CPU pixels before the batch reaches
        get_batch.

        The descriptor field is dropped on every rank. Vision ranks re-attach
        the ``pixel_values`` rebuilt during prefetch — falling back to a
        synchronous rebuild when prefetch was paused (eval / checkpoint
        lookahead) — and verify the rebuilt grid against the producer's. The
        producer tokens, masks and grid are never replaced.
        """
        descriptors = rollout_data.pop(SFT_IMAGE_REFS_FIELD, None)
        if descriptors is None:
            return
        mm_inputs = rollout_data.get("multimodal_train_inputs")
        if mm_inputs is None:
            raise RuntimeError(
                f"rollout_id={rollout_id}: image-ref descriptors present but multimodal_train_inputs is missing."
            )
        owns_vision = self.owns_vision(model)
        features_by_sample, timings = self.features.pop(rollout_id, (None, {})) if owns_vision else (None, {})
        for name, elapsed in timings.items():
            Timer().add(name, elapsed)
        resolved_samples = 0
        pixel_bytes = 0
        for index, descriptor in enumerate(descriptors):
            if descriptor is None:
                continue
            if not owns_vision:
                continue
            features = None
            if features_by_sample is not None and index < len(features_by_sample):
                features = features_by_sample[index]
            if features is None:
                from relax.utils.data.image_rebuild import build_batch_image_features

                rebuilt, timings = build_batch_image_features(self._ensure_pool(), [descriptor])
                features = rebuilt[0]
                for name, elapsed in timings.items():
                    Timer().add(name, elapsed)
            mm = mm_inputs[index]
            if mm is None:
                raise RuntimeError(f"rollout_id={rollout_id} sample {index}: image references without grid metadata.")
            rebuilt_grid = features.get("image_grid_thw")
            producer_grid = mm.get("image_grid_thw")
            if rebuilt_grid is not None and producer_grid is not None:
                # Cheap in-memory re-check; the authoritative comparison against
                # the descriptor grid already ran in the prefetch worker.
                if torch.as_tensor(producer_grid).tolist() != torch.as_tensor(rebuilt_grid).tolist():
                    raise ValueError(
                        f"rollout_id={rollout_id} sample {index}: rank-rebuilt image grid does not match the "
                        "producer grid; the consumer processor config diverged from the producer."
                    )
            mm_inputs[index] = {**mm, "pixel_values": features["pixel_values"]}
            resolved_samples += 1
            pixel_bytes += features["pixel_values"].numel() * features["pixel_values"].element_size()
        if resolved_samples and log:
            logger.info(
                "Rank-side image rebuild for rollout_id=%d: %d sample(s), %.1f MiB of CPU pixel tensors.",
                rollout_id,
                resolved_samples,
                pixel_bytes / 1024 / 1024,
            )
