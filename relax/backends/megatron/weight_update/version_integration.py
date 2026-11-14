# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Construct the independent publisher from the initialized training actor."""

import hashlib
from pathlib import Path

from relax.distributed.weight_sync import traffic
from relax.distributed.weight_sync.config import PublicationConfig
from relax.distributed.weight_sync.metrics import SyncMetrics
from relax.distributed.weight_sync.store import VersionStore
from relax.utils.logging_utils import get_logger

from .publish_weight_versions import MegatronVersionPublisher


def create_version_publisher(actor: object, process_group: object) -> MegatronVersionPublisher:
    import torch.distributed as dist

    rank = dist.get_rank(process_group)
    metrics = SyncMetrics("source:setup")
    metrics.source_participant = {"rank": rank, "world_size": dist.get_world_size(process_group)}
    config, store, error = None, None, None
    try:
        with traffic.capture(metrics):
            try:
                config = PublicationConfig.read(actor.args.weight_version_config)
                metrics.update_id = f"{config.stream}:setup"
                store = VersionStore(
                    config.store,
                    disk_budget=config.disk_budget,
                    max_weight_bytes=config.max_weight_bytes,
                    canonical_root=config.canonical_root,
                    canonical_budget=config.canonical_budget,
                    canonical_blocks=config.canonical_blocks,
                )
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
            failures = [None] * dist.get_world_size(process_group)
            traffic.all_gather_object(failures, error, group=process_group)
            if any(failures):
                raise RuntimeError(f"Source publication setup failed: {failures}")
            publisher = _create_version_publisher(actor, process_group, config, store)
            metrics.update_id = f"{config.stream}:{publisher.source_epoch}:setup"
            return publisher
    finally:
        if config is not None:
            try:
                metrics.save(Path(config.scratch) / f"setup-metrics-{metrics.operation_id}-rank-{rank}.json")
            except OSError:
                get_logger(__name__).warning("Publisher setup diagnostic output failed", exc_info=True)


def _create_version_publisher(actor, process_group, config, store):
    from .common import named_params_and_buffers
    from .hf_weight_iterator_base import HfWeightIteratorBase

    source_root = Path(__file__).resolve().parent.parent
    fingerprint = hashlib.sha256()
    for directory in (source_root / "weight_conversion", source_root / "weight_update"):
        for source in sorted(directory.rglob("*.py")):
            fingerprint.update(source.relative_to(source_root).as_posix().encode())
            fingerprint.update(source.read_bytes())

    # Export the logical unquantized checkpoint. Inference quantization belongs
    # to the target Loader, as in a normal checkpoint startup.
    def agreement(error):
        import torch.distributed as dist

        failures = [None] * dist.get_world_size(process_group)
        traffic.all_gather_object(failures, error, group=process_group)
        if any(failures):
            raise RuntimeError(f"Bridge initialization failed: {failures}")

    extra = {"publication_agreement": agreement} if actor.args.megatron_to_hf_mode == "bridge" else {}
    iterator = HfWeightIteratorBase.create(
        actor.args,
        actor.model,
        model_name=actor.args.model_name or type(actor.hf_config).__name__.lower(),
        quantization_config=None,
        **extra,
    )

    def weights() -> dict:
        if hasattr(actor, "weights_backuper"):
            return actor.weights_backuper.get("actor")
        return dict(
            named_params_and_buffers(
                actor.args,
                actor.model,
                convert_to_global_name=actor.args.megatron_to_hf_mode == "raw",
                include_persistent_buffers=actor.args.megatron_to_hf_mode == "bridge",
            )
        )

    return MegatronVersionPublisher(
        iterator,
        weights,
        process_group,
        store,
        config.stream,
        config.scratch,
        model_config=actor.hf_config.to_dict(),
        converter_revision=f"relax-{actor.args.megatron_to_hf_mode}-{type(iterator).__name__}:{fingerprint.hexdigest()}",
        compression=config.compression,
        encode_workers=config.encode_workers,
        anchor_interval=config.anchor_interval,
        full_only=config.full_only,
        source_cache_root=config.source_cache_root,
        source_limits=config.source_limits,
        source_io_timeout=config.source_io_timeout,
    )
